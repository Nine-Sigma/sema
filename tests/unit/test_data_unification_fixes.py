"""Regression tests for the data-unification adversarial-review fixes.

Covers:
  F1  composite Term embedding match + allowlist guard
  F2  source_schema resolution in materialize_unified (no null-scoped edges)
  F3  idempotent assertion writes + Assertion.id uniqueness constraint
  #2  MEMBER_OF links to the column_ref-keyed ValueSet, not a name-keyed one
  #3  value-set retrieval is schema-scoped and surfaces vocabulary_name
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.unit

from sema.graph.embedding_match import (
    EMBEDDING_MATCH_KEYS,
    validate_embedding_label,
    validate_match_props,
)
from sema.graph.loader import GraphLoader, _ASSERTION_UPSERT_CYPHER
from sema.graph.materializer import _resolve_source_schema, materialize_unified
from sema.graph.queries import CypherQueries
from sema.models.assertions import (
    Assertion,
    AssertionPredicate,
    AssertionStatus,
)


def _assertion(
    subject_ref, source_schema=None,
    predicate=AssertionPredicate.TABLE_EXISTS,
):
    return Assertion(
        id=f"a-{subject_ref}-{predicate.value}",
        subject_ref=subject_ref,
        predicate=predicate,
        payload={},
        source="test",
        confidence=0.9,
        status=AssertionStatus.AUTO,
        run_id="r1",
        observed_at=datetime.now(timezone.utc),
        source_schema=source_schema,
    )


@pytest.fixture
def loader_with_session():
    driver = MagicMock()
    session = MagicMock()
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=session)
    ctx.__exit__ = MagicMock(return_value=False)
    driver.session.return_value = ctx
    return GraphLoader(driver), session


# --- F1: embedding match allowlist + composite Term key --------------------

class TestEmbeddingMatchAllowlist:
    def test_term_key_is_composite(self):
        assert EMBEDDING_MATCH_KEYS["Term"] == ("vocabulary_name", "code")

    def test_validate_label_rejects_unknown(self):
        with pytest.raises(ValueError):
            validate_embedding_label("Bogus")

    def test_validate_props_rejects_unknown_prop(self):
        with pytest.raises(ValueError):
            validate_match_props("Term", ["vocabulary_name", "evil"])

    def test_validate_props_accepts_term_composite(self):
        validate_match_props("Term", ["vocabulary_name", "code"])

    def test_set_node_embedding_requires_match(self, loader_with_session):
        loader, _ = loader_with_session
        with pytest.raises(ValueError):
            loader.set_node_embedding("Term", {}, [0.1])

    def test_set_node_embedding_builds_composite_match(
        self, loader_with_session,
    ):
        loader, session = loader_with_session
        loader.set_node_embedding(
            "Term", {"vocabulary_name": "Gender", "code": "M"}, [0.1],
        )
        cypher = session.run.call_args[0][0]
        params = session.run.call_args[1]
        assert "vocabulary_name: $m_vocabulary_name" in cypher
        assert "code: $m_code" in cypher
        assert params["m_vocabulary_name"] == "Gender"
        assert params["m_code"] == "M"

    def test_set_node_embedding_rejects_bad_prop(self, loader_with_session):
        loader, _ = loader_with_session
        with pytest.raises(ValueError):
            loader.set_node_embedding("Term", {"evil": "x"}, [0.1])

    def test_set_node_embedding_rejects_partial_composite_key(
        self, loader_with_session,
    ):
        loader, _ = loader_with_session
        with pytest.raises(ValueError):
            loader.set_node_embedding("Term", {"code": "M"}, [0.1])

    def test_set_embedding_term_code_only_raises(self, loader_with_session):
        # The back-compat single-prop wrapper must not allow a code-only
        # Term match to slip through and clobber same-code terms.
        loader, _ = loader_with_session
        with pytest.raises(ValueError):
            loader.set_embedding("Term", "code", "M", [0.1])


# --- F3: idempotent assertion writes ---------------------------------------

class TestIdempotentAssertionWrites:
    def test_shared_upsert_is_merge_not_create(self):
        assert "MERGE (n:Assertion {id: a.id})" in _ASSERTION_UPSERT_CYPHER
        assert "CREATE (n:Assertion" not in _ASSERTION_UPSERT_CYPHER
        assert "CREATE (a:Assertion" not in _ASSERTION_UPSERT_CYPHER

    def test_store_assertion_uses_merge(self, loader_with_session):
        loader, session = loader_with_session
        loader.store_assertion(
            _assertion("unity://cat.sch.tbl"), source_schema="sch",
        )
        cypher = session.run.call_args[0][0]
        assert "MERGE (n:Assertion {id: a.id})" in cypher

    def test_store_assertion_idempotent_on_replay(self, loader_with_session):
        loader, session = loader_with_session
        a = _assertion("unity://cat.sch.tbl")
        loader.store_assertion(a, source_schema="sch")
        loader.store_assertion(a, source_schema="sch")
        # Same MERGE Cypher both times — replay is a graph no-op.
        cyphers = {c.args[0] for c in session.run.call_args_list}
        assert all("MERGE (n:Assertion {id: a.id})" in c for c in cyphers)

    def test_ensure_core_constraints_dedupes_then_constrains(
        self, loader_with_session,
    ):
        loader, session = loader_with_session
        loader.ensure_core_constraints()
        cyphers = [c.args[0] for c in session.run.call_args_list]
        assert any("DETACH DELETE dup" in c for c in cyphers)
        assert any(
            "CREATE CONSTRAINT assertion_id_unique" in c
            and "REQUIRE a.id IS UNIQUE" in c
            for c in cyphers
        )


# --- F2: source_schema resolution ------------------------------------------

class TestResolveSourceSchema:
    def test_explicit_arg_wins(self):
        a = _assertion("unity://cat.sch.tbl", source_schema="other")
        assert _resolve_source_schema([a], "explicit") == "explicit"

    def test_derives_single_assertion_schema(self):
        # Stamp and ref agree on the same schema.
        a = _assertion("unity://cat.study_a.tbl", source_schema="study_a")
        assert _resolve_source_schema([a], None) == "study_a"

    def test_raises_on_mixed_assertion_schemas(self):
        a = _assertion("unity://cat.study_a.tbl", source_schema="study_a")
        b = _assertion("unity://cat.study_b.tbl", source_schema="study_b")
        with pytest.raises(ValueError):
            _resolve_source_schema([a, b], None)

    def test_raises_when_stamp_and_ref_disagree(self):
        # One assertion stamped study_a, an unstamped ref under study_b:
        # must raise, not silently materialize everything as study_a.
        a = _assertion("unity://cat.study_a.tbl", source_schema="study_a")
        b = _assertion("unity://cat.study_b.tbl")
        with pytest.raises(ValueError):
            _resolve_source_schema([a, b], None)

    def test_derives_from_refs_when_unset(self):
        # Legacy delegate path: no source_schema on assertions, parse refs.
        a = _assertion("unity://cat.sch.tbl")
        assert _resolve_source_schema([a], None) == "sch"

    def test_raises_on_cross_schema_object_ref(self):
        # A join assertion whose object_ref is in another schema must raise,
        # not resolve to the subject_ref's schema alone.
        a = Assertion(
            id="join-1",
            subject_ref="unity://cat.study_a.tbl.col",
            predicate=AssertionPredicate.HAS_JOIN_EVIDENCE,
            payload={},
            object_ref="unity://cat.study_b.tbl2.col2",
            source="test",
            confidence=0.9,
            status=AssertionStatus.AUTO,
            run_id="r1",
            observed_at=datetime.now(timezone.utc),
        )
        with pytest.raises(ValueError):
            _resolve_source_schema([a], None)

    def test_raises_when_unresolvable(self):
        a = _assertion("plain-unparseable-ref")
        with pytest.raises(ValueError):
            _resolve_source_schema([a], None)


class TestMaterializeUnifiedScoping:
    def test_empty_assertions_is_noop(self):
        loader = MagicMock()
        materialize_unified(loader, [], source_schema=None)
        loader.materialize_provenance_edges.assert_not_called()

    def test_unresolvable_scope_raises_before_writing(self):
        loader = MagicMock()
        with pytest.raises(ValueError):
            materialize_unified(
                loader, [_assertion("plain-unparseable")], source_schema=None,
            )
        # Fail-fast: nothing materialized.
        loader.materialize_provenance_edges.assert_not_called()


# --- #2: MEMBER_OF keys on column_ref --------------------------------------

class TestMemberOfValueSetIdentity:
    def test_links_by_column_ref_when_given(self, loader_with_session):
        loader, session = loader_with_session
        loader.add_term_to_value_set(
            "M", "tbl_col_values", source_schema="sch",
            vocabulary_name="Gender", value_set_ref="cat.sch.tbl.col",
        )
        cypher = session.run.call_args[0][0]
        assert "MERGE (vs:ValueSet {column_ref: $value_set_ref})" in cypher
        assert session.run.call_args[1]["value_set_ref"] == "cat.sch.tbl.col"

    def test_links_by_name_in_legacy_fallback(self, loader_with_session):
        loader, session = loader_with_session
        loader.add_term_to_value_set("M", "dx_types")
        cypher = session.run.call_args[0][0]
        assert "MERGE (vs:ValueSet {name: $value_set_name})" in cypher


# --- #3: retrieval is schema-scoped and namespace-aware --------------------

class TestRetrievalNamespacing:
    def test_value_set_members_query_is_schema_scoped(self):
        q = CypherQueries.find_value_set_members_by_column()
        assert "$schema_name" in q
        assert "vocabulary_name" in q

    def test_expand_value_set_returns_vocabulary(self):
        assert "vocabulary_name" in CypherQueries.expand_value_set()


class TestStudyScopingValidator:
    def test_clean_graph_summarizes_empty(self):
        from scripts.validate_study_scoping import summarize_residual
        assert summarize_residual({}) == ""

    def test_residual_member_of_is_reported_with_details(self):
        from scripts.validate_study_scoping import summarize_residual
        report = summarize_residual({
            "MEMBER_OF": [
                {"value_set_ref": "cat.sch.tbl.col",
                 "value_set_name": "tbl_col_values", "n": 3},
            ],
            "JoinPath": [{"n": 2}],
        })
        assert "MEMBER_OF: 3" in report
        assert "cat.sch.tbl.col" in report
        assert "JoinPath: 2" in report

    def test_has_property_is_gated(self):
        from scripts.validate_study_scoping import _GATED_QUERIES
        assert "HAS_PROPERTY" in _GATED_QUERIES

    def test_study_anchored_vocab_and_alias_edges_gated(self):
        from scripts.validate_study_scoping import _GATED_QUERIES
        assert "CLASSIFIED_AS" in _GATED_QUERIES
        assert "REFERS_TO" in _GATED_QUERIES
        # REFERS_TO gate is scoped to study targets, not Alias->Term.
        assert "target:Entity OR target:Property" in (
            _GATED_QUERIES["REFERS_TO"]
        )

    def test_term_only_edges_not_gated(self):
        from scripts.validate_study_scoping import _GATED_QUERIES
        # Preload-overlapping Term/Vocabulary edges stay informational.
        assert "IN_VOCABULARY" not in _GATED_QUERIES
        assert "PARENT_OF" not in _GATED_QUERIES

    def test_join_path_edges_gated(self):
        from scripts.validate_study_scoping import _GATED_QUERIES
        for rel in ("USES", "FROM_ENTITY", "TO_ENTITY"):
            assert rel in _GATED_QUERIES

    def test_residual_classified_as_reported(self):
        from scripts.validate_study_scoping import summarize_residual
        report = summarize_residual({"CLASSIFIED_AS": [{"n": 5}]})
        assert "CLASSIFIED_AS: 5" in report

    def test_residual_has_property_reported_when_property_on_column_clean(
        self,
    ):
        # PROPERTY_ON_COLUMN backfilled/clean, but a null-scoped HAS_PROPERTY
        # survives — the validator must still flag it.
        from scripts.validate_study_scoping import summarize_residual
        report = summarize_residual({"HAS_PROPERTY": [{"n": 4}]})
        assert "HAS_PROPERTY: 4" in report

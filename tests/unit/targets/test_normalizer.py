"""Normalizer behavior: cross-ref resolution, ordering, endpoint synthesis."""

from __future__ import annotations

import pytest

from sema.models.planner._enums import PrimaryKeyStrategy, TargetArtifactKind
from sema.models.planner.target_model import ForeignKeyObligation
from sema.models.target.obligation import TargetObligationDecl
from sema.models.target.properties import PropertyKind, TargetPropertyDecl
from sema.models.target.refs import (
    TargetEntityRef,
    VocabularyRef,
    VocabularySource,
)
from sema.models.target.term import TargetTermDecl
from sema.models.target.vocab_binding import VocabularyBindingDecl
from sema.targets.exceptions import DanglingRefError
from sema.targets.normalizer import TargetModelNormalizer

from tests.unit.targets.conftest import (
    ScriptedAdapter,
    make_descriptor,
    make_graph_edge_entity,
    make_graph_node_entity,
    make_obligation,
    make_table_row_entity,
)

pytestmark = pytest.mark.unit


def test_dangling_property_in_obligation_rejected() -> None:
    entity = make_table_row_entity(qualified_name="fake.person")
    bad_obligation = TargetObligationDecl(
        target_entity="fake.person",
        required_fields=["nonexistent_field"],
        primary_key=PrimaryKeyStrategy.NATURAL_KEY,
    )
    adapter = ScriptedAdapter(make_descriptor(), [entity], [bad_obligation])
    with pytest.raises(DanglingRefError) as excinfo:
        TargetModelNormalizer.normalize(adapter)
    assert "nonexistent_field" in str(excinfo.value)


def test_dangling_vocabulary_binding_rejected() -> None:
    entity = make_table_row_entity()
    obligation = make_obligation()
    binding = VocabularyBindingDecl(
        entity_ref=entity.ref,
        property_name="person_id",
        vocabulary=VocabularyRef(name="UNKNOWN", source=VocabularySource.INLINE),
    )
    adapter = ScriptedAdapter(
        make_descriptor(), [entity], [obligation], bindings=[binding]
    )
    with pytest.raises(DanglingRefError, match="UNKNOWN"):
        TargetModelNormalizer.normalize(adapter)


def test_external_vocabulary_binding_accepted() -> None:
    entity = make_table_row_entity()
    obligation = make_obligation()
    binding = VocabularyBindingDecl(
        entity_ref=entity.ref,
        property_name="person_id",
        vocabulary=VocabularyRef(name="SNOMED", source=VocabularySource.EXTERNAL),
    )
    adapter = ScriptedAdapter(
        make_descriptor(), [entity], [obligation], bindings=[binding]
    )
    normalized = TargetModelNormalizer.normalize(adapter)
    assert any(b.vocabulary.name == "SNOMED" for b in normalized.vocabulary_bindings)


def test_dangling_fk_referenced_entity_rejected() -> None:
    entity = make_table_row_entity()
    obligation = TargetObligationDecl(
        target_entity="fake.person",
        required_fields=["person_id"],
        primary_key=PrimaryKeyStrategy.NATURAL_KEY,
        foreign_keys=[
            ForeignKeyObligation(
                referenced_entity="fake.absent",
                join_keys=[("person_id", "ref_id")],
            )
        ],
    )
    adapter = ScriptedAdapter(make_descriptor(), [entity], [obligation])
    with pytest.raises(DanglingRefError, match="fake.absent"):
        TargetModelNormalizer.normalize(adapter)


def test_endpoint_synthesis_creates_subject_and_object_properties() -> None:
    llc = make_graph_node_entity(qualified_name="acris.LLC")
    prop_node = make_graph_node_entity(qualified_name="acris.Property")
    edge = make_graph_edge_entity(
        target_model_id="fake-target",
        qualified_name="acris.OWNS",
        subject_target=llc.ref,
        object_target=prop_node.ref,
        columnar_properties=[
            TargetPropertyDecl(name="valid_from", type="date", nullable=False)
        ],
    )
    obligation = TargetObligationDecl(
        target_entity="acris.OWNS",
        required_fields=["subject", "object", "valid_from"],
        primary_key=PrimaryKeyStrategy.NATURAL_KEY,
    )
    adapter = ScriptedAdapter(
        make_descriptor(),
        [llc, prop_node, edge],
        [
            make_obligation("acris.LLC", required_fields=["name"]),
            make_obligation("acris.Property", required_fields=["name"]),
            obligation,
        ],
    )
    normalized = TargetModelNormalizer.normalize(adapter)
    edges = [e for e in normalized.entities if e.ref.qualified_name == "acris.OWNS"]
    assert len(edges) == 1
    edge_props = edges[0].properties
    by_name = {p.name: p for p in edge_props}
    assert {"subject", "object", "valid_from"} <= set(by_name)
    assert by_name["subject"].property_kind is PropertyKind.ENDPOINT
    assert by_name["subject"].endpoint_target_entity_qualified_name == "acris.LLC"
    assert by_name["object"].endpoint_target_entity_qualified_name == "acris.Property"
    assert by_name["valid_from"].property_kind is PropertyKind.COLUMN
    sorted_names = [p.name for p in edge_props]
    assert sorted_names == sorted(sorted_names)


def test_obligation_referencing_unsynthesized_subject_on_node_rejected() -> None:
    bad_node = make_graph_node_entity(qualified_name="acris.LLC")
    bad_obligation = TargetObligationDecl(
        target_entity="acris.LLC",
        required_fields=["subject", "name"],
        primary_key=PrimaryKeyStrategy.NATURAL_KEY,
    )
    adapter = ScriptedAdapter(make_descriptor(), [bad_node], [bad_obligation])
    with pytest.raises(DanglingRefError, match="subject"):
        TargetModelNormalizer.normalize(adapter)


def test_endpoint_to_table_row_rejected() -> None:
    table = make_table_row_entity(qualified_name="fake.person")
    other_node = make_graph_node_entity(qualified_name="fake.LLC")
    edge = make_graph_edge_entity(
        target_model_id="fake-target",
        qualified_name="fake.SAME_AS",
        subject_target=table.ref,
        object_target=other_node.ref,
    )
    adapter = ScriptedAdapter(
        make_descriptor(),
        [table, other_node, edge],
        [
            make_obligation("fake.person", required_fields=["person_id"]),
            make_obligation("fake.LLC", required_fields=["name"]),
            make_obligation("fake.SAME_AS", required_fields=["subject", "object"]),
        ],
    )
    with pytest.raises(DanglingRefError, match="TABLE_ROW"):
        TargetModelNormalizer.normalize(adapter)


def test_endpoint_to_missing_entity_rejected() -> None:
    a = make_graph_node_entity(qualified_name="acris.LLC")
    edge = make_graph_edge_entity(
        target_model_id="fake-target",
        qualified_name="acris.OWNS",
        subject_target=a.ref,
        object_target=TargetEntityRef(
            target_model_id="fake-target",
            qualified_name="acris.Missing",
            kind=TargetArtifactKind.GRAPH_NODE,
        ),
    )
    adapter = ScriptedAdapter(
        make_descriptor(),
        [a, edge],
        [
            make_obligation("acris.LLC", required_fields=["name"]),
            make_obligation("acris.OWNS", required_fields=["subject", "object"]),
        ],
    )
    with pytest.raises(DanglingRefError, match="acris.Missing"):
        TargetModelNormalizer.normalize(adapter)


def test_normalizer_ordering_invariance_across_adapter_runs() -> None:
    a = make_table_row_entity(qualified_name="z.last")
    b = make_table_row_entity(qualified_name="a.first")
    obligations = [
        make_obligation("z.last", required_fields=["person_id"]),
        make_obligation("a.first", required_fields=["person_id"]),
    ]
    descriptor = make_descriptor()
    n1 = TargetModelNormalizer.normalize(ScriptedAdapter(descriptor, [a, b], obligations))
    n2 = TargetModelNormalizer.normalize(ScriptedAdapter(descriptor, [b, a], obligations))
    assert n1 == n2
    assert [e.ref.qualified_name for e in n1.entities] == ["a.first", "z.last"]


def test_inline_terms_collected_when_iter_terms_works() -> None:
    entity = make_table_row_entity()
    binding = VocabularyBindingDecl(
        entity_ref=entity.ref,
        property_name="person_id",
        vocabulary=VocabularyRef(name="GENDER_CV", source=VocabularySource.INLINE),
    )
    term = TargetTermDecl(
        vocabulary=VocabularyRef(name="GENDER_CV", source=VocabularySource.INLINE),
        code="M",
        display="Male",
    )
    adapter = ScriptedAdapter(
        make_descriptor(),
        [entity],
        [make_obligation()],
        bindings=[binding],
        terms=[term],
    )
    normalized = TargetModelNormalizer.normalize(adapter)
    assert any(t.code == "M" for t in normalized.terms)


def test_six_method_adapter_without_iter_terms_loads_cleanly() -> None:
    """The protocol declares iter_terms optional. An adapter that does not
    define iter_terms (and does not inherit the mixin) MUST normalize
    cleanly; bound vocabularies are treated as EXTERNAL."""
    entity = make_table_row_entity()
    binding = VocabularyBindingDecl(
        entity_ref=entity.ref,
        property_name="person_id",
        vocabulary=VocabularyRef(name="SNOMED", source=VocabularySource.EXTERNAL),
    )

    class SixMethodAdapter:
        def describe(self):
            return make_descriptor()

        def discover_entities(self):
            return [entity.ref]

        def load_entity(self, ref):
            return entity

        def load_obligation(self, ref):
            return make_obligation(target_entity=entity.ref.qualified_name)

        def load_vocabulary_bindings(self, ref):
            return [binding] if ref.property_name == "person_id" else []

        def load_context_card(self, ref):
            from sema.models.target.context_card import TargetContextCard

            return TargetContextCard(
                entity_ref=ref,
                card_version="1.0.0",
                description=f"card for {ref.qualified_name}",
            )

    adapter = SixMethodAdapter()
    normalized = TargetModelNormalizer.normalize(adapter)
    assert any(b.vocabulary.name == "SNOMED" for b in normalized.vocabulary_bindings)
    assert normalized.terms == []
    from sema.targets.loader import load_target
    from sema.targets.materializer import InMemoryGraphWriter

    loaded = load_target(adapter, writer=InMemoryGraphWriter())
    assert loaded.target_schema_snapshot_hash


def test_lazy_selected_refs_loads_subset() -> None:
    a = make_table_row_entity(qualified_name="fake.alpha")
    b = make_table_row_entity(qualified_name="fake.beta")
    adapter = ScriptedAdapter(
        make_descriptor(),
        [a, b],
        [
            make_obligation("fake.alpha", required_fields=["person_id"]),
            make_obligation("fake.beta", required_fields=["person_id"]),
        ],
    )
    normalized = TargetModelNormalizer.normalize(adapter, selected_refs=[a.ref])
    assert len(normalized.entities) == 1
    assert normalized.entities[0].ref.qualified_name == "fake.alpha"

"""Neo4j round-trip integration tests for `TargetModelMaterializer`.

Covers tasks 5.17–5.20b and 8.1: golden-manifest materialization round
trip; idempotency under repeat loads; coexisting hash-versioned
generations; `is_current` scoped flip; cross-generation relationship
absence; enrichment-status indexed query path; endpoint property
materialization; FieldMap targeting an endpoint property. Skipped
automatically when no Neo4j is reachable.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from sema.graph.target_loader_migrations import cypher_down, cypher_up
from sema.targets.adapters.manifest import ManifestTargetAdapter
from sema.targets.loader import load_target
from sema.targets.neo4j_writer import Neo4jGraphWriter

pytestmark = pytest.mark.integration


_GOLDEN = Path(__file__).resolve().parents[2] / "unit" / "targets" / "fixtures" / "golden_manifest.yaml"
_GOLDEN_HASH = (_GOLDEN.parent / "golden_manifest_hash.txt").read_text().strip()


@pytest.fixture
def migrated_neo4j(clean_neo4j):
    with clean_neo4j.session() as session:
        for stmt in cypher_up():
            session.run(stmt)
    yield clean_neo4j
    with clean_neo4j.session() as session:
        for stmt in cypher_down():
            session.run(stmt)


def _writer(driver) -> Neo4jGraphWriter:
    return Neo4jGraphWriter(driver)


def test_golden_manifest_round_trip_writes_target_role_nodes(migrated_neo4j) -> None:
    adapter = ManifestTargetAdapter(_GOLDEN)
    loaded = load_target(adapter, writer=_writer(migrated_neo4j))
    assert loaded.target_schema_snapshot_hash == _GOLDEN_HASH
    with migrated_neo4j.session() as s:
        rows = list(
            s.run(
                "MATCH (n:Entity {target_schema_snapshot_hash: $h, model_role: 'TARGET'}) "
                "RETURN n.qualified_name AS qname",
                h=_GOLDEN_HASH,
            )
        )
    qnames = {r["qname"] for r in rows}
    assert "omop.person" in qnames
    assert "omop.observation" in qnames


def test_idempotent_repeat_load_does_not_duplicate(migrated_neo4j) -> None:
    adapter = ManifestTargetAdapter(_GOLDEN)
    load_target(adapter, writer=_writer(migrated_neo4j))
    with migrated_neo4j.session() as s:
        first_nodes = s.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        first_rels = s.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
    load_target(adapter, writer=_writer(migrated_neo4j))
    with migrated_neo4j.session() as s:
        second_nodes = s.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        second_rels = s.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
    assert first_nodes == second_nodes
    assert first_rels == second_rels


def test_two_generations_coexist_with_distinct_snapshot_hashes(
    migrated_neo4j, tmp_path
) -> None:
    load_target(ManifestTargetAdapter(_GOLDEN), writer=_writer(migrated_neo4j))
    raw = yaml.safe_load(_GOLDEN.read_text())
    raw["entities"][0]["properties"][0]["type"] = "string"
    drifted = tmp_path / "drifted.yaml"
    drifted.write_text(yaml.safe_dump(raw))
    drifted_loaded = load_target(
        ManifestTargetAdapter(drifted), writer=_writer(migrated_neo4j)
    )
    assert drifted_loaded.target_schema_snapshot_hash != _GOLDEN_HASH
    with migrated_neo4j.session() as s:
        rows = list(
            s.run(
                "MATCH (n:Entity {qualified_name: 'omop.person'}) "
                "RETURN DISTINCT n.target_schema_snapshot_hash AS h"
            )
        )
    hashes = {r["h"] for r in rows}
    assert _GOLDEN_HASH in hashes
    assert drifted_loaded.target_schema_snapshot_hash in hashes


def test_is_current_flip_scoped_to_loaded_subset(
    migrated_neo4j, tmp_path
) -> None:
    eager = ManifestTargetAdapter(_GOLDEN)
    load_target(eager, writer=_writer(migrated_neo4j))
    raw = yaml.safe_load(_GOLDEN.read_text())
    raw["entities"][0]["properties"][0]["type"] = "string"
    drifted = tmp_path / "drifted.yaml"
    drifted.write_text(yaml.safe_dump(raw))
    drifted_adapter = ManifestTargetAdapter(drifted)
    person_ref = next(
        r for r in drifted_adapter.discover_entities() if r.qualified_name == "omop.person"
    )
    drifted_loaded = load_target(
        drifted_adapter,
        writer=_writer(migrated_neo4j),
        selected_refs=[person_ref],
    )
    with migrated_neo4j.session() as s:
        person_currents = list(
            s.run(
                "MATCH (n:Entity {qualified_name: 'omop.person'}) "
                "RETURN n.target_schema_snapshot_hash AS h, n.is_current AS c"
            )
        )
        observation_currents = list(
            s.run(
                "MATCH (n:Entity {qualified_name: 'omop.observation'}) "
                "RETURN n.target_schema_snapshot_hash AS h, n.is_current AS c"
            )
        )
    person_status = {r["h"]: r["c"] for r in person_currents}
    assert person_status[_GOLDEN_HASH] is False
    assert person_status[drifted_loaded.target_schema_snapshot_hash] is True
    for r in observation_currents:
        assert r["c"] is True


def test_no_cross_generation_has_property_relationships(
    migrated_neo4j, tmp_path
) -> None:
    load_target(ManifestTargetAdapter(_GOLDEN), writer=_writer(migrated_neo4j))
    raw = yaml.safe_load(_GOLDEN.read_text())
    raw["entities"][0]["properties"][0]["type"] = "string"
    drifted = tmp_path / "drifted.yaml"
    drifted.write_text(yaml.safe_dump(raw))
    drifted_loaded = load_target(
        ManifestTargetAdapter(drifted), writer=_writer(migrated_neo4j)
    )
    new_hash = drifted_loaded.target_schema_snapshot_hash
    with migrated_neo4j.session() as s:
        rows = list(
            s.run(
                "MATCH (e:Entity)-[r:HAS_PROPERTY]->(p:Property) "
                "WHERE e.target_schema_snapshot_hash <> p.target_schema_snapshot_hash "
                "RETURN count(*) AS c"
            )
        )
        assert rows[0]["c"] == 0
        endpoint_check = list(
            s.run(
                "MATCH (e:Entity {target_schema_snapshot_hash: $a}) "
                "-[r:HAS_PROPERTY]->"
                "(p:Property {target_schema_snapshot_hash: $b}) "
                "RETURN count(*) AS c",
                a=_GOLDEN_HASH,
                b=new_hash,
            )
        )
    assert endpoint_check[0]["c"] == 0


def test_enrichment_status_indexed_query_returns_deferred_entities(
    migrated_neo4j,
) -> None:
    load_target(ManifestTargetAdapter(_GOLDEN), writer=_writer(migrated_neo4j))
    with migrated_neo4j.session() as s:
        rows = list(
            s.run(
                "MATCH (n:Entity {is_current: true}) "
                "WHERE n.enrichment_vocabulary_bindings_status = 'required_deferred' "
                "RETURN n.qualified_name AS qname"
            )
        )
    assert any(r["qname"] for r in rows)


def test_endpoint_property_carries_endpoint_typing_in_neo4j(
    migrated_neo4j,
) -> None:
    load_target(ManifestTargetAdapter(_GOLDEN), writer=_writer(migrated_neo4j))
    with migrated_neo4j.session() as s:
        rows = list(
            s.run(
                "MATCH (p:Property {property_kind: 'ENDPOINT'}) "
                "WHERE p.parent_entity_qualified_name = 'acris.OWNS' "
                "RETURN p.name AS name, p.endpoint_role AS role, "
                "p.endpoint_target_entity_qualified_name AS tgt, "
                "p.materialized_as_edge_property AS mat_edge"
            )
        )
    by_name = {r["name"]: r for r in rows}
    assert "subject" in by_name and "object" in by_name
    for name, expected in (("subject", "subject"), ("object", "object")):
        assert by_name[name]["role"] == expected
        assert by_name[name]["mat_edge"] is False
        assert by_name[name]["tgt"]


def test_endpoint_property_has_no_separate_constraint(migrated_neo4j) -> None:
    load_target(ManifestTargetAdapter(_GOLDEN), writer=_writer(migrated_neo4j))
    with migrated_neo4j.session() as s:
        rows = list(
            s.run(
                "MATCH (c:Constraint) "
                "WHERE c.attached_property_id IN ['acris.OWNS.subject', 'acris.OWNS.object'] "
                "RETURN count(c) AS c"
            )
        )
    assert rows[0]["c"] == 0


def test_vocabulary_bindings_persist_to_neo4j(migrated_neo4j) -> None:
    loaded = load_target(ManifestTargetAdapter(_GOLDEN), writer=_writer(migrated_neo4j))
    with migrated_neo4j.session() as s:
        rows = list(
            s.run(
                "MATCH (b:VocabularyBinding {is_current: true, "
                "target_schema_snapshot_hash: $h}) "
                "RETURN b.parent_entity_qualified_name AS entity, "
                "b.property_name AS prop, b.vocabulary_name AS vocab, "
                "b.domain AS domain, b.require_standard AS req",
                h=loaded.target_schema_snapshot_hash,
            )
        )
    assert any(
        r["entity"] == "omop.person"
        and r["prop"] == "gender_concept_id"
        and r["vocab"] == "GENDER_CV"
        for r in rows
    )


def test_context_cards_persist_with_hash_and_content(migrated_neo4j) -> None:
    loaded = load_target(ManifestTargetAdapter(_GOLDEN), writer=_writer(migrated_neo4j))
    with migrated_neo4j.session() as s:
        rows = list(
            s.run(
                "MATCH (c:ContextCard {is_current: true, "
                "target_schema_snapshot_hash: $h}) "
                "RETURN c.entity_qualified_name AS qname, "
                "c.card_version AS version, c.card_hash AS card_hash, "
                "c.description AS description",
                h=loaded.target_schema_snapshot_hash,
            )
        )
    by_qname = {r["qname"]: r for r in rows}
    assert "omop.person" in by_qname
    assert len(by_qname["omop.person"]["card_hash"]) == 64
    assert by_qname["omop.person"]["version"]
    assert by_qname["omop.person"]["description"]


def test_target_obligation_and_term_carry_is_current(migrated_neo4j) -> None:
    load_target(ManifestTargetAdapter(_GOLDEN), writer=_writer(migrated_neo4j))
    with migrated_neo4j.session() as s:
        oblig = s.run(
            "MATCH (n:TargetObligation) "
            "RETURN n.is_current AS c LIMIT 1"
        ).single()
        term = s.run(
            "MATCH (n:Term) RETURN n.is_current AS c LIMIT 1"
        ).single()
    assert oblig is not None and oblig["c"] is True
    assert term is not None and term["c"] is True


def test_card_only_bump_creates_new_generation_and_flips_prior(
    migrated_neo4j, tmp_path
) -> None:
    """Spec 8.3a: a card_version bump WITHOUT a schema change must
    produce a new ContextCard generation and flip the prior to
    is_current=false. Pins under the prior card_version need to
    re-read the exact prior content for revalidation."""
    base_loaded = load_target(
        ManifestTargetAdapter(_GOLDEN), writer=_writer(migrated_neo4j)
    )
    raw = yaml.safe_load(_GOLDEN.read_text())
    person = next(
        e for e in raw["entities"] if e["qualified_name"] == "omop.person"
    )
    person["context_card"]["card_version"] = "2.0.0"
    person["context_card"]["description"] = "Bumped wording"
    bumped_path = tmp_path / "bumped.yaml"
    bumped_path.write_text(yaml.safe_dump(raw))
    bumped_loaded = load_target(
        ManifestTargetAdapter(bumped_path), writer=_writer(migrated_neo4j)
    )
    assert bumped_loaded.target_schema_snapshot_hash == base_loaded.target_schema_snapshot_hash
    with migrated_neo4j.session() as s:
        rows = list(
            s.run(
                "MATCH (c:ContextCard {entity_qualified_name: 'omop.person'}) "
                "RETURN c.card_version AS version, c.is_current AS is_current, "
                "c.description AS description"
            )
        )
    by_version = {r["version"]: r for r in rows}
    assert "1.0.0" in by_version, by_version
    assert "2.0.0" in by_version
    assert by_version["1.0.0"]["is_current"] is False
    assert by_version["2.0.0"]["is_current"] is True
    assert by_version["1.0.0"]["description"] != by_version["2.0.0"]["description"]


def test_card_only_bump_has_context_card_relationship_resolves_current(
    migrated_neo4j, tmp_path
) -> None:
    load_target(ManifestTargetAdapter(_GOLDEN), writer=_writer(migrated_neo4j))
    raw = yaml.safe_load(_GOLDEN.read_text())
    person = next(
        e for e in raw["entities"] if e["qualified_name"] == "omop.person"
    )
    person["context_card"]["card_version"] = "2.0.0"
    person["context_card"]["description"] = "Bumped wording"
    bumped = tmp_path / "bumped.yaml"
    bumped.write_text(yaml.safe_dump(raw))
    load_target(ManifestTargetAdapter(bumped), writer=_writer(migrated_neo4j))
    with migrated_neo4j.session() as s:
        rows = list(
            s.run(
                "MATCH (e:Entity {qualified_name: 'omop.person'})"
                "-[r:HAS_CONTEXT_CARD]->(c:ContextCard) "
                "RETURN c.card_version AS version"
            )
        )
    versions = {r["version"] for r in rows}
    assert versions == {"1.0.0", "2.0.0"}


def test_loaded_target_exposes_full_context_cards_in_python(migrated_neo4j) -> None:
    loaded = load_target(ManifestTargetAdapter(_GOLDEN), writer=_writer(migrated_neo4j))
    qnames = {c.entity_ref.qualified_name for c in loaded.context_cards}
    assert "omop.person" in qnames
    assert all(c.card_hash and len(c.card_hash) == 64 for c in loaded.context_cards)


def test_field_map_targeting_endpoint_property_is_addressable(
    migrated_neo4j,
) -> None:
    """The synthesized endpoint Property is the target slot for a FieldMap.

    A FieldMap.target_field_ref points at the endpoint property by the
    same hash-versioned identity tuple any other Property uses; the
    integration test asserts the slot is queryable, which is what the
    planner contract's plan-verdict derivation needs.
    """
    loaded = load_target(
        ManifestTargetAdapter(_GOLDEN), writer=_writer(migrated_neo4j)
    )
    with migrated_neo4j.session() as s:
        rows = list(
            s.run(
                "MATCH (p:Property {property_kind: 'ENDPOINT'}) "
                "WHERE p.target_schema_snapshot_hash = $h "
                "AND p.parent_entity_qualified_name = 'acris.OWNS' "
                "AND p.name = 'subject' "
                "RETURN p.id AS pid",
                h=loaded.target_schema_snapshot_hash,
            )
        )
    assert len(rows) == 1
    assert rows[0]["pid"]

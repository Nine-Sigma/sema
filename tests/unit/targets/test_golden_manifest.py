"""Golden compliance fixture: snapshot hashing, materialization, round-trip."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from sema.models.target.enrichment import EnrichmentStatus, Facet
from sema.targets.adapters.manifest import ManifestTargetAdapter
from sema.targets.adapters.manifest_exceptions import ManifestEndpointError
from sema.targets.exceptions import DanglingRefError
from sema.targets.hashing import SnapshotHasher
from sema.targets.loader import load_target
from sema.targets.materializer import InMemoryGraphWriter
from sema.targets.materializer_ops import (
    EntityOp,
    PropertyOp,
    TargetObligationOp,
)
from sema.targets.normalizer import TargetModelNormalizer

pytestmark = pytest.mark.unit

_FIXTURES = Path(__file__).parent / "fixtures"
GOLDEN = _FIXTURES / "golden_manifest.yaml"
PINNED_HASH_FILE = _FIXTURES / "golden_manifest_hash.txt"


def _pinned_hash() -> str:
    return PINNED_HASH_FILE.read_text().strip()


def _normalize(path: Path):
    adapter = ManifestTargetAdapter(path)
    return TargetModelNormalizer.normalize(adapter)


def test_pinned_digest_matches() -> None:
    digest = SnapshotHasher.hash(_normalize(GOLDEN))
    assert digest == _pinned_hash()


def test_two_loads_produce_same_hash() -> None:
    h1 = SnapshotHasher.hash(_normalize(GOLDEN))
    h2 = SnapshotHasher.hash(_normalize(GOLDEN))
    assert h1 == h2


def test_permuted_entity_order_yields_same_hash(tmp_path: Path) -> None:
    raw = yaml.safe_load(GOLDEN.read_text())
    permuted = dict(raw)
    permuted["entities"] = list(reversed(raw["entities"]))
    permuted_path = tmp_path / "permuted_manifest.yaml"
    permuted_path.write_text(yaml.safe_dump(permuted))
    h_original = SnapshotHasher.hash(_normalize(GOLDEN))
    h_permuted = SnapshotHasher.hash(_normalize(permuted_path))
    assert h_original == h_permuted


def test_mutating_property_type_changes_hash(tmp_path: Path) -> None:
    raw = yaml.safe_load(GOLDEN.read_text())
    raw["entities"][0]["properties"][0]["type"] = "string"
    mutated_path = tmp_path / "mutated_manifest.yaml"
    mutated_path.write_text(yaml.safe_dump(raw))
    h_original = SnapshotHasher.hash(_normalize(GOLDEN))
    h_mutated = SnapshotHasher.hash(_normalize(mutated_path))
    assert h_original != h_mutated


def test_model_role_target_on_every_materialized_node() -> None:
    adapter = ManifestTargetAdapter(GOLDEN)
    writer = InMemoryGraphWriter()
    load_target(adapter, writer=writer)
    target_model_id = adapter.parsed_manifest.descriptor.target_model_id
    target_model_version = adapter.parsed_manifest.descriptor.target_model_version
    for op in writer.ops:
        prov = getattr(op, "target_model_id", None)
        if prov is not None:
            assert prov == target_model_id
            assert op.target_model_version == target_model_version  # type: ignore[union-attr]


def test_obligation_round_trip_against_writer() -> None:
    adapter = ManifestTargetAdapter(GOLDEN)
    writer = InMemoryGraphWriter()
    load_target(adapter, writer=writer)
    obligation_ops = {
        op.target_entity: op
        for op in writer.ops
        if isinstance(op, TargetObligationOp)
    }
    assert "omop.person" in obligation_ops
    person = obligation_ops["omop.person"]
    assert "person_id" in person.payload["required_fields"]
    assert "gender_concept_id" in person.payload["required_fields"]
    owns = obligation_ops["acris.OWNS"]
    minimum = owns.payload["minimum_viable_row"]
    fields = sorted(c["field"] for c in minimum["clauses"])
    assert fields == ["object", "subject", "valid_from"]


def test_invalid_property_ref_rejected(tmp_path: Path) -> None:
    raw = yaml.safe_load(GOLDEN.read_text())
    raw["entities"][0]["obligation"]["required_fields"].append("nonexistent_field")
    bad_path = tmp_path / "bad_property_ref.yaml"
    bad_path.write_text(yaml.safe_dump(raw))
    adapter = ManifestTargetAdapter(bad_path)
    with pytest.raises(DanglingRefError, match="nonexistent_field"):
        TargetModelNormalizer.normalize(adapter)


def test_invalid_vocabulary_ref_rejected(tmp_path: Path) -> None:
    raw = yaml.safe_load(GOLDEN.read_text())
    raw["entities"][0]["properties"][1]["vocabulary_binding"] = {
        "vocabulary": "GENDER_CV_TYPO"
    }
    bad_path = tmp_path / "bad_vocab_ref.yaml"
    bad_path.write_text(yaml.safe_dump(raw))
    adapter = ManifestTargetAdapter(bad_path)
    with pytest.raises(DanglingRefError, match="GENDER_CV_TYPO"):
        TargetModelNormalizer.normalize(adapter)


def test_invalid_fk_ref_rejected(tmp_path: Path) -> None:
    raw = yaml.safe_load(GOLDEN.read_text())
    raw["entities"][1]["obligation"]["foreign_keys"] = [
        {
            "referenced_entity": "omop.absent",
            "join_keys": [["person_id", "person_id"]],
        }
    ]
    bad_path = tmp_path / "bad_fk_ref.yaml"
    bad_path.write_text(yaml.safe_dump(raw))
    adapter = ManifestTargetAdapter(bad_path)
    with pytest.raises(DanglingRefError, match="omop.absent"):
        TargetModelNormalizer.normalize(adapter)


def test_endpoint_synthesis_for_temporal_and_non_temporal_edges() -> None:
    adapter = ManifestTargetAdapter(GOLDEN)
    writer = InMemoryGraphWriter()
    load_target(adapter, writer=writer)
    endpoint_ops_owns = [
        op
        for op in writer.ops
        if isinstance(op, PropertyOp)
        and op.parent_entity_qualified_name == "acris.OWNS"
        and op.property_kind == "ENDPOINT"
    ]
    endpoint_ops_same_as = [
        op
        for op in writer.ops
        if isinstance(op, PropertyOp)
        and op.parent_entity_qualified_name == "match.SAME_AS"
        and op.property_kind == "ENDPOINT"
    ]
    assert {p.name for p in endpoint_ops_owns} == {"subject", "object"}
    assert {p.name for p in endpoint_ops_same_as} == {"subject", "object"}
    for op in endpoint_ops_owns + endpoint_ops_same_as:
        assert op.materialized_as_edge_property is False


def test_vocabulary_binding_round_trip() -> None:
    adapter = ManifestTargetAdapter(GOLDEN)
    normalized = TargetModelNormalizer.normalize(adapter)
    bindings_by_prop = {
        (b.entity_ref.qualified_name, b.property_name): b
        for b in normalized.vocabulary_bindings
    }
    binding = bindings_by_prop[("omop.person", "gender_concept_id")]
    assert binding.vocabulary.name == "GENDER_CV"
    assert binding.domain == "Gender"
    assert binding.require_standard is True


def test_one_decision_record_per_entity_in_loaded_target() -> None:
    adapter = ManifestTargetAdapter(GOLDEN)
    loaded = load_target(adapter, writer=InMemoryGraphWriter())
    qnames = {r.qualified_name for r in loaded.entity_refs}
    decision_qnames = {d.entity_ref.qualified_name for d in loaded.enrichment_decisions}
    assert decision_qnames == qnames


def test_decisions_cover_five_facets_and_compact_status_agrees() -> None:
    adapter = ManifestTargetAdapter(GOLDEN)
    writer = InMemoryGraphWriter()
    loaded = load_target(adapter, writer=writer)
    decision_by_qname = {d.entity_ref.qualified_name: d for d in loaded.enrichment_decisions}
    entity_ops = [op for op in writer.ops if isinstance(op, EntityOp)]
    for op in entity_ops:
        record = decision_by_qname[op.qualified_name]
        for facet in Facet:
            assert op.enrichment_status[facet.value] == record.decisions[facet].status.value


def test_inline_synonyms_yield_supplied_for_semantic_aliases() -> None:
    adapter = ManifestTargetAdapter(GOLDEN)
    loaded = load_target(adapter, writer=InMemoryGraphWriter())
    person = next(
        d for d in loaded.enrichment_decisions if d.entity_ref.qualified_name == "omop.person"
    )
    assert (
        person.decisions[Facet.semantic_aliases].status
        is EnrichmentStatus.supplied_by_adapter
    )


def test_card_version_aggregate_changes_when_one_card_bumps(tmp_path: Path) -> None:
    raw = yaml.safe_load(GOLDEN.read_text())
    base_path = tmp_path / "base.yaml"
    base_path.write_text(yaml.safe_dump(raw))
    base = load_target(
        ManifestTargetAdapter(base_path), writer=InMemoryGraphWriter()
    )
    bumped: dict[str, Any] = yaml.safe_load(GOLDEN.read_text())
    bumped["entities"][0]["context_card"]["card_version"] = "1.1.0"
    bumped["entities"][0]["context_card"]["description"] = "Bumped"
    bumped_path = tmp_path / "bumped.yaml"
    bumped_path.write_text(yaml.safe_dump(bumped))
    after = load_target(
        ManifestTargetAdapter(bumped_path), writer=InMemoryGraphWriter()
    )
    assert base.target_schema_snapshot_hash == after.target_schema_snapshot_hash
    assert base.aggregate_context_card_version != after.aggregate_context_card_version


def test_endpoints_block_on_table_row_rejected(tmp_path: Path) -> None:
    raw = yaml.safe_load(GOLDEN.read_text())
    raw["entities"][0]["endpoints"] = {
        "subject": {"target_entity": "omop.person"},
        "object": {"target_entity": "omop.person"},
    }
    bad_path = tmp_path / "bad_endpoints.yaml"
    bad_path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ManifestEndpointError):
        ManifestTargetAdapter(bad_path)

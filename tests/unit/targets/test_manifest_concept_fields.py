"""Manifest parse of the concept-binding fields (design 5.2) and the 5.9 golden guard."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from sema.targets.adapters.manifest import ManifestTargetAdapter
from sema.targets.adapters.manifest_parser import parse_manifest_file
from sema.targets.hashing import SnapshotHasher
from sema.targets.normalizer import TargetModelNormalizer

pytestmark = pytest.mark.unit

FIXTURES = Path(__file__).parent / "fixtures"
GOLDEN_MANIFEST = FIXTURES / "golden_manifest.yaml"
CONCEPT_MANIFEST = FIXTURES / "concept_manifest.yaml"
# Pinned on the pre-slice tree (35f2ac8); design 5.9 requires it never moves.
GOLDEN_SNAPSHOT_HASH = "e8c9bfdc1026f1593686d197047c0d49acbd97db343badb73d0bba847241644b"

UNLOADED_FAR_END_ROW = {
    "from": {"system": "partner", "id": "PRT4013886"},
    "to": {"system": "omop", "id": "4013886"},
    "type": "Maps to",
    "hierarchical": False,
    "far_end_loaded": False,
}


def test_golden_manifest_snapshot_hash_unchanged_regression_guard_5_9() -> None:
    normalized = TargetModelNormalizer.normalize(ManifestTargetAdapter(GOLDEN_MANIFEST))
    assert SnapshotHasher.hash(normalized) == GOLDEN_SNAPSHOT_HASH


def test_concept_manifest_parses_every_new_field() -> None:
    parsed = parse_manifest_file(CONCEPT_MANIFEST)
    scheme = parsed.entities[0]
    assert scheme.kind.value == "CONCEPT_SCHEME"
    race = scheme.properties[0]
    assert race.concept_ref is not None
    assert race.concept_ref.identity_system == "partner"
    assert race.concept_ref.id == "PRT4013886"
    assert race.concept_ref.version == "2026-09"
    assert race.definition == "A self-identified social construct of ancestral grouping."
    assert [(r.system, r.id) for r in race.external_refs] == [
        ("omop", "4013886"),
        ("SNOMED", "103579009"),
    ]
    white = next(t for t in parsed.terms if t.code == "PRT1000001")
    assert white.definition is not None
    assert white.synonyms == ["Caucasian"]
    assert [(r.system, r.id) for r in white.external_refs] == [("omop", "8527")]


def test_relationship_row_with_unloaded_far_end_is_parsed() -> None:
    parsed = parse_manifest_file(CONCEPT_MANIFEST)
    rows = [r.model_dump(by_alias=True) for r in parsed.relationships]
    assert rows == [UNLOADED_FAR_END_ROW]


def test_manifest_without_new_fields_yields_empty_defaults() -> None:
    parsed = parse_manifest_file(GOLDEN_MANIFEST)
    assert parsed.relationships == []
    for entity in parsed.entities:
        for prop in entity.properties:
            assert prop.concept_ref is None
            assert prop.definition is None
            assert prop.external_refs == []
    for term in parsed.terms:
        assert term.definition is None
        assert term.synonyms == []
        assert term.external_refs == []


def test_relationships_are_parse_only_and_leave_normalized_model_unchanged(
    tmp_path: Path,
) -> None:
    raw = yaml.safe_load(GOLDEN_MANIFEST.read_text())
    raw["relationships"] = [UNLOADED_FAR_END_ROW]
    with_relationships = tmp_path / "golden_with_relationships.yaml"
    with_relationships.write_text(yaml.safe_dump(raw))

    adapter = ManifestTargetAdapter(with_relationships)
    assert len(adapter.parsed_manifest.relationships) == 1
    normalized = TargetModelNormalizer.normalize(adapter)
    baseline = TargetModelNormalizer.normalize(ManifestTargetAdapter(GOLDEN_MANIFEST))

    assert "relationships" not in type(normalized).model_fields
    assert normalized == baseline
    # SnapshotHasher projects the normalized model, not the manifest, so the hash holds.
    assert SnapshotHasher.hash(normalized) == GOLDEN_SNAPSHOT_HASH

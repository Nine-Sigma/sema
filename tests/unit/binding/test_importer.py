"""Concept source importer: manifest document and authored-evidence sidecar."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from sema.targets.adapters.manifest_models import ParsedManifest
from sema.targets.adapters.manifest_parser import parse_manifest_raw
from tests.unit.binding.fixtures.concept_library import (
    CONCEPT_IDS,
    ETHNICITY_REF,
    RACE_ID,
    RACE_REF,
    RACE_VALUE_CODES,
    RESOLVED_PHYSICAL,
    SPEC_PATH,
    build_library,
    load_dataset_map,
    provider_for,
)

pytestmark = pytest.mark.unit

RACE_TERM_OMOP_IDS = {
    "SYN_RACE_WHITE": "900101",
    "SYN_RACE_BLACK": "900102",
    "SYN_RACE_ASIAN": "900103",
}


def _import() -> Any:
    from sema.binding.importer import import_concept_source
    from sema.binding.spec import load_concept_source_spec

    spec = load_concept_source_spec(SPEC_PATH)
    con = build_library()
    try:
        return import_concept_source(
            spec, provider_for(con), dataset_map=load_dataset_map()
        )
    finally:
        con.close()


def _parsed(result: Any) -> ParsedManifest:
    # Round-trip through YAML: the manifest must be plain, dumpable data.
    return parse_manifest_raw(yaml.safe_load(yaml.safe_dump(result.manifest)))


def _concept_properties(parsed: ParsedManifest) -> list[Any]:
    return [p for e in parsed.entities for p in e.properties if p.concept_ref]


def _race_property(parsed: ParsedManifest) -> Any:
    return next(p for p in _concept_properties(parsed) if p.concept_ref.id == RACE_ID)


def test_four_concept_properties_grouped_from_value_rows() -> None:
    props = _concept_properties(_parsed(_import()))

    assert sorted(p.concept_ref.id for p in props) == sorted(CONCEPT_IDS)
    assert {p.concept_ref.identity_system for p in props} == {"syn"}
    assert {p.concept_ref.version for p in props} == {"2026-09"}
    assert all(p.definition for p in props)
    race = [p for p in props if p.concept_ref.id == RACE_ID]
    assert len(race) == 1
    assert race[0].definition == "Self-identified ancestral grouping."


def test_race_value_set_has_three_terms_with_omop_refs() -> None:
    parsed = _parsed(_import())
    race = _race_property(parsed)

    assert race.vocabulary_binding is not None
    vocab_name = race.vocabulary_binding.vocabulary
    assert vocab_name == f"VS_{RACE_ID}"
    vocab = next(v for v in parsed.vocabularies if v.name == vocab_name)
    assert vocab.source.value == "INLINE"
    terms = {t.code: t for t in parsed.terms if t.vocabulary == vocab_name}
    assert set(terms) == set(RACE_VALUE_CODES)
    for code, omop_id in RACE_TERM_OMOP_IDS.items():
        refs = {(r.system, r.id) for r in terms[code].external_refs}
        assert ("omop", omop_id) in refs
    assert terms["SYN_RACE_WHITE"].definition == "Origins in Europe."


def test_concept_level_external_refs_are_on_race_property() -> None:
    race = _race_property(_parsed(_import()))

    refs = {(r.system, r.id) for r in race.external_refs}
    assert ("omop", "900001") in refs
    assert ("SYNVOCAB", "C0001") in refs


def test_relationship_to_external_only_far_end_is_not_loaded() -> None:
    rels = _parsed(_import()).relationships

    unloaded = [r for r in rels if not r.far_end_loaded]
    assert len(unloaded) == 1
    row = unloaded[0]
    assert (row.from_.system, row.from_.id) == ("syn", RACE_ID)
    assert (row.to.system, row.to.id) == ("omop", "900999")
    assert row.type == "Maps to"
    loaded = [r for r in rels if r.far_end_loaded]
    assert [(r.from_.id, r.to.id) for r in loaded] == [("SYN_ETHNICITY", RACE_ID)]
    # Neither endpoint of the SYN_OTHER row is loaded, so it is dropped.
    assert len(rels) == 2


def _snapshot(result: Any, path: Path) -> tuple[Any, str]:
    from sema.targets.adapters.manifest import ManifestTargetAdapter
    from sema.targets.hashing import SnapshotHasher
    from sema.targets.normalizer import TargetModelNormalizer

    path.write_text(yaml.safe_dump(result.manifest, sort_keys=False))
    normalized = TargetModelNormalizer.normalize(ManifestTargetAdapter(path))
    return normalized, SnapshotHasher.hash(normalized)


def test_manifest_loads_through_adapter_with_stable_snapshot_hash(
    tmp_path: Path,
) -> None:
    normalized, first_hash = _snapshot(_import(), tmp_path / "first.yaml")
    _, second_hash = _snapshot(_import(), tmp_path / "second.yaml")

    assert normalized.descriptor.target_model_version == "2026-09"
    assert len(normalized.entities) == 1
    scheme = normalized.entities[0]
    assert scheme.ref.kind.value == "CONCEPT_SCHEME"
    assert scheme.ref.qualified_name == "syn_library.demographics"
    concept_props = [p for p in scheme.properties if p.property_kind.value == "CONCEPT"]
    assert len(concept_props) == 4
    assert first_hash == second_hash


def test_sidecar_column_bindings_resolve_through_dataset_map() -> None:
    from sema.binding.models import AuthoredEvidence

    authored = _import().authored
    assert isinstance(authored, AuthoredEvidence)

    bindings = {b.concept_ref: b for b in authored.column_bindings}
    assert set(bindings) == {RACE_REF, ETHNICITY_REF}
    resolved, unresolved = bindings[RACE_REF], bindings[ETHNICITY_REF]
    assert resolved.resolved is True
    assert resolved.physical == RESOLVED_PHYSICAL
    assert resolved.columns == ["race_label", "race_cd"]
    assert resolved.dataset_label == "Synthetic Patients"
    assert resolved.dataset_id == "ds-0001"
    assert resolved.project == "Synthetic Project"
    assert unresolved.resolved is False
    assert unresolved.physical is None
    assert unresolved.columns == ["ethnicity"]
    assert unresolved.project is None


def test_sidecar_value_mappings_approved_only_for_reviewed() -> None:
    mappings = _import().authored.value_mappings

    assert len(mappings) == 3
    assert {m.concept_ref for m in mappings} == {RACE_REF}
    approved = {(m.source_value, m.status, m.approved) for m in mappings}
    assert approved == {
        ("white", "Reviewed", True),
        ("caucasian", "Awaiting Review - LLM Prediction", False),
        ("asian american", "Pending Ontologist Review", False),
    }
    white = next(m for m in mappings if m.source_value == "white")
    assert white.codes == ["SYN_RACE_WHITE"]

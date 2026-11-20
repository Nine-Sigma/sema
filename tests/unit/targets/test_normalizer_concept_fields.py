"""Normalizer carries the concept-binding fields from the manifest (design 5.2)."""

from __future__ import annotations

from pathlib import Path

import pytest

from sema.models.target.normalized import NormalizedTargetModel
from sema.targets.adapters.manifest import ManifestTargetAdapter
from sema.targets.normalizer import TargetModelNormalizer

pytestmark = pytest.mark.unit

CONCEPT_MANIFEST = Path(__file__).parent / "fixtures" / "concept_manifest.yaml"


def _normalized_concept_manifest() -> NormalizedTargetModel:
    return TargetModelNormalizer.normalize(ManifestTargetAdapter(CONCEPT_MANIFEST))


def test_scheme_entity_kind_is_concept_scheme() -> None:
    entity = _normalized_concept_manifest().entities[0]
    assert entity.ref.kind.value == "CONCEPT_SCHEME"


def test_property_with_concept_ref_is_concept_kind_and_carries_fields() -> None:
    entity = _normalized_concept_manifest().entities[0]
    props = {p.name: p for p in entity.properties}
    race = props["race"]
    assert race.property_kind.value == "CONCEPT"
    assert race.concept_ref is not None
    assert race.concept_ref.identity_system == "partner"
    assert race.concept_ref.id == "PRT4013886"
    assert race.concept_ref.version == "2026-09"
    assert race.definition == "A self-identified social construct of ancestral grouping."
    assert [(r.system, r.id) for r in race.external_refs] == [
        ("omop", "4013886"),
        ("SNOMED", "103579009"),
    ]


def test_property_without_concept_ref_stays_column_kind() -> None:
    entity = _normalized_concept_manifest().entities[0]
    note = next(p for p in entity.properties if p.name == "record_note")
    assert note.property_kind.value == "COLUMN"
    assert note.concept_ref is None
    assert note.definition is None
    assert note.external_refs == []


def test_terms_carry_definition_synonyms_and_external_refs() -> None:
    terms = {t.code: t for t in _normalized_concept_manifest().terms}
    white, unknown = terms["PRT1000001"], terms["PRT1000002"]
    assert white.definition is not None
    assert white.synonyms == ["Caucasian"]
    assert [(r.system, r.id) for r in white.external_refs] == [("omop", "8527")]
    assert unknown.definition is None
    assert unknown.synonyms == []
    assert unknown.external_refs == []

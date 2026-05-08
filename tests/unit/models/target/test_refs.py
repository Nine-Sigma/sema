"""Tests for target-side refs (TargetEntityRef, TargetPropertyRef, VocabularyRef)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from sema.models.planner._enums import TargetArtifactKind
from sema.models.target.refs import (
    TargetEntityRef,
    TargetPropertyRef,
    VocabularyRef,
    VocabularySource,
)


pytestmark = pytest.mark.unit


def _entity_ref() -> TargetEntityRef:
    return TargetEntityRef(
        target_model_id="omop-cdm",
        qualified_name="omop.person",
        kind=TargetArtifactKind.TABLE_ROW,
    )


def test_entity_ref_round_trip() -> None:
    ref = _entity_ref()
    blob = ref.model_dump_json()
    assert TargetEntityRef.model_validate_json(blob) == ref


def test_entity_ref_qualified_name_must_be_dotted() -> None:
    with pytest.raises(ValidationError):
        TargetEntityRef(
            target_model_id="omop-cdm",
            qualified_name="person",
            kind=TargetArtifactKind.TABLE_ROW,
        )


def test_property_ref_extra_field_rejected() -> None:
    with pytest.raises(ValidationError):
        TargetPropertyRef(  # type: ignore[call-arg]
            entity_ref=_entity_ref(),
            property_name="person_id",
            extra=1,
        )


def test_property_ref_frozen() -> None:
    pr = TargetPropertyRef(entity_ref=_entity_ref(), property_name="person_id")
    with pytest.raises(ValidationError):
        pr.property_name = "x"  # type: ignore[misc]


def test_vocabulary_ref_source_enum() -> None:
    ext = VocabularyRef(name="SNOMED", source=VocabularySource.EXTERNAL)
    inline = VocabularyRef(name="local-list", source=VocabularySource.INLINE)
    assert ext.source.value == "EXTERNAL"
    assert inline.source.value == "INLINE"

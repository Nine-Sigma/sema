"""Tests for VocabularyBindingDecl."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from sema.models.planner._enums import TargetArtifactKind
from sema.models.target.refs import TargetEntityRef, VocabularyRef, VocabularySource
from sema.models.target.vocab_binding import VocabularyBindingDecl


pytestmark = pytest.mark.unit


def _entity_ref() -> TargetEntityRef:
    return TargetEntityRef(
        target_model_id="omop-cdm",
        qualified_name="omop.person",
        kind=TargetArtifactKind.TABLE_ROW,
    )


def test_minimal_binding() -> None:
    vb = VocabularyBindingDecl(
        entity_ref=_entity_ref(),
        property_name="gender_concept_id",
        vocabulary=VocabularyRef(name="SNOMED", source=VocabularySource.EXTERNAL),
    )
    assert vb.require_standard is False
    assert vb.allow_zero_default is False
    assert vb.domain is None


def test_binding_round_trip_with_all_hooks() -> None:
    vb = VocabularyBindingDecl(
        entity_ref=_entity_ref(),
        property_name="gender_concept_id",
        vocabulary=VocabularyRef(name="SNOMED", source=VocabularySource.EXTERNAL),
        domain="Gender",
        require_standard=True,
        allow_zero_default=True,
        effective_date_ref="omop.person.start_date",
        resolver_policy_ref="omop.policy.standard_only",
    )
    assert VocabularyBindingDecl.model_validate_json(vb.model_dump_json()) == vb


def test_binding_extra_field_rejected() -> None:
    with pytest.raises(ValidationError):
        VocabularyBindingDecl(  # type: ignore[call-arg]
            entity_ref=_entity_ref(),
            property_name="x",
            vocabulary=VocabularyRef(name="SNOMED", source=VocabularySource.EXTERNAL),
            mystery=1,
        )

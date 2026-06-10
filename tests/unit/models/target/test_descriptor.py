"""Tests for TargetModelDescriptor."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from sema.models.target.completeness import (
    SemanticCompleteness,
    SemanticCompletenessAnnotations,
)
from sema.models.target.descriptor import TargetModelDescriptor


pytestmark = pytest.mark.unit


def _annotations() -> SemanticCompletenessAnnotations:
    return SemanticCompletenessAnnotations(
        structure=SemanticCompleteness.COMPLETE,
        obligations=SemanticCompleteness.COMPLETE,
        vocabulary_bindings=SemanticCompleteness.COMPLETE,
        semantic_aliases=SemanticCompleteness.PARTIAL,
        terms=SemanticCompleteness.EXTERNAL,
    )


def test_descriptor_round_trip() -> None:
    desc = TargetModelDescriptor(
        target_model_id="omop-cdm",
        target_model_version="5.4.0",
        display_name="OMOP CDM",
        owner="ohdsi",
        vocabulary_release="2025-01",
        completeness=_annotations(),
    )
    assert TargetModelDescriptor.model_validate_json(desc.model_dump_json()) == desc


@pytest.mark.parametrize(
    "bad_id",
    [
        "OMOP-CDM",
        "omop_cdm",
        "1omop",
        "-omop",
        "omop cdm",
        "omop.cdm",
    ],
)
def test_descriptor_rejects_non_kebab_target_model_id(bad_id: str) -> None:
    with pytest.raises(ValidationError):
        TargetModelDescriptor(
            target_model_id=bad_id,
            target_model_version="1.0.0",
            display_name="x",
            completeness=_annotations(),
        )


def test_descriptor_extra_field_rejected() -> None:
    with pytest.raises(ValidationError):
        TargetModelDescriptor(  # type: ignore[call-arg]
            target_model_id="omop-cdm",
            target_model_version="5.4.0",
            display_name="OMOP CDM",
            completeness=_annotations(),
            extra="oops",
        )

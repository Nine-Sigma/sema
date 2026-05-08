"""Tests for SemanticCompleteness enum and SemanticCompletenessAnnotations."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from sema.models.target.completeness import (
    SemanticCompleteness,
    SemanticCompletenessAnnotations,
)


pytestmark = pytest.mark.unit


def test_enum_values_exact_set() -> None:
    assert {m.value for m in SemanticCompleteness} == {
        "COMPLETE",
        "PARTIAL",
        "NONE",
        "EXTERNAL",
    }


def test_annotations_requires_all_five_facets() -> None:
    with pytest.raises(ValidationError) as exc:
        SemanticCompletenessAnnotations(  # type: ignore[call-arg]
            structure=SemanticCompleteness.COMPLETE,
        )
    msg = str(exc.value)
    for facet in ("obligations", "vocabulary_bindings", "semantic_aliases", "terms"):
        assert facet in msg


def test_annotations_extra_field_rejected() -> None:
    with pytest.raises(ValidationError):
        SemanticCompletenessAnnotations(  # type: ignore[call-arg]
            structure=SemanticCompleteness.COMPLETE,
            obligations=SemanticCompleteness.COMPLETE,
            vocabulary_bindings=SemanticCompleteness.PARTIAL,
            semantic_aliases=SemanticCompleteness.PARTIAL,
            terms=SemanticCompleteness.EXTERNAL,
            extra="boom",
        )


def test_annotations_frozen() -> None:
    ann = SemanticCompletenessAnnotations(
        structure=SemanticCompleteness.COMPLETE,
        obligations=SemanticCompleteness.COMPLETE,
        vocabulary_bindings=SemanticCompleteness.PARTIAL,
        semantic_aliases=SemanticCompleteness.PARTIAL,
        terms=SemanticCompleteness.EXTERNAL,
    )
    with pytest.raises(ValidationError):
        ann.structure = SemanticCompleteness.NONE  # type: ignore[misc]

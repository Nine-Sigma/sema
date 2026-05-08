"""Tests for TargetTermDecl."""

from __future__ import annotations

import pytest

from sema.models.target.refs import VocabularyRef, VocabularySource
from sema.models.target.term import TargetTermDecl


pytestmark = pytest.mark.unit


def test_term_round_trip() -> None:
    t = TargetTermDecl(
        vocabulary=VocabularyRef(name="local-list", source=VocabularySource.INLINE),
        code="MALE",
        display="Male",
        domain="Gender",
    )
    assert TargetTermDecl.model_validate_json(t.model_dump_json()) == t


def test_term_minimum_fields() -> None:
    t = TargetTermDecl(
        vocabulary=VocabularyRef(name="local-list", source=VocabularySource.INLINE),
        code="MALE",
        display="Male",
    )
    assert t.domain is None

"""Tests for TargetContextCard."""

from __future__ import annotations

import pytest
from packaging.version import Version
from pydantic import ValidationError

from sema.models.planner._enums import TargetArtifactKind
from sema.models.target.context_card import TargetContextCard
from sema.models.target.refs import TargetEntityRef


pytestmark = pytest.mark.unit


def _ref() -> TargetEntityRef:
    return TargetEntityRef(
        target_model_id="omop-cdm",
        qualified_name="omop.person",
        kind=TargetArtifactKind.TABLE_ROW,
    )


def _card(**kwargs: object) -> TargetContextCard:
    base: dict[str, object] = {
        "entity_ref": _ref(),
        "card_version": "1.0.0",
        "description": "OMOP person table",
        "examples": ["A patient record with demographic columns."],
    }
    base.update(kwargs)
    return TargetContextCard(**base)  # type: ignore[arg-type]


def test_card_round_trip() -> None:
    card = _card(curated_synonyms=["patient", "subject"])
    assert TargetContextCard.model_validate_json(card.model_dump_json()) == card


def test_card_default_card_hash_is_none() -> None:
    card = _card()
    assert card.card_hash is None


def test_card_rejects_non_none_card_hash_at_construction() -> None:
    with pytest.raises(ValidationError):
        _card(card_hash="deadbeef")


def test_card_rejects_64_char_hex_card_hash_at_construction() -> None:
    """Adapter MUST NOT supply a card_hash, even if shaped as a 64-char
    hex SHA-256 digest. Hash computation is Sema-owned via the loader."""
    with pytest.raises(ValidationError):
        _card(card_hash="0" * 64)
    with pytest.raises(ValidationError):
        _card(card_hash="abcdef0123456789" * 4)


def test_card_description_must_be_non_empty() -> None:
    with pytest.raises(ValidationError):
        _card(description="")


def test_card_description_max_4000_chars() -> None:
    with pytest.raises(ValidationError):
        _card(description="x" * 4001)


@pytest.mark.parametrize(
    "version",
    ["1.0.0", "2.1.3b1", "0.0.0+synthesized", "1.0.0.post1"],
)
def test_card_version_accepts_pep440(version: str) -> None:
    card = _card(card_version=version)
    assert Version(card.card_version) == Version(version)


@pytest.mark.parametrize("bad_version", ["not-a-version", "0.0.0-synthesized", ""])
def test_card_version_rejects_non_pep440(bad_version: str) -> None:
    with pytest.raises(ValidationError):
        _card(card_version=bad_version)


def test_card_extra_field_rejected() -> None:
    with pytest.raises(ValidationError):
        _card(extra=1)


def test_card_frozen() -> None:
    card = _card()
    with pytest.raises(ValidationError):
        card.description = "x"  # type: ignore[misc]

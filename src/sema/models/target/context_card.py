"""TargetContextCard — stable target-Entity prefix material for prompts."""

from __future__ import annotations

from typing import Self

from packaging.version import InvalidVersion, Version
from pydantic import BaseModel, ConfigDict, Field, model_validator

from sema.models.target.refs import TargetEntityRef


class TargetContextCard(BaseModel):
    """Adapter-constructed context card. Adapters MUST set
    `card_hash=None`; the loader populates the hash via
    `LoadedContextCard.from_target_card`.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    entity_ref: TargetEntityRef
    card_version: str = Field(min_length=1)
    description: str = Field(min_length=1, max_length=4000)
    examples: list[str] = Field(default_factory=list)
    obligation_summary: str | None = None
    curated_synonyms: list[str] = Field(default_factory=list)
    card_hash: None = None

    @model_validator(mode="after")
    def _validate_construction_invariants(self) -> Self:
        if self.card_hash is not None:
            raise ValueError(
                "card_hash is owned by Sema; adapters MUST construct cards with "
                "card_hash=None and let the loader populate it"
            )
        try:
            Version(self.card_version)
        except InvalidVersion as exc:
            raise ValueError(
                f"card_version {self.card_version!r} is not a PEP 440-parseable version"
            ) from exc
        return self


class LoadedContextCard(BaseModel):
    """Loader-populated context card carrying the Sema-computed
    `card_hash`. Constructed only by the loader from a validated
    `TargetContextCard`; adapters never instantiate this class.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    entity_ref: TargetEntityRef
    card_version: str = Field(min_length=1)
    description: str = Field(min_length=1, max_length=4000)
    examples: list[str] = Field(default_factory=list)
    obligation_summary: str | None = None
    curated_synonyms: list[str] = Field(default_factory=list)
    card_hash: str = Field(min_length=64, max_length=64)

    @classmethod
    def from_target_card(
        cls, source: TargetContextCard, card_hash: str
    ) -> "LoadedContextCard":
        return cls(
            entity_ref=source.entity_ref,
            card_version=source.card_version,
            description=source.description,
            examples=list(source.examples),
            obligation_summary=source.obligation_summary,
            curated_synonyms=list(source.curated_synonyms),
            card_hash=card_hash,
        )

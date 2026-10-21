"""The shape-template contract: relation profiles, slot bindings, shared floors (Decision 1).

Profiles are computed once per landed relation and passed to every template, so a template
reads aggregates and issues statements only through the profiling tools.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from sema.binding.spec_sections import Provider, RelationRef
from sema.intake.landing import LOCATOR_COLUMN, SHEET_SEPARATOR, LandedRelation, LandResult
from sema.intake.profile import ProfileTools
from sema.intake.profile_utils import ColumnProfile
from sema.intake.verify.base import Verdict

__all__ = [
    "ID_DISTINCT_RATIO_FLOOR",
    "JOIN_RESOLVE_FLOOR",
    "LABEL_MEAN_LENGTH_BOUNDS",
    "ColumnProfile",
    "RelationProfile",
    "RelationSignature",
    "SlotBinding",
    "Template",
    "id_columns",
    "is_text",
    "label_columns",
    "profile_relations",
]

# Measured: the syn-library concept relation's key 1.0 and A2 Concepts.concept_code 1.0,
# while the repeating key A2 Values.concept_code is 0.333.
ID_DISTINCT_RATIO_FLOOR = 0.98
# Measured: A2 labels mean 5.4 (Values.label) to 6.4 (Concepts.concept_name), A3 semsql_terms
# labels 7.7, ACRIS code-table descriptions 9.2 to 24.1.
LABEL_MEAN_LENGTH_BOUNDS = (3.0, 120.0)
# Measured: A2 Values.concept_code into Concepts.concept_code 1.0, syn-library
# data_element.data_element_id into the concept relation's key 1.0.
JOIN_RESOLVE_FLOOR = 0.95

TEXT_TYPES = frozenset({"VARCHAR", "STRING", "TEXT", "CHAR"})
SIZED_TEXT_PREFIXES = ("VARCHAR(", "CHAR(", "STRING(")

RelationSignature = tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class RelationProfile:
    ref: RelationRef
    landed: LandedRelation
    row_count: int
    columns: tuple[ColumnProfile, ...]
    headerless: bool

    @property
    def signature(self) -> RelationSignature:
        return tuple(sorted((column.name, column.type) for column in self.columns))

    @property
    def stem(self) -> str:
        stem, separator, _ = self.landed.name.rpartition(SHEET_SEPARATOR)
        return stem if separator else self.landed.name

    def column(self, name: str) -> ColumnProfile | None:
        return next((column for column in self.columns if column.name == name), None)


class SlotBinding(BaseModel):
    """One way a template's slots bind; ``relations`` lists every relation it spans."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    template: str
    section: str | None = None
    slots: dict[str, str]
    evidence: dict[str, float | str] = {}
    relations: tuple[str, ...]


class Template(Protocol):
    ID: str

    def candidates(
        self, group: Sequence[RelationProfile], tools: ProfileTools
    ) -> list[SlotBinding]: ...

    def verify(
        self, binding: SlotBinding, group: Sequence[RelationProfile], tools: ProfileTools
    ) -> Verdict: ...


def profile_relations(
    land: LandResult, tools: ProfileTools, provider: Provider = "staging"
) -> list[RelationProfile]:
    headerless = set(land.headerless)
    profiles = []
    for landed in land.relations:
        ref = RelationRef(provider=provider, name=f"{landed.schema}.{landed.name}")
        described = tools.describe_relation(ref)
        columns = tuple(c for c in described.columns if c.name != LOCATOR_COLUMN)
        profiles.append(
            RelationProfile(ref, landed, described.row_count, columns, landed.name in headerless)
        )
    return profiles


def is_text(column: ColumnProfile) -> bool:
    kind = column.type.upper()
    return kind in TEXT_TYPES or kind.startswith(SIZED_TEXT_PREFIXES)


def is_id(profile: RelationProfile, column: ColumnProfile) -> bool:
    return (
        profile.row_count > 0
        and column.null_rate == 0.0
        and not column.is_constant
        and column.distinct_ratio >= ID_DISTINCT_RATIO_FLOOR
    )


def is_label(column: ColumnProfile) -> bool:
    low, high = LABEL_MEAN_LENGTH_BOUNDS
    mean = column.mean_length
    return is_text(column) and not column.is_constant and mean is not None and low <= mean <= high


def id_columns(profile: RelationProfile) -> tuple[str, ...]:
    return tuple(column.name for column in profile.columns if is_id(profile, column))


def label_columns(profile: RelationProfile) -> tuple[str, ...]:
    return tuple(column.name for column in profile.columns if is_label(column))

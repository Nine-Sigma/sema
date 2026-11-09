"""EntityIdentity: how stored rows resolve to one row per entity.

An entity whose grain differs from its stored rows declares its key columns, what a
null key means, and one fixed aggregate rule per resolved attribute. This is not a
rules engine: the rules are the four forms of :class:`ResolutionRule`, each with an
optional preferred value.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ResolutionRule(StrEnum):
    MIN = "min"
    MAX = "max"
    MIN_BY = "min_by"
    MAX_BY = "max_by"


ORDERED_BY_ANOTHER_COLUMN = frozenset({ResolutionRule.MIN_BY, ResolutionRule.MAX_BY})


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PreferredValue(_Frozen):
    """Rows where ``column`` equals ``value`` win before the rule orders the rest."""

    column: str
    value: str | int | float | bool


class ResolvedAttribute(_Frozen):
    column: str
    rule: ResolutionRule
    by: str | None = None
    prefer: PreferredValue | None = None
    name: str | None = None

    @property
    def output_name(self) -> str:
        return self.name or self.column

    @property
    def columns(self) -> tuple[str, ...]:
        prefer = (self.prefer.column,) if self.prefer else ()
        by = (self.by,) if self.by else ()
        return (self.column, *by, *prefer)

    @model_validator(mode="after")
    def _by_matches_rule(self) -> Self:
        ordered = self.rule in ORDERED_BY_ANOTHER_COLUMN
        if ordered and self.by is None:
            raise ValueError(f"rule {self.rule.value} requires by <column>")
        if not ordered and self.by is not None:
            raise ValueError(f"rule {self.rule.value} takes no by column")
        return self


class EntityIdentity(_Frozen):
    key_columns: tuple[str, ...] = Field(min_length=1)
    exclude_null_keys: bool = False
    null_keys_group: bool = True
    attributes: tuple[ResolvedAttribute, ...] = ()

    @property
    def columns(self) -> tuple[str, ...]:
        """Every physical column the resolution reads, in first-use order."""
        used = [*self.key_columns, *(c for a in self.attributes for c in a.columns)]
        return tuple(dict.fromkeys(used))

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if len(set(self.key_columns)) < len(self.key_columns):
            raise ValueError(f"key columns repeat: {list(self.key_columns)}")
        outputs = [a.output_name for a in self.attributes]
        if len(set(outputs)) < len(outputs):
            raise ValueError(f"attribute names repeat: {outputs}")
        if clash := set(outputs) & set(self.key_columns):
            raise ValueError(f"attribute names a key column: {sorted(clash)}")
        if self.exclude_null_keys and "null_keys_group" in self.model_fields_set:
            raise ValueError("null_keys_group applies only when null keys are kept")
        return self

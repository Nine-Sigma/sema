"""EntityIdentity: how stored rows resolve to one row per entity.

An entity whose grain differs from its stored rows declares its key columns, what a
null key means (for every key column or only the named ones), and one fixed aggregate
rule per resolved attribute. This is not a
rules engine: the rules are the four forms of :class:`ResolutionRule`, each with an
optional preferred value or ranked list of preferred values.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, StrictBool, model_validator


class ResolutionRule(StrEnum):
    MIN = "min"
    MAX = "max"
    MIN_BY = "min_by"
    MAX_BY = "max_by"


ORDERED_BY_ANOTHER_COLUMN = frozenset({ResolutionRule.MIN_BY, ResolutionRule.MAX_BY})

Scalar = str | int | float | bool


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _distinct(values: tuple[object, ...], what: str) -> None:
    if len(set(values)) < len(values):
        raise ValueError(f"{what} repeat: {list(values)}")


class PreferredValue(_Frozen):
    """Rows where ``column`` equals ``value`` win before the rule orders the rest.

    A list of values ranks them: rows matching the first win, then the second, and so on.
    """

    column: str
    value: Scalar | tuple[Scalar, ...]

    @property
    def ranked(self) -> tuple[Scalar, ...]:
        return self.value if isinstance(self.value, tuple) else (self.value,)

    @model_validator(mode="after")
    def _ranked_values(self) -> Self:
        if not self.ranked:
            raise ValueError("a preference ranks at least one value")
        _distinct(self.ranked, "preferred values")
        return self


class ResolvedAttribute(_Frozen):
    column: str
    rule: ResolutionRule
    by: str | None = None
    prefer: PreferredValue | None = None
    name: str | None = None
    label_of: str | None = None

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
    exclude_null_keys: StrictBool | tuple[str, ...] = False
    null_keys_group: bool = True
    attributes: tuple[ResolvedAttribute, ...] = ()

    @property
    def non_null_keys(self) -> tuple[str, ...]:
        """Key columns a row must have non-null to be kept; ``True`` names every key."""
        if isinstance(self.exclude_null_keys, tuple):
            return self.exclude_null_keys
        return self.key_columns if self.exclude_null_keys else ()

    @property
    def nullable_keys(self) -> tuple[str, ...]:
        return tuple(k for k in self.key_columns if k not in self.non_null_keys)

    @property
    def columns(self) -> tuple[str, ...]:
        """Every physical column the resolution reads, in first-use order."""
        used = [*self.key_columns, *(c for a in self.attributes for c in a.columns)]
        return tuple(dict.fromkeys(used))

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        _distinct(self.key_columns, "key columns")
        self._check_non_null_keys()
        outputs = [a.output_name for a in self.attributes]
        if len(set(outputs)) < len(outputs):
            raise ValueError(f"attribute names repeat: {outputs}")
        if clash := set(outputs) & set(self.key_columns):
            raise ValueError(f"attribute names a key column: {sorted(clash)}")
        self._check_labels(outputs)
        if not self.nullable_keys and "null_keys_group" in self.model_fields_set:
            raise ValueError("null_keys_group applies only when null keys are kept")
        return self

    def _check_labels(self, outputs: list[str]) -> None:
        resolved = {*self.key_columns, *outputs}
        for attribute in self.attributes:
            code = attribute.label_of
            if code is not None and (code == attribute.output_name or code not in resolved):
                raise ValueError(
                    f"{attribute.output_name} is the label of {code!r}, which is not "
                    "another resolved column of the entity"
                )

    def _check_non_null_keys(self) -> None:
        if not isinstance(self.exclude_null_keys, tuple):
            return
        if not self.exclude_null_keys:
            raise ValueError("exclude_null_keys names at least one key column")
        _distinct(self.exclude_null_keys, "exclude_null_keys columns")
        if strays := set(self.exclude_null_keys) - set(self.key_columns):
            raise ValueError(f"exclude_null_keys names {sorted(strays)}, not a key column")

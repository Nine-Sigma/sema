"""Generic non-vocabulary projection compile (US-014).

R29-scanned generic spine: this module names no showcase literal. A curated
analytics target with no ``vocabulary_binding`` flows source -> target through
the SAME compiler as the vocab path, with zero concept resolution. Each
:class:`~sema.models.planner.field_map.FieldMap` is projected per its
:class:`~sema.models.planner.patterns.MappingPattern` — DIRECT_COPY (copy or
rename), DERIVED (a cast), CONSTANT (a literal) — into one plain staging SELECT,
built once as a SQLGlot AST and rendered per dialect.

There is deliberately no vocabulary structure here (no StagingColumns, no
StagingDecision, no domain gate): proving the spine handles a non-OMOP target
without an OMOP-shaped branch is the whole point of US-014.
"""

from __future__ import annotations

from typing import Sequence

import sqlglot.expressions as exp

from sema.models.planner.field_map import FieldMap
from sema.models.planner.patterns import (
    ConstantValue,
    DerivedExpression,
    DirectCopyPayload,
    MappingPattern,
)

__all__ = [
    "UnsupportedProjectionError",
    "build_projection_select",
    "target_column_name",
    "target_columns",
]

_SOURCE_ALIAS = "src"


class UnsupportedProjectionError(ValueError):
    """Raised when a FieldMap pattern has no plain-projection builder."""


def target_column_name(target_field_ref: str) -> str:
    """Physical staging column name = the last dotted segment of the ref."""
    return target_field_ref.rsplit(".", 1)[-1]


def target_columns(field_maps: Sequence[FieldMap]) -> tuple[str, ...]:
    """Ordered physical staging column names for the projection."""
    return tuple(target_column_name(fm.target_field_ref) for fm in field_maps)


def _source_column(source_field_ref: str) -> exp.Column:
    return exp.column(source_field_ref.rsplit(".", 1)[-1], _SOURCE_ALIAS)


def _direct_copy(payload: DirectCopyPayload) -> exp.Expr:
    return _source_column(payload.source_field_ref)


def _constant(payload: ConstantValue) -> exp.Expr:
    if payload.literal_value is None:
        return exp.null()
    return exp.convert(payload.literal_value)


def _derived(payload: DerivedExpression) -> exp.Expr:
    cast = payload.expression_ast.get("cast")
    if not isinstance(cast, dict) or "to_type" not in cast:
        raise UnsupportedProjectionError(
            "DERIVED projection supports only a {'cast': {'to_type': ...}} ast"
        )
    source_ref = cast.get("source_field_ref") or payload.source_field_refs[0]
    return exp.cast(_source_column(source_ref), to=str(cast["to_type"]))


def _value_for(field_map: FieldMap) -> exp.Expr:
    payload = field_map.payload
    if field_map.pattern is MappingPattern.DIRECT_COPY and isinstance(
        payload, DirectCopyPayload
    ):
        return _direct_copy(payload)
    if field_map.pattern is MappingPattern.CONSTANT and isinstance(
        payload, ConstantValue
    ):
        return _constant(payload)
    if field_map.pattern is MappingPattern.DERIVED and isinstance(
        payload, DerivedExpression
    ):
        return _derived(payload)
    raise UnsupportedProjectionError(
        f"no plain projection for pattern {field_map.pattern.value}"
    )


def _project(field_map: FieldMap) -> exp.Expr:
    return exp.alias_(
        _value_for(field_map), target_column_name(field_map.target_field_ref)
    )


def build_projection_select(
    source_schema: str, source_table: str, field_maps: Sequence[FieldMap]
) -> exp.Select:
    """Build the plain staging SELECT projecting each field map from source."""
    if not field_maps:
        raise UnsupportedProjectionError("no field maps to project")
    source = exp.to_table(f"{source_schema}.{source_table}").as_(_SOURCE_ALIAS)
    projections: list[exp.Expr] = [_project(fm) for fm in field_maps]
    return exp.select(*projections).from_(source)

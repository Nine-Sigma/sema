"""§1.5(b) staging projection — SQLGlot AST builders and column config (US-010).

This module is R29-scanned (``src/sema/compile`` is a guard core path): it names
no showcase literal. The physical staging column names arrive as a
:class:`StagingColumns` instance whose showcase values (the source value column,
the target concept column) live behind the policy boundary
(:mod:`sema.resolve.policies.omop`). Resolved decisions arrive as generic
:class:`StagingDecision` rows so this module never touches the store's
concept-id field.

The AST is built once (:func:`build_staging_select`) and rendered per dialect by
the caller, so DuckDB (dev) and Databricks (prod) share one definition.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Sequence

import sqlglot.expressions as exp

_VALUES_ALIAS = "m"
_VALUES_COLUMNS = ("code", "target_value", "resolution_status", "no_map_reason", "status")
_SOURCE_ALIAS = "src"


@dataclass(frozen=True)
class StagingColumns:
    """Physical staging column names (§1.5(b)).

    ``source_value_column`` and ``target_concept_column`` carry the showcase
    names and therefore have no defaults — the policy supplies them. The
    remaining names are generic spine columns with neutral defaults.
    """

    source_value_column: str
    target_concept_column: str
    source_schema: str = "source_schema"
    source_table: str = "source_table"
    source_row_ref: str = "source_row_ref"
    source_patient_key: str = "source_patient_key"
    resolver_policy_ref: str = "resolver_policy_ref"
    vocab_release: str = "vocab_release"
    resolution_status: str = "resolution_status"
    no_map_reason: str = "no_map_reason"
    status_column: str = "status"
    run_id: str = "run_id"


@dataclass(frozen=True)
class StagingDecision:
    """One inlined resolved decision (§1.5(b) projection input).

    ``target_value`` is the resolved target id, NULL for a NO_MAP code.
    """

    normalized_source_value: str
    target_value: int | None
    resolution_status: str
    no_map_reason: str | None
    status: str


@dataclass(frozen=True)
class SourceTableSpec:
    """The source table a study is compiled from."""

    schema: str
    table: str
    value_column: str
    row_ref_column: str | None = None
    patient_key_column: str | None = None


@dataclass(frozen=True)
class CompileContext:
    """Run-scoped provenance literals stamped onto every staging row."""

    resolver_policy_ref: str
    vocab_release: str
    run_id: str


# §1.5(b) column order. ``StagingColumns`` field order minus the two required
# showcase columns interleaved at their projection positions.
_COLUMN_FIELD_ORDER: tuple[str, ...] = (
    "source_schema",
    "source_table",
    "source_row_ref",
    "source_patient_key",
    "source_value_column",
    "target_concept_column",
    "resolver_policy_ref",
    "vocab_release",
    "resolution_status",
    "no_map_reason",
    "status_column",
    "run_id",
)

_BIGINT_FIELD = "target_concept_column"


def staging_column_order(columns: StagingColumns) -> tuple[str, ...]:
    """Physical column names in §1.5(b) order."""
    return tuple(getattr(columns, field) for field in _COLUMN_FIELD_ORDER)


def _column_types(columns: StagingColumns) -> dict[str, str]:
    types: dict[str, str] = {}
    for field in _COLUMN_FIELD_ORDER:
        name = getattr(columns, field)
        types[name] = "BIGINT" if field is _BIGINT_FIELD else "VARCHAR"
    return types


def _qualified(schema: str, table: str) -> str:
    return f'"{schema}"."{table}"'


def create_staging_table_sql(
    columns: StagingColumns, schema: str, table: str
) -> str:
    """``CREATE TABLE IF NOT EXISTS`` for the §1.5(b) staging columns."""
    types = _column_types(columns)
    cols = ",\n  ".join(
        f'"{name}" {types[name]}' for name in staging_column_order(columns)
    )
    return f"CREATE TABLE IF NOT EXISTS {_qualified(schema, table)} (\n  {cols}\n)"


def _decision_row(decision: StagingDecision) -> exp.Tuple:
    target = (
        exp.null()
        if decision.target_value is None
        else exp.Literal.number(decision.target_value)
    )
    reason = (
        exp.null()
        if decision.no_map_reason is None
        else exp.Literal.string(decision.no_map_reason)
    )
    return exp.tuple_(
        exp.Literal.string(decision.normalized_source_value),
        target,
        exp.Literal.string(decision.resolution_status),
        reason,
        exp.Literal.string(decision.status),
    )


def _values_join(decisions: Sequence[StagingDecision]) -> exp.Expression:
    rows = [_decision_row(d) for d in decisions]
    return exp.values(rows, alias=_VALUES_ALIAS, columns=list(_VALUES_COLUMNS))


def _src(column: str) -> exp.Column:
    return exp.column(column, _SOURCE_ALIAS)


def _m(column: str) -> exp.Column:
    return exp.column(column, _VALUES_ALIAS)


def _optional_source(column: str | None) -> exp.Expression:
    return _src(column) if column else exp.null()


def _projections(
    columns: StagingColumns,
    source: SourceTableSpec,
    context: CompileContext,
) -> list[exp.Expr]:
    return [
        exp.alias_(exp.Literal.string(source.schema), columns.source_schema),
        exp.alias_(exp.Literal.string(source.table), columns.source_table),
        exp.alias_(_optional_source(source.row_ref_column), columns.source_row_ref),
        exp.alias_(
            _optional_source(source.patient_key_column), columns.source_patient_key
        ),
        exp.alias_(_src(source.value_column), columns.source_value_column),
        exp.alias_(_m("target_value"), columns.target_concept_column),
        exp.alias_(
            exp.Literal.string(context.resolver_policy_ref), columns.resolver_policy_ref
        ),
        exp.alias_(exp.Literal.string(context.vocab_release), columns.vocab_release),
        exp.alias_(_m("resolution_status"), columns.resolution_status),
        exp.alias_(_m("no_map_reason"), columns.no_map_reason),
        exp.alias_(_m("status"), columns.status_column),
        exp.alias_(exp.Literal.string(context.run_id), columns.run_id),
    ]


def build_staging_select(
    columns: StagingColumns,
    source: SourceTableSpec,
    context: CompileContext,
    decisions: Sequence[StagingDecision],
) -> exp.Select:
    """Build the §1.5(b) staging SELECT: source LEFT JOIN inlined decisions."""
    source_table = exp.to_table(f"{source.schema}.{source.table}").as_(_SOURCE_ALIAS)
    join_on = _src(source.value_column).eq(_m("code"))
    return (
        exp.select(*_projections(columns, source, context))
        .from_(source_table)
        .join(_values_join(decisions), on=join_on, join_type="left")
    )


def insert_from_temp_sql(
    columns: StagingColumns, schema: str, table: str, temp: str
) -> str:
    """Insert every staging column from the temp build table (explicit order)."""
    col_list = ", ".join(f'"{c}"' for c in staging_column_order(columns))
    return (
        f"INSERT INTO {_qualified(schema, table)} ({col_list}) "
        f"SELECT {col_list} FROM {temp}"
    )


def delete_scope_sql(columns: StagingColumns, schema: str, table: str) -> str:
    """Delete the (source_schema, source_table) partition before re-insert."""
    return (
        f"DELETE FROM {_qualified(schema, table)} "
        f'WHERE "{columns.source_schema}" = ? AND "{columns.source_table}" = ?'
    )


def count_scope_sql(columns: StagingColumns, schema: str, table: str) -> str:
    return (
        f"SELECT COUNT(*) FROM {_qualified(schema, table)} "
        f'WHERE "{columns.source_schema}" = ? AND "{columns.source_table}" = ?'
    )


def _assert_full_column_coverage() -> None:
    """Guard: the §1.5(b) order must enumerate every StagingColumns field."""
    declared = {f.name for f in fields(StagingColumns)}
    assert declared == set(_COLUMN_FIELD_ORDER)


_assert_full_column_coverage()

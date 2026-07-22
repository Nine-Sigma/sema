"""S1-06 — FK-closed multi-table compile: specs + SQLGlot builders (generic).

R29-scanned generic spine: every physical column name arrives via a spec from
the policy boundary; nothing here names ``person``/``condition-occurrence``/OMOP.

The child table (e.g. condition-occurrence) is a projection of the source with:
  * a source-derived surrogate PK (S1-05), stable across a dedup rebuild;
  * a resolved-identity FK (e.g. person_id) supplied by an INNER JOIN to the
    identity registry — a source row whose key is absent from the registry
    (blank/missing key, routed to review in S1-02) is excluded, never given a
    synthetic FK;
  * a resolved value column with a NO_MAP sentinel default (D8);
  * NULL nullable columns (D4 — no fabricated date);
  * the source-scope columns so the write can swap one study at a time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import sqlglot.expressions as exp

from sema.compile.compiler_utils import StagingDecision
from sema.compile.row_surrogate import surrogate_row_id_expr

_SRC = "src"
_MAP = "m"
_REG = "reg"
_VALUES_COLUMNS = ("code", "target_value")


@dataclass(frozen=True)
class RegistryJoinSpec:
    """Where the resolved-identity FK is read from (the identity registry)."""

    schema: str
    table: str
    namespace_column: str
    key_column: str
    id_column: str


@dataclass(frozen=True)
class ParentTableSpec:
    """The FK-target table (e.g. omop.person): distinct canonical ids."""

    schema: str
    table: str
    id_column: str


@dataclass(frozen=True)
class ChildTableSpec:
    """The FK-closed child table (e.g. omop.condition-occurrence)."""

    schema: str
    table: str
    pk_column: str
    fk_column: str
    value_column: str
    null_columns: tuple[str, ...]
    row_ref_column: str
    patient_key_column: str
    scope_schema_column: str
    scope_table_column: str

    def column_order(self) -> tuple[str, ...]:
        return (
            self.pk_column,
            self.fk_column,
            self.value_column,
            *self.null_columns,
            self.row_ref_column,
            self.patient_key_column,
            self.scope_schema_column,
            self.scope_table_column,
        )

    def column_types(self) -> dict[str, str]:
        types = {
            self.pk_column: "BIGINT",
            self.fk_column: "BIGINT",
            self.value_column: "BIGINT",
            self.row_ref_column: "VARCHAR",
            self.patient_key_column: "VARCHAR",
            self.scope_schema_column: "VARCHAR",
            self.scope_table_column: "VARCHAR",
        }
        for col in self.null_columns:
            types[col] = "DATE"
        return types


@dataclass(frozen=True)
class ChildSourceSpec:
    """The source table a study's child rows are compiled from."""

    schema: str
    table: str
    value_column: str
    row_ref_column: str
    patient_key_column: str


def _src(column: str) -> exp.Column:
    return exp.column(column, _SRC)


def _values_join(decisions: Sequence[StagingDecision]) -> exp.Expression:
    rows = [
        exp.tuple_(
            exp.Literal.string(d.normalized_source_value),
            exp.null()
            if d.target_value is None
            else exp.Literal.number(d.target_value),
        )
        for d in decisions
    ]
    return exp.values(rows, alias=_MAP, columns=list(_VALUES_COLUMNS))


def build_child_select(
    child: ChildTableSpec,
    source: ChildSourceSpec,
    registry: RegistryJoinSpec,
    decisions: Sequence[StagingDecision],
    *,
    no_map_default: int,
    dialect: str,
) -> exp.Select:
    """Build the FK-closed child SELECT (surrogate PK + registry FK join)."""
    surrogate = surrogate_row_id_expr(
        source_row_column=source.row_ref_column,
        source_schema=source.schema,
        source_table=source.table,
        source_alias=_SRC,
        dialect=dialect,
    )
    concept = exp.func(
        "COALESCE", exp.column("target_value", _MAP), exp.Literal.number(no_map_default)
    )
    projections = [
        exp.alias_(surrogate, child.pk_column),
        exp.alias_(exp.column(registry.id_column, _REG), child.fk_column),
        exp.alias_(concept, child.value_column),
        *[exp.alias_(exp.null(), col) for col in child.null_columns],
        exp.alias_(_src(source.row_ref_column), child.row_ref_column),
        exp.alias_(_src(source.patient_key_column), child.patient_key_column),
        exp.alias_(exp.Literal.string(source.schema), child.scope_schema_column),
        exp.alias_(exp.Literal.string(source.table), child.scope_table_column),
    ]
    src_table = exp.to_table(f"{source.schema}.{source.table}").as_(_SRC)
    reg_table = exp.to_table(f"{registry.schema}.{registry.table}").as_(_REG)
    value_join = _src(source.value_column).eq(exp.column("code", _MAP))
    reg_join = exp.and_(
        exp.column(registry.namespace_column, _REG).eq(
            exp.Literal.string(source.schema)
        ),
        exp.column(registry.key_column, _REG).eq(
            exp.func("TRIM", _src(source.patient_key_column))
        ),
    )
    return (
        exp.select(*projections)
        .from_(src_table)
        .join(_values_join(decisions), on=value_join, join_type="left")
        .join(reg_table, on=reg_join, join_type="inner")
    )


def _qualified(schema: str, table: str) -> str:
    return f'"{schema}"."{table}"'


def create_child_table_sql(child: ChildTableSpec) -> str:
    types = child.column_types()
    cols = ",\n  ".join(f'"{c}" {types[c]}' for c in child.column_order())
    return (
        f"CREATE TABLE IF NOT EXISTS {_qualified(child.schema, child.table)} "
        f"(\n  {cols}\n)"
    )


def insert_child_from_temp_sql(child: ChildTableSpec, temp: str) -> str:
    cols = ", ".join(f'"{c}"' for c in child.column_order())
    return (
        f"INSERT INTO {_qualified(child.schema, child.table)} ({cols}) "
        f"SELECT {cols} FROM {temp}"
    )


def delete_child_scope_sql(child: ChildTableSpec) -> str:
    return (
        f"DELETE FROM {_qualified(child.schema, child.table)} "
        f'WHERE "{child.scope_schema_column}" = ?'
    )


def count_child_scope_sql(child: ChildTableSpec) -> str:
    return (
        f"SELECT COUNT(*) FROM {_qualified(child.schema, child.table)} "
        f'WHERE "{child.scope_schema_column}" = ?'
    )


def replace_parent_sql(parent: ParentTableSpec, registry: RegistryJoinSpec) -> str:
    """Rebuild the FK-target from the WHOLE registry (all studies), distinct ids."""
    return (
        f"CREATE OR REPLACE TABLE {_qualified(parent.schema, parent.table)} AS "
        f'SELECT DISTINCT "{registry.id_column}" AS "{parent.id_column}" '
        f"FROM {_qualified(registry.schema, registry.table)}"
    )


def orphan_fk_count_sql(parent: ParentTableSpec, child: ChildTableSpec) -> str:
    """Count child rows whose FK is absent from the parent (must be 0 at rest)."""
    return (
        f"SELECT COUNT(*) FROM {_qualified(child.schema, child.table)} c "
        f"LEFT JOIN {_qualified(parent.schema, parent.table)} p "
        f'ON c."{child.fk_column}" = p."{parent.id_column}" '
        f'WHERE p."{parent.id_column}" IS NULL'
    )


def missing_key_count_sql(source: ChildSourceSpec) -> str:
    """Count source rows with a blank patient key (the review disposition, D5)."""
    return (
        f"SELECT COUNT(*) FROM {_qualified(source.schema, source.table)} "
        f"WHERE TRIM(COALESCE(\"{source.patient_key_column}\", '')) = ''"
    )

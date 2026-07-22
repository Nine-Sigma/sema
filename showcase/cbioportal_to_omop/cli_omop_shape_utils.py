"""S1-08 CLI helpers: source-identity enum + value-decision loading (DuckDB).

Kept out of ``cli_fit_omop`` so the command module stays thin and these small,
pure readers are unit-testable without the Click layer.
"""

from __future__ import annotations

from typing import Any, Sequence

import duckdb

from sema.compile.compiler_utils import StagingDecision
from sema.compile.fk_closed_compiler_utils import ChildSourceSpec, RegistryJoinSpec
from sema.pipeline.fk_closed_fit import FkClosedFitRequest
from sema.resolve.engine_utils import staging_decision_from_value_mapping
from sema.resolve.identity_registry import DEFAULT_SCHEMA, DEFAULT_TABLE
from showcase.cbioportal_to_omop.omop_policy import MISSING_PERSON_KEY_REASON, make_omop_fk_specs
from sema.resolve.value_mapping_store import ValueMappingStore
from sema.resolve.value_mapping_store_utils import GRAIN_KEY, ValueMapping

__all__ = [
    "build_omop_shape_request",
    "enumerate_identity_source_duckdb",
    "load_staging_decisions",
]


def build_omop_shape_request(
    *,
    study_schema: str,
    source_table: str,
    value_column: str,
    patient_key_column: str,
    row_ref_column: str,
    omop_schema: str,
    keys: Sequence[str],
    row_count: int,
    decisions: Sequence[StagingDecision],
    run_id: str,
) -> FkClosedFitRequest:
    """Assemble one study's OMOP-shape request from CLI-supplied physical names."""
    parent, child, required = make_omop_fk_specs(omop_schema)
    source = ChildSourceSpec(
        schema=study_schema,
        table=source_table,
        value_column=value_column,
        row_ref_column=row_ref_column,
        patient_key_column=patient_key_column,
    )
    registry_spec = RegistryJoinSpec(
        schema=DEFAULT_SCHEMA,
        table=DEFAULT_TABLE,
        namespace_column="source_namespace",
        key_column="source_entity_key",
        id_column="entity_id",
    )
    return FkClosedFitRequest(
        source=source,
        source_row_count=row_count,
        distinct_patient_keys=list(keys),
        parent=parent,
        child=child,
        registry_spec=registry_spec,
        decisions=list(decisions),
        required_fields=required,
        no_map_default=0,
        missing_key_reason=MISSING_PERSON_KEY_REASON,
        run_id=run_id,
    )


def enumerate_identity_source_duckdb(
    conn: duckdb.DuckDBPyConnection, *, schema: str, table: str, patient_key_column: str
) -> tuple[list[str], int]:
    """Distinct non-blank patient keys + TOTAL source row count (row-count identity)."""
    tbl = f'"{schema}"."{table}"'
    col = f'"{patient_key_column}"'
    keys = [
        str(row[0])
        for row in conn.execute(
            f"SELECT DISTINCT {col} FROM {tbl} "
            f"WHERE TRIM(COALESCE({col}, '')) <> '' ORDER BY {col}"
        ).fetchall()
    ]
    row = conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()
    return keys, int(row[0]) if row else 0


def _dedupe_to_grain(mappings: list[ValueMapping]) -> list[ValueMapping]:
    """Collapse to one row per store grain (last-wins), matching the store PK."""
    by_grain: dict[tuple[Any, ...], ValueMapping] = {}
    for mapping in mappings:
        by_grain[tuple(getattr(mapping, col) for col in GRAIN_KEY)] = mapping
    return list(by_grain.values())


def load_staging_decisions(
    conn: duckdb.DuckDBPyConnection, *, policy_ref: str
) -> list[StagingDecision]:
    """Read the value-mapping store, scope to one resolver policy, project to decisions.

    Scoping to ``policy_ref`` (and collapsing to grain) mirrors ``run_fit``'s
    current-run scoping (bug-374): a stale row under another policy ref would
    otherwise inline a duplicate VALUES row and inflate the FK-closed join.
    """
    scoped = [
        m for m in ValueMappingStore(conn).read_all() if m.resolver_policy_ref == policy_ref
    ]
    return [staging_decision_from_value_mapping(m) for m in _dedupe_to_grain(scoped)]

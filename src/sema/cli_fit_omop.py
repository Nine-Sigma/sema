"""CLI: ``sema fit-omop-shape`` — materialize the FK-closed OMOP shape (S1-08).

Lands a production-shaped ``omop.person`` + ``omop.condition_occurrence`` for one
study, FK-closed against a materialized person, with full identity resolution.
``--backend duckdb`` runs the whole chain locally (registry + target in one file);
``--backend databricks`` bridges the DuckDB-canonical identity registry into a
Delta table, then writes person/condition via Delta ``REPLACE WHERE``. Resolved
value decisions come from the DuckDB value-mapping store (its canonical home).

No new chain logic — only the backend, the registry bridge, and the source enum
differ per warehouse.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import click
import duckdb

from sema.cli_fit_utils import (
    enumerate_identity_source_databricks,
    open_databricks_cursor,
)
from sema.cli_fit_omop_utils import (
    enumerate_identity_source_duckdb,
    load_staging_decisions,
)
from sema.compile.fk_backend import DATABRICKS_FK_BACKEND, DUCKDB_FK_BACKEND, FkBackend
from sema.compile.fk_closed_compiler_utils import ChildSourceSpec, RegistryJoinSpec
from sema.models.config import DatabricksConfig
from sema.pipeline.fit_omop_shape import (
    OmopShapeRequest,
    OmopShapeResult,
    run_omop_shape_fit,
)
from sema.resolve.identity_registry import (
    DEFAULT_SCHEMA,
    DEFAULT_TABLE,
    IdentityRegistry,
)
from sema.resolve.policies.omop import (
    MISSING_PERSON_KEY_REASON,
    OMOP_ONCOTREE_CONDITION_REF,
    make_omop_fk_specs,
)

_DEFAULT_DUCKDB = Path.home() / ".sema" / "poc.duckdb"


@click.command(name="fit-omop-shape")
@click.option("--backend", type=click.Choice(["duckdb", "databricks"]), default="duckdb", show_default=True)
@click.option("--duckdb", "duckdb_path", type=click.Path(dir_okay=False, path_type=Path), default=str(_DEFAULT_DUCKDB), show_default=True, help="Registry + value-mapping store (canonical); also the source for the duckdb backend.")
@click.option("--catalog", default="workspace", show_default=True, help="Databricks catalog.")
@click.option("--study-schema", required=True, help="Source study schema (the identity namespace).")
@click.option("--source-table", default="sample", show_default=True)
@click.option("--value-column", default="ONCOTREE_CODE", show_default=True)
@click.option("--patient-key-column", default="PATIENT_ID", show_default=True)
@click.option("--row-ref-column", default="SAMPLE_ID", show_default=True)
@click.option("--omop-schema", default="omop_stage_a", show_default=True, help="Target schema for person + condition_occurrence.")
@click.option("--policy-ref", default=OMOP_ONCOTREE_CONDITION_REF, show_default=True, help="resolver_policy_ref to select decisions from the store.")
@click.option("--run-id", default="s1-08", show_default=True)
@click.option("--strict/--no-strict", default=False, show_default=True, help="Exit 3 if Gate-D-lite fails.")
def fit_omop_shape_cmd(
    backend: str,
    duckdb_path: Path,
    catalog: str,
    study_schema: str,
    source_table: str,
    value_column: str,
    patient_key_column: str,
    row_ref_column: str,
    omop_schema: str,
    policy_ref: str,
    run_id: str,
    strict: bool,
) -> None:
    """Materialize the FK-closed OMOP shape for one study."""
    try:
        result = _run(
            backend=backend, duckdb_path=duckdb_path, catalog=catalog,
            study_schema=study_schema, source_table=source_table,
            value_column=value_column, patient_key_column=patient_key_column,
            row_ref_column=row_ref_column, omop_schema=omop_schema,
            policy_ref=policy_ref, run_id=run_id,
        )
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 — surface any wiring error to the user
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    click.echo(json.dumps(_summary(result, omop_schema), indent=2, default=str))
    if strict and not result.qa.passed:
        click.echo(f"STRICT FAIL — Gate-D-lite: {result.qa.outcome.value}", err=True)
        sys.exit(3)


def _build_request(
    *, study_schema: str, source_table: str, value_column: str,
    patient_key_column: str, row_ref_column: str, omop_schema: str,
    keys: list[str], row_count: int, decisions: list[Any], run_id: str,
) -> OmopShapeRequest:
    parent, child, required = make_omop_fk_specs(omop_schema)
    source = ChildSourceSpec(
        schema=study_schema, table=source_table, value_column=value_column,
        row_ref_column=row_ref_column, patient_key_column=patient_key_column,
    )
    registry_spec = RegistryJoinSpec(
        schema=DEFAULT_SCHEMA, table=DEFAULT_TABLE,
        namespace_column="source_namespace", key_column="source_entity_key",
        id_column="entity_id",
    )
    return OmopShapeRequest(
        source=source, source_row_count=row_count, distinct_patient_keys=keys,
        parent=parent, child=child, registry_spec=registry_spec,
        decisions=decisions, required_fields=required, no_map_default=0,
        missing_key_reason=MISSING_PERSON_KEY_REASON, run_id=run_id,
    )


def _run(
    *, backend: str, duckdb_path: Path, catalog: str, study_schema: str,
    source_table: str, value_column: str, patient_key_column: str,
    row_ref_column: str, omop_schema: str, policy_ref: str, run_id: str,
) -> OmopShapeResult:
    store_path = Path(duckdb_path).expanduser()
    if not store_path.exists():
        click.echo(f"Error: DuckDB file not found: {store_path}", err=True)
        sys.exit(2)
    store_conn = duckdb.connect(str(store_path))
    decisions = load_staging_decisions(store_conn, policy_ref=policy_ref)
    if not decisions:
        store_conn.close()
        click.echo(f"Error: no value-mapping decisions for policy {policy_ref!r}", err=True)
        sys.exit(2)

    if backend == "databricks":
        return _run_databricks(
            store_conn, catalog=catalog, study_schema=study_schema,
            source_table=source_table, value_column=value_column,
            patient_key_column=patient_key_column, row_ref_column=row_ref_column,
            omop_schema=omop_schema, decisions=decisions, run_id=run_id,
        )
    return _run_duckdb(
        store_conn, study_schema=study_schema, source_table=source_table,
        value_column=value_column, patient_key_column=patient_key_column,
        row_ref_column=row_ref_column, omop_schema=omop_schema,
        decisions=decisions, run_id=run_id,
    )


def _run_duckdb(
    conn: duckdb.DuckDBPyConnection, *, study_schema: str, source_table: str,
    value_column: str, patient_key_column: str, row_ref_column: str,
    omop_schema: str, decisions: list[Any], run_id: str,
) -> OmopShapeResult:
    keys, row_count = enumerate_identity_source_duckdb(
        conn, schema=study_schema, table=source_table,
        patient_key_column=patient_key_column,
    )
    request = _build_request(
        study_schema=study_schema, source_table=source_table,
        value_column=value_column, patient_key_column=patient_key_column,
        row_ref_column=row_ref_column, omop_schema=omop_schema, keys=keys,
        row_count=row_count, decisions=decisions, run_id=run_id,
    )
    try:
        return run_omop_shape_fit(
            request, registry=IdentityRegistry(conn), target_cursor=conn,
            backend=DUCKDB_FK_BACKEND, bridge=False,
        )
    finally:
        conn.close()


def _run_databricks(
    store_conn: duckdb.DuckDBPyConnection, *, catalog: str, study_schema: str,
    source_table: str, value_column: str, patient_key_column: str,
    row_ref_column: str, omop_schema: str, decisions: list[Any], run_id: str,
) -> OmopShapeResult:
    cursor = open_databricks_cursor(
        DatabricksConfig(), catalog=catalog, schema=study_schema
    )
    keys, row_count = enumerate_identity_source_databricks(
        cursor, schema=study_schema, table=source_table,
        patient_key_column=patient_key_column,
    )
    request = _build_request(
        study_schema=study_schema, source_table=source_table,
        value_column=value_column, patient_key_column=patient_key_column,
        row_ref_column=row_ref_column, omop_schema=omop_schema, keys=keys,
        row_count=row_count, decisions=decisions, run_id=run_id,
    )
    registry = IdentityRegistry(store_conn)  # DuckDB-canonical
    try:
        return run_omop_shape_fit(
            request, registry=registry, target_cursor=cursor,
            backend=DATABRICKS_FK_BACKEND, bridge=True,
        )
    finally:
        store_conn.close()


def _summary(result: OmopShapeResult, omop_schema: str) -> dict[str, object]:
    return {
        "omop_schema": omop_schema,
        "parent_rows": result.fk.parent_rows,
        "child_rows": result.fk.child_rows,
        "missing_key_rows": result.fk.missing_key_rows,
        "review_count": result.review_count,
        "registry_rows_bridged": result.registry_rows_bridged,
        "gate_d_lite": result.qa.as_dict(),
    }


__all__ = ["fit_omop_shape_cmd"]

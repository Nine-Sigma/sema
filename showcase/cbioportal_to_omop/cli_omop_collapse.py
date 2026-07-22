"""CLI: ``sema collapse-omop-identities`` — Stage B deterministic person collapse.

Once ≥2 studies of one institution are materialized (S1-08), the same patient
sequenced in each has been assigned a distinct ``person_id``. This command
collapses those onto a single canonical id when they share an exact institutional
key within one identity namespace (the safe, no-probability case), then rebuilds
``person`` + ``condition_occurrence`` for every study through the collapsed
registry: surrogate PKs are preserved, FKs recomputed, duplicate persons retired.

``--identity-namespace`` names the institutional id space the given studies share
(e.g. all MSK-DMP studies). Cross-namespace keys are never collapsed. The
DuckDB-canonical registry stays authoritative; ``--backend databricks`` bridges
it before the rebuild. Idempotent: a re-run collapses nothing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Sequence

import click
import duckdb

from showcase.cbioportal_to_omop.cli_omop_shape_utils import (
    build_omop_shape_request,
    enumerate_identity_source_duckdb,
    load_staging_decisions,
)
from sema.cli_fit_utils import (
    enumerate_identity_source_databricks,
    open_databricks_cursor,
)
from sema.compile.fk_backend import DATABRICKS_FK_BACKEND, DUCKDB_FK_BACKEND, FkBackend
from sema.models.config import DatabricksConfig
from sema.pipeline.fk_closed_fit import (
    FkClosedFitRequest,
    StageBResult,
    run_stage_b_collapse,
)
from sema.resolve.identity_registry import IdentityRegistry
from showcase.cbioportal_to_omop.omop_policy import OMOP_ONCOTREE_CONDITION_REF

_DEFAULT_DUCKDB = Path.home() / ".sema" / "poc.duckdb"


@click.command(name="collapse-omop-identities")
@click.option("--backend", type=click.Choice(["duckdb", "databricks"]), default="duckdb", show_default=True)
@click.option("--duckdb", "duckdb_path", type=click.Path(dir_okay=False, path_type=Path), default=str(_DEFAULT_DUCKDB), show_default=True, help="Registry + value-mapping store (canonical); also the source for the duckdb backend.")
@click.option("--catalog", default="workspace", show_default=True)
@click.option("--study-schema", "study_schemas", required=True, multiple=True, help="A study schema to collapse; pass once per study (≥2 to have anything to collapse).")
@click.option("--identity-namespace", required=True, help="The institutional id namespace the given studies share (shared keys across them are one person).")
@click.option("--source-table", default="sample", show_default=True)
@click.option("--value-column", default="ONCOTREE_CODE", show_default=True)
@click.option("--patient-key-column", default="PATIENT_ID", show_default=True)
@click.option("--row-ref-column", default="SAMPLE_ID", show_default=True)
@click.option("--omop-schema", default="omop_stage_a", show_default=True)
@click.option("--policy-ref", default=OMOP_ONCOTREE_CONDITION_REF, show_default=True)
@click.option("--run-id", default="s1-10", show_default=True)
@click.option("--strict/--no-strict", default=False, show_default=True, help="Exit 3 if any study's Gate-D-lite fails.")
def collapse_omop_identities_cmd(
    backend: str,
    duckdb_path: Path,
    catalog: str,
    study_schemas: tuple[str, ...],
    identity_namespace: str,
    source_table: str,
    value_column: str,
    patient_key_column: str,
    row_ref_column: str,
    omop_schema: str,
    policy_ref: str,
    run_id: str,
    strict: bool,
) -> None:
    """Collapse shared-identity persons across the given studies, then rebuild."""
    try:
        result = _run(
            backend=backend, duckdb_path=duckdb_path, catalog=catalog,
            study_schemas=study_schemas, identity_namespace=identity_namespace,
            source_table=source_table, value_column=value_column,
            patient_key_column=patient_key_column, row_ref_column=row_ref_column,
            omop_schema=omop_schema, policy_ref=policy_ref, run_id=run_id,
        )
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 — surface any wiring error to the user
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    click.echo(json.dumps(_summary(result, omop_schema), indent=2, default=str))
    if strict and not all(qa.passed for qa in result.qa):
        click.echo("STRICT FAIL — Gate-D-lite failed for at least one study", err=True)
        sys.exit(3)


def _run(
    *, backend: str, duckdb_path: Path, catalog: str,
    study_schemas: Sequence[str], identity_namespace: str, source_table: str,
    value_column: str, patient_key_column: str, row_ref_column: str,
    omop_schema: str, policy_ref: str, run_id: str,
) -> StageBResult:
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

    common = dict(
        source_table=source_table, value_column=value_column,
        patient_key_column=patient_key_column, row_ref_column=row_ref_column,
        omop_schema=omop_schema, decisions=decisions, run_id=run_id,
    )
    grouping = {schema: identity_namespace for schema in study_schemas}
    if backend == "databricks":
        return _run_databricks(
            store_conn, catalog=catalog, study_schemas=study_schemas,
            grouping=grouping, **common,
        )
    return _run_duckdb(
        store_conn, study_schemas=study_schemas, grouping=grouping, **common
    )


def _requests_duckdb(
    conn: duckdb.DuckDBPyConnection, study_schemas: Sequence[str], **common: Any
) -> list[FkClosedFitRequest]:
    requests = []
    for schema in study_schemas:
        keys, row_count = enumerate_identity_source_duckdb(
            conn, schema=schema, table=common["source_table"],
            patient_key_column=common["patient_key_column"],
        )
        requests.append(
            build_omop_shape_request(
                study_schema=schema, keys=keys, row_count=row_count, **common
            )
        )
    return requests


def _run_duckdb(
    conn: duckdb.DuckDBPyConnection, *, study_schemas: Sequence[str],
    grouping: dict[str, str], **common: Any,
) -> StageBResult:
    requests = _requests_duckdb(conn, study_schemas, **common)
    try:
        return run_stage_b_collapse(
            requests, namespace_grouping=grouping,
            registry=IdentityRegistry(conn), target_cursor=conn,
            backend=DUCKDB_FK_BACKEND, bridge=False,
        )
    finally:
        conn.close()


def _run_databricks(
    store_conn: duckdb.DuckDBPyConnection, *, catalog: str,
    study_schemas: Sequence[str], grouping: dict[str, str], **common: Any,
) -> StageBResult:
    cursor = open_databricks_cursor(
        DatabricksConfig(), catalog=catalog, schema=study_schemas[0]
    )
    requests = []
    for schema in study_schemas:
        keys, row_count = enumerate_identity_source_databricks(
            cursor, schema=schema, table=common["source_table"],
            patient_key_column=common["patient_key_column"],
        )
        requests.append(
            build_omop_shape_request(
                study_schema=schema, keys=keys, row_count=row_count, **common
            )
        )
    try:
        return run_stage_b_collapse(
            requests, namespace_grouping=grouping,
            registry=IdentityRegistry(store_conn), target_cursor=cursor,
            backend=DATABRICKS_FK_BACKEND, bridge=True,
        )
    finally:
        store_conn.close()


def _summary(result: StageBResult, omop_schema: str) -> dict[str, object]:
    return {
        "omop_schema": omop_schema,
        "collapsed_person_count": result.collapse.collapsed_person_count,
        "remapped_row_count": result.collapse.remapped_row_count,
        "retired_person_ids": list(result.collapse.retired_entity_ids),
        "parent_rows": result.fk.parent_rows,
        "child_rows": result.fk.child_rows,
        "registry_rows_bridged": result.registry_rows_bridged,
        "gate_d_lite": [qa.as_dict() for qa in result.qa],
    }


__all__ = ["collapse_omop_identities_cmd"]

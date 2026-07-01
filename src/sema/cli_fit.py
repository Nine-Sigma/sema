"""CLI: ``sema fit`` — run the Slice-0 spine end-to-end (DuckDB or Databricks).

Consumes the already-materialised TARGET binding (authored by ``sema target
load``, US-007) from its manifest and runs resolve -> produce -> assemble ->
compile -> staging-write -> Gate D-lite QA -> eval for one study.

``--backend duckdb`` (default, US-012A) runs the whole chain locally against the
``--duckdb`` file. ``--backend databricks`` (US-013) reads the OMOP vocabulary
and source study from the live workspace and writes the §1.5(b) staging table to
Databricks via an atomic Delta ``REPLACE WHERE``; the value-mapping store stays
on DuckDB (its canonical home, US-005). No new chain logic — only the staging
backend + connections differ.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import click
import duckdb

from sema.cli_fit_utils import enumerate_source_databricks, open_databricks_cursor
from sema.compile.staging_backend import (
    DATABRICKS_BACKEND,
    DUCKDB_BACKEND,
    StagingBackend,
)
from sema.eval.mapping_goldset import GoldSet, load_gold_set
from sema.eval.mapping_report_utils import AcceptanceVerdict
from sema.models.config import DatabricksConfig
from sema.pipeline.fit_slice0 import FitResult, run_fit
from sema.pipeline.fit_slice0_utils import (
    build_slice0_fit_request,
    discover_study,
    enumerate_source,
)
from sema.resolve.engine import VocabularyResolver
from sema.resolve.policies.omop import OMOP_VOCAB_SCHEMA
from sema.resolve.vocab_store import (
    VocabStore,
    VocabStoreBackend,
    namespace_for_backend,
)

_DEFAULT_DUCKDB = Path.home() / ".sema" / "poc.duckdb"


@click.command(name="fit")
@click.option(
    "--manifest",
    "manifest_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Authored TARGET manifest carrying the vocabulary binding (US-007).",
)
@click.option(
    "--backend",
    type=click.Choice(["duckdb", "databricks"]),
    default="duckdb",
    show_default=True,
    help="Staging warehouse: local DuckDB (US-012A) or live Databricks (US-013).",
)
@click.option(
    "--duckdb",
    "duckdb_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=str(_DEFAULT_DUCKDB),
    show_default=True,
    help="DuckDB file: the source+vocab (duckdb backend) or value-mapping store.",
)
@click.option("--catalog", default=None, help="Databricks catalog (default: workspace).")
@click.option("--study-schema", default=None, help="Source schema (auto-discovered if omitted).")
@click.option("--source-table", default="sample", show_default=True, help="Source table name.")
@click.option("--value-column", default="ONCOTREE_CODE", show_default=True, help="Source code column.")
@click.option(
    "--gold",
    "gold_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Gold-set JSONL for Gate D-lite / eval reconciliation (optional).",
)
@click.option("--staging-schema", default="sema_staging", show_default=True)
@click.option("--staging-table", default="condition_staging", show_default=True)
@click.option(
    "--strict/--no-strict",
    default=False,
    show_default=True,
    help="Exit non-zero (3) if Gate D-lite fails or the eval verdict is not ACCEPTED.",
)
def fit_cmd(
    manifest_path: Path,
    backend: str,
    duckdb_path: Path,
    catalog: str | None,
    study_schema: str | None,
    source_table: str,
    value_column: str,
    gold_path: Path | None,
    staging_schema: str,
    staging_table: str,
    strict: bool,
) -> None:
    """Run the full resolve->...->eval chain for one study."""
    gold = GoldSet(rows=load_gold_set(gold_path)) if gold_path else GoldSet(rows=[])
    common: dict[str, Any] = dict(
        manifest_path=manifest_path,
        study_schema=study_schema,
        source_table=source_table,
        value_column=value_column,
        gold=gold,
        staging_schema=staging_schema,
        staging_table=staging_table,
    )
    try:
        if backend == "databricks":
            result = _run_databricks(duckdb_path, catalog, **common)
        else:
            result = _run_duckdb(duckdb_path, **common)
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 — surface any wiring error to the user
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    click.echo(json.dumps(_summary(result), indent=2, default=str))
    if strict:
        _enforce_strict(result)


def _enforce_strict(result: FitResult) -> None:
    """Exit 3 when the run is not clean (Gate D-lite fail or verdict != ACCEPTED)."""
    reasons = []
    if not result.qa.passed:
        reasons.append(f"Gate D-lite: {result.qa.outcome.value}")
    verdict = result.report.verdict
    if verdict is not AcceptanceVerdict.ACCEPTED:
        reasons.append(f"eval verdict: {verdict.value}")
    if reasons:
        click.echo("STRICT FAIL — " + "; ".join(reasons), err=True)
        sys.exit(3)


def _run_duckdb(duckdb_path: Path, *, manifest_path: Path, study_schema: str | None,
                source_table: str, value_column: str, gold: GoldSet,
                staging_schema: str, staging_table: str) -> FitResult:
    path = Path(duckdb_path).expanduser()
    if not path.exists():
        click.echo(f"Error: DuckDB file not found: {path}", err=True)
        sys.exit(2)
    conn = duckdb.connect(str(path))
    try:
        schema = study_schema or _discover(conn, value_column, source_table)
        codes, row_count = enumerate_source(
            conn, schema=schema, table=source_table, value_column=value_column
        )
        store = VocabStore(conn, schema=OMOP_VOCAB_SCHEMA, namespace="vocabulary_omop")
        return _execute(
            store, conn, conn, DUCKDB_BACKEND,
            manifest_path=manifest_path, schema=schema, source_table=source_table,
            value_column=value_column, codes=codes, row_count=row_count, gold=gold,
            staging_schema=staging_schema, staging_table=staging_table,
        )
    finally:
        conn.close()


def _run_databricks(duckdb_path: Path, catalog: str | None, *, manifest_path: Path,
                    study_schema: str | None, source_table: str, value_column: str,
                    gold: GoldSet, staging_schema: str, staging_table: str) -> FitResult:
    if not study_schema:
        click.echo("Error: --study-schema is required for the databricks backend", err=True)
        sys.exit(2)
    cursor = open_databricks_cursor(
        DatabricksConfig(), catalog=catalog or "workspace", schema=study_schema
    )
    store_conn = duckdb.connect(str(Path(duckdb_path).expanduser()))
    try:
        codes, row_count = enumerate_source_databricks(
            cursor, schema=study_schema, table=source_table, value_column=value_column
        )
        vocab = VocabStore(
            cursor,
            schema=OMOP_VOCAB_SCHEMA,
            namespace=namespace_for_backend(VocabStoreBackend.DATABRICKS),
            dialect="databricks",
        )
        return _execute(
            vocab, store_conn, cursor, DATABRICKS_BACKEND,
            manifest_path=manifest_path, schema=study_schema, source_table=source_table,
            value_column=value_column, codes=codes, row_count=row_count, gold=gold,
            staging_schema=staging_schema, staging_table=staging_table,
        )
    finally:
        store_conn.close()


def _execute(vocab_store: VocabStore, store_conn: duckdb.DuckDBPyConnection,
             staging_conn: Any, backend: StagingBackend, *, manifest_path: Path,
             schema: str, source_table: str, value_column: str, codes: list[str],
             row_count: int, gold: GoldSet, staging_schema: str,
             staging_table: str) -> FitResult:
    policy, request = build_slice0_fit_request(
        manifest_path=manifest_path,
        source_schema=schema,
        source_table=source_table,
        value_column=value_column,
        source_codes=codes,
        source_row_count=row_count,
        gold=gold,
        staging_schema=staging_schema,
        staging_table=staging_table,
    )
    resolver = VocabularyResolver(vocab_store, policy)
    return run_fit(
        resolver, request, value_mapping_conn=store_conn,
        staging_conn=staging_conn, staging_backend=backend,
    )


def _discover(
    conn: duckdb.DuckDBPyConnection, value_column: str, source_table: str
) -> str:
    found = discover_study(conn, value_column=value_column, source_table=source_table)
    if found is None:
        raise click.ClickException(
            f"no schema has {source_table}.{value_column}; pass --study-schema"
        )
    return found[0]


def _summary(result: FitResult) -> dict[str, object]:
    return {
        "rows_staged": result.rows_staged,
        "source_row_count": result.source_row_count,
        "staging": f"{result.staging_schema}.{result.staging_table}",
        "gate_d_lite": result.qa.as_dict(),
        "eval": result.report.as_dict(),
    }


__all__ = ["fit_cmd"]

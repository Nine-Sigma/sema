"""CLI: ``sema fit`` — run the local Slice-0 spine end-to-end on DuckDB.

Consumes the already-materialised TARGET binding (authored by ``sema target
load``, US-007) from its manifest and runs resolve -> produce -> assemble ->
compile -> staging-write -> Gate D-lite QA -> eval for one study against the
DuckDB backend. Writes NO Databricks objects (US-013 is the live Databricks
gate). The value-mapping store and §1.5(b) staging table are written into the
``--duckdb`` file.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import duckdb

from sema.eval.mapping_goldset import GoldSet, load_gold_set
from sema.pipeline.fit_slice0 import FitResult, run_fit
from sema.pipeline.fit_slice0_utils import (
    build_slice0_fit_request,
    discover_study,
    enumerate_source,
)
from sema.resolve.engine import VocabularyResolver
from sema.resolve.policies.omop import OMOP_VOCAB_SCHEMA
from sema.resolve.vocab_store import VocabStore

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
    "--duckdb",
    "duckdb_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=str(_DEFAULT_DUCKDB),
    show_default=True,
    help="DuckDB file holding the OMOP vocabulary + cBioPortal source study.",
)
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
def fit_cmd(
    manifest_path: Path,
    duckdb_path: Path,
    study_schema: str | None,
    source_table: str,
    value_column: str,
    gold_path: Path | None,
    staging_schema: str,
    staging_table: str,
) -> None:
    """Run the full resolve->...->eval chain for one study on DuckDB."""
    path = Path(duckdb_path).expanduser()
    if not path.exists():
        click.echo(f"Error: DuckDB file not found: {path}", err=True)
        sys.exit(2)
    conn = duckdb.connect(str(path))
    try:
        result = _run(
            conn,
            manifest_path=manifest_path,
            study_schema=study_schema,
            source_table=source_table,
            value_column=value_column,
            gold_path=gold_path,
            staging_schema=staging_schema,
            staging_table=staging_table,
        )
    except Exception as exc:  # noqa: BLE001 — surface any wiring error to the user
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    finally:
        conn.close()
    click.echo(json.dumps(_summary(result), indent=2, default=str))


def _run(
    conn: duckdb.DuckDBPyConnection,
    *,
    manifest_path: Path,
    study_schema: str | None,
    source_table: str,
    value_column: str,
    gold_path: Path | None,
    staging_schema: str,
    staging_table: str,
) -> FitResult:
    schema = study_schema or _discover(conn, value_column, source_table)
    codes, row_count = enumerate_source(
        conn, schema=schema, table=source_table, value_column=value_column
    )
    gold = GoldSet(rows=load_gold_set(gold_path)) if gold_path else GoldSet(rows=[])
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
    resolver = VocabularyResolver(
        VocabStore(conn, schema=OMOP_VOCAB_SCHEMA, namespace="vocabulary_omop"),
        policy,
    )
    return run_fit(resolver, request, value_mapping_conn=conn, staging_conn=conn)


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

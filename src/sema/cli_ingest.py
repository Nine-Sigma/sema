from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Sequence

import click

from sema.ingest.comment_recovery import (
    LiveTableComments,
    ParsedTableComments,
    PartialOverrideError,
    RecoveryReport,
    StudyCacheMissingError,
    StudyNotRegisteredError,
    build_recovery_plan,
    execute_recovery_plan,
    read_databricks_comments,
    resolve_recovery_context,
)
from sema.ingest.databricks_push import Bridge, PushError
from sema.ingest.duckdb_staging import Staging
from sema.ingest.study_registry import StudyRegistry
from sema.log import logger
from sema.models.config import IngestConfig

SUPPORTED_PUSH_TARGETS: frozenset[str] = frozenset({"databricks"})


def _load_ingest_config(duckdb_path: str | None) -> IngestConfig:
    config = IngestConfig()
    if duckdb_path:
        config.duckdb_path = duckdb_path
    return config


def _open_staging(duckdb_path: str | None) -> Staging:
    config = _load_ingest_config(duckdb_path)
    return Staging(config.duckdb_path)


@click.group()
def ingest() -> None:
    """Ingest source data and target ontologies into local DuckDB staging."""


@ingest.command("cbioportal")
@click.argument("study_id")
@click.option(
    "--cache-dir",
    "cache_dir",
    default=None,
    help="Directory for caching downloaded cBioPortal study files.",
)
@click.option(
    "--duckdb-path",
    "duckdb_path",
    default=None,
    help="Override DuckDB staging file path.",
)
def ingest_cbioportal_cmd(
    study_id: str,
    cache_dir: str | None,
    duckdb_path: str | None,
) -> None:
    """Download, parse, and stage a cBioPortal study into the DuckDB staging file."""
    try:
        from showcase.cbioportal_to_omop.parsers import ingest_study
    except ImportError as err:
        raise click.ClickException(
            "The cBioPortal showcase is not importable. Run from a source "
            "checkout where the 'showcase/' directory is on sys.path."
        ) from err
    config = _load_ingest_config(duckdb_path)
    resolved_cache = Path(cache_dir).expanduser() if cache_dir else Path(
        config.cache_dir
    ).expanduser()
    staging = Staging(config.duckdb_path)
    try:
        ingest_study(study_id=study_id, staging=staging, cache_dir=resolved_cache)
    finally:
        staging.close()


@ingest.command("omop")
@click.option("--cdm-version", "cdm_version", default="5.4", help="OMOP CDM version tag.")
@click.option(
    "--vocab-path",
    "vocab_path",
    default=None,
    help="Path to Athena OMOP vocabulary bundle directory.",
)
@click.option(
    "--duckdb-path",
    "duckdb_path",
    default=None,
    help="Override DuckDB staging file path.",
)
def ingest_omop_cmd(
    cdm_version: str,
    vocab_path: str | None,
    duckdb_path: str | None,
) -> None:
    """Download OMOP CDM schema and (optionally) load Athena vocabulary into DuckDB staging."""
    try:
        from showcase.cbioportal_to_omop.omop_ingest import (
            ingest_cdm_schema,
            ingest_vocabulary,
        )
    except ImportError:
        raise click.ClickException(
            "The cBioPortal→OMOP showcase is not importable. Run from a source "
            "checkout where the 'showcase/' directory is on sys.path."
        ) from None
    staging = Staging(_load_ingest_config(duckdb_path).duckdb_path)
    try:
        ingest_cdm_schema(version=cdm_version, staging=staging)
        ingest_vocabulary(
            vocab_path=Path(vocab_path).expanduser() if vocab_path else None,
            staging=staging,
        )
    finally:
        staging.close()


def _extract_study_comments_lazy(
    study_dir: Path,
) -> dict[str, ParsedTableComments]:
    try:
        from showcase.cbioportal_to_omop.comment_extract import (
            extract_study_comments,
        )
    except ImportError as err:
        raise click.ClickException(
            "The cBioPortal showcase is not importable. Run from a source "
            "checkout where the 'showcase/' directory is on sys.path."
        ) from err
    return extract_study_comments(study_dir)


def _open_recovery_executor(
    config: IngestConfig,
) -> tuple[Callable[[str], None], Callable[[str, list[str]], Sequence[Sequence[Any]]]]:
    creds = config.databricks_creds
    from databricks import sql as databricks_sql
    conn = databricks_sql.connect(
        server_hostname=creds.host.replace("https://", ""),
        http_path=creds.http_path,
        access_token=creds.token.get_secret_value(),
    )

    def execute(sql: str) -> None:
        cursor = conn.cursor()
        try:
            cursor.execute(sql)
        finally:
            cursor.close()

    def query(sql: str, params: list[str]) -> Sequence[Sequence[Any]]:
        cursor = conn.cursor()
        try:
            cursor.execute(sql, params)
            return cursor.fetchall()
        finally:
            cursor.close()

    return execute, query


def _emit_summary(
    study_id: str | None,
    target_catalog: str,
    target_schema: str,
    report: RecoveryReport,
    as_json: bool,
) -> None:
    if as_json:
        payload = {
            "study_id": study_id,
            "target_catalog": target_catalog,
            "target_schema": target_schema,
            "tables_visited": _count_tables_visited(report),
            "columns_updated": report.columns_updated,
            "columns_skipped": report.columns_skipped,
            "columns_failed": report.columns_failed,
            "table_comments_updated": report.table_comments_updated,
        }
        click.echo(json.dumps(payload))
        return
    click.echo("\nRecovery Report")
    click.echo("=" * 40)
    click.echo(f"  Study: {study_id or '<override>'}")
    click.echo(f"  Target: {target_catalog}.{target_schema}")
    click.echo(f"  Columns updated: {report.columns_updated}")
    click.echo(f"  Table comments updated: {report.table_comments_updated}")
    click.echo(f"  Columns skipped: {report.columns_skipped}")
    click.echo(f"  Columns failed: {report.columns_failed}")
    if report.failed:
        click.echo("  Failures:")
        for f in report.failed:
            click.echo(f"    {f.table}.{f.column}: {f.error}")


def _count_tables_visited(report: RecoveryReport) -> int:
    tables: set[str] = set()
    for s in report.skipped:
        tables.add(s.table)
    for f in report.failed:
        tables.add(f.table)
    return len(tables)


@ingest.command("recover-comments")
@click.option("--study", "study_id", default=None, help="cBioPortal study_id.")
@click.option("--source-cache", "source_cache", default=None,
              help="Override path to the local cBioPortal cache.")
@click.option("--target-catalog", "target_catalog", default=None,
              help="Override Databricks catalog.")
@click.option("--target-schema", "target_schema", default=None,
              help="Override Databricks schema.")
@click.option("--cache-dir", "cache_dir", default=None,
              help="Override IngestConfig.cache_dir for this run.")
@click.option("--duckdb-path", "duckdb_path", default=None,
              help="Override DuckDB staging file path.")
@click.option("--dry-run", is_flag=True, default=False,
              help="Print SQL without executing.")
@click.option("--force", is_flag=True, default=False,
              help="Overwrite existing comments.")
@click.option("--json", "as_json", is_flag=True, default=False,
              help="Emit a JSON summary on stdout.")
def recover_comments_cmd(
    study_id: str | None,
    source_cache: str | None,
    target_catalog: str | None,
    target_schema: str | None,
    cache_dir: str | None,
    duckdb_path: str | None,
    dry_run: bool,
    force: bool,
    as_json: bool,
) -> None:
    """Re-apply parser-extracted column and table comments to Databricks."""
    config = _load_ingest_config(duckdb_path)
    if cache_dir:
        config.cache_dir = cache_dir
    staging = Staging(config.duckdb_path)
    try:
        registry = StudyRegistry(staging)
        try:
            ctx = resolve_recovery_context(
                study_id=study_id, registry=registry, ingest_config=config,
                source_cache_override=Path(source_cache) if source_cache else None,
                target_catalog_override=target_catalog,
                target_schema_override=target_schema,
            )
        except StudyNotRegisteredError as err:
            raise click.ClickException(str(err)) from err
        except StudyCacheMissingError as err:
            raise click.ClickException(str(err)) from err
        except PartialOverrideError as err:
            raise click.UsageError(str(err)) from err

        parsed = _extract_study_comments_lazy(ctx.source_cache)
        if dry_run:
            executor: Callable[[str], None] = lambda _sql: None
            query_fn: Callable[..., Sequence[Sequence[Any]]] = (
                lambda *_a, **_k: []
            )
            try:
                executor, query_fn = _open_recovery_executor(config)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Skipping live read in --dry-run (no Databricks): {}", exc
                )
                live: dict[str, LiveTableComments] = {}
            else:
                live = read_databricks_comments(
                    ctx.target_catalog, ctx.target_schema, query_fn,
                )
        else:
            executor, query_fn = _open_recovery_executor(config)
            live = read_databricks_comments(
                ctx.target_catalog, ctx.target_schema, query_fn,
            )
        plan = build_recovery_plan(ctx, parsed, live, force=force)
        report = execute_recovery_plan(plan, executor, dry_run=dry_run)
        _emit_summary(
            ctx.study_id, ctx.target_catalog, ctx.target_schema, report, as_json,
        )
        if report.columns_failed > 0:
            raise click.ClickException(
                f"{report.columns_failed} column(s) failed during recovery."
            )
    finally:
        staging.close()


@click.command("push")
@click.option("--target", default="databricks", help="Push target (databricks only).")
@click.option(
    "--schemas",
    "schemas_csv",
    default=None,
    help="Comma-separated subset of schemas to push (default: all).",
)
@click.option(
    "--duckdb-path",
    "duckdb_path",
    default=None,
    help="Override DuckDB staging file path.",
)
def push_cmd(target: str, schemas_csv: str | None, duckdb_path: str | None) -> None:
    """Push staged DuckDB tables to the target Databricks workspace."""
    if target not in SUPPORTED_PUSH_TARGETS:
        raise click.UsageError(
            f"Unsupported push target '{target}'. Supported: {sorted(SUPPORTED_PUSH_TARGETS)}"
        )
    schemas = [s.strip() for s in schemas_csv.split(",")] if schemas_csv else None
    config = _load_ingest_config(duckdb_path)
    staging = Staging(config.duckdb_path)
    try:
        bridge = Bridge(config, staging=staging)
        try:
            results = bridge.push_schemas(schemas)
            for r in results:
                logger.info(
                    "pushed {}.{} via {} ({} rows, target {})",
                    r.schema, r.table, r.mechanism, r.rows_pushed, r.target_count,
                )
        except PushError as err:
            logger.error("{}", err)
            raise click.ClickException(str(err))
        finally:
            bridge.close()
    finally:
        staging.close()

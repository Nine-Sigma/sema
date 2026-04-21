from __future__ import annotations

from pathlib import Path

import click

from sema.ingest.cbioportal import ingest_study
from sema.ingest.databricks_push import Bridge, PushError
from sema.ingest.duckdb_staging import Staging
from sema.ingest.omop import ingest_cdm_schema, ingest_vocabulary
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
    staging = Staging(_load_ingest_config(duckdb_path).duckdb_path)
    try:
        ingest_cdm_schema(version=cdm_version, staging=staging)
        ingest_vocabulary(
            vocab_path=Path(vocab_path).expanduser() if vocab_path else None,
            staging=staging,
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

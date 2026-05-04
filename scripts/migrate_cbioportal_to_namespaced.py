"""DuckDB migration: rename legacy `cbioportal` schema to namespaced equivalent.

Renames the local DuckDB staging schema and registers the renamed schema in
`_sema_study_registry` so push discovery picks it up by default.

Idempotent: safe to re-run. No-ops if the legacy schema is already absent
or the namespaced schema already exists.

Usage:
    uv run python scripts/migrate_cbioportal_to_namespaced.py \\
        --duckdb-path ~/.sema/staging.duckdb \\
        --study-id gbm_tcga_pan_can_atlas_2018
"""
from __future__ import annotations

import sys
from pathlib import Path

import click

from sema.ingest.duckdb_staging import Staging
from sema.ingest.naming import sanitize_schema_name
from sema.ingest.study_registry import StudyCollisionError, StudyRegistry
from sema.log import logger
from sema.models.config import IngestConfig

LEGACY_SCHEMA = "cbioportal"
PREFIX = "cbioportal"


def _has_schema(staging: Staging, name: str) -> bool:
    row = staging.execute(
        "SELECT count(*) FROM information_schema.schemata WHERE schema_name = ?",
        [name],
    ).fetchone()
    return bool(row and row[0])


def _rename_schema(staging: Staging, src: str, dst: str) -> None:
    """DuckDB has no ALTER SCHEMA RENAME — emulate via copy + drop.

    Uses CREATE TABLE ... AS to preserve column types; column comments and
    table comments do not survive the rebuild but are recoverable from
    sema's catalog metadata.
    """
    staging.execute(f'CREATE SCHEMA IF NOT EXISTS "{dst}"')
    rows = staging.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = ?",
        [src],
    ).fetchall()
    for (table,) in rows:
        staging.execute(
            f'CREATE TABLE "{dst}"."{table}" AS SELECT * FROM "{src}"."{table}"'
        )
        staging.execute(f'DROP TABLE "{src}"."{table}"')
    staging.execute(f'DROP SCHEMA "{src}"')


def _resolve_target(study_id: str) -> str:
    return sanitize_schema_name(PREFIX, study_id)


def _backfill_registry(staging: Staging, schema: str, study_id: str) -> None:
    registry = StudyRegistry(staging)
    try:
        registry.register(
            schema_name=schema,
            original_study_id=study_id,
            source_type="cbioportal",
        )
    except StudyCollisionError as err:
        raise click.ClickException(str(err)) from err


@click.command()
@click.option("--duckdb-path", "duckdb_path", default=None, help="DuckDB staging file path.")
@click.option(
    "--study-id",
    "study_id",
    required=True,
    help="Original cBioPortal study_id whose data lives in the legacy `cbioportal` schema.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Report what would change without executing the rename.",
)
def main(duckdb_path: str | None, study_id: str, dry_run: bool) -> None:
    """Rename `cbioportal` -> namespaced schema in DuckDB and backfill registry."""
    config = IngestConfig()
    if duckdb_path:
        config.duckdb_path = duckdb_path
    target_schema = _resolve_target(study_id)
    logger.info(
        "Migration target: schema {} -> {} in {}",
        LEGACY_SCHEMA, target_schema, config.duckdb_path,
    )

    staging = Staging(config.duckdb_path)
    try:
        legacy_present = _has_schema(staging, LEGACY_SCHEMA)
        target_present = _has_schema(staging, target_schema)

        if not legacy_present and target_present:
            logger.info(
                "Already migrated: {} absent, {} present. Backfilling registry only.",
                LEGACY_SCHEMA, target_schema,
            )
            if not dry_run:
                _backfill_registry(staging, target_schema, study_id)
            return

        if not legacy_present:
            logger.info("Nothing to migrate: schema {} not present.", LEGACY_SCHEMA)
            if not dry_run:
                _backfill_registry(staging, target_schema, study_id)
            return

        if target_present:
            raise click.ClickException(
                f"Both {LEGACY_SCHEMA} and {target_schema} schemas exist; "
                "manual reconciliation required (the migration cannot merge schemas)."
            )

        if dry_run:
            logger.info(
                "DRY RUN: would ALTER SCHEMA {} RENAME TO {} and register {}",
                LEGACY_SCHEMA, target_schema, study_id,
            )
            return

        _rename_schema(staging, LEGACY_SCHEMA, target_schema)
        _backfill_registry(staging, target_schema, study_id)
        logger.info("Migration complete: {} -> {}", LEGACY_SCHEMA, target_schema)
    finally:
        staging.close()


if __name__ == "__main__":
    sys.exit(main(standalone_mode=True))

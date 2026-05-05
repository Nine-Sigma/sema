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
from typing import Callable

import click

from sema.ingest.comment_recovery import ParsedTableComments
from sema.ingest.duckdb_staging import Staging
from sema.ingest.duckdb_staging_utils import (
    build_column_comment_sql,
    build_table_comment_sql,
)
from sema.ingest.naming import sanitize_schema_name
from sema.ingest.study_registry import StudyCollisionError, StudyRegistry
from sema.log import logger
from sema.models.config import IngestConfig

LEGACY_SCHEMA = "cbioportal"
PREFIX = "cbioportal"

CommentSource = Callable[[str], ParsedTableComments]


def _has_schema(staging: Staging, name: str) -> bool:
    row = staging.execute(
        "SELECT count(*) FROM information_schema.schemata WHERE schema_name = ?",
        [name],
    ).fetchone()
    return bool(row and row[0])


def _rename_schema(
    staging: Staging,
    src: str,
    dst: str,
    *,
    comment_source: CommentSource | None = None,
) -> None:
    """DuckDB has no ALTER SCHEMA RENAME — emulate via copy + drop.

    `comment_source` is an optional callable that returns
    `ParsedTableComments` for a given table name. When provided, parser
    comments are re-applied after each `CREATE TABLE ... AS SELECT *`
    so the rename round-trip preserves comments.
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
        _reapply_comments(staging, dst, table, comment_source)
    staging.execute(f'DROP SCHEMA "{src}"')


def _reapply_comments(
    staging: Staging, schema: str, table: str,
    comment_source: CommentSource | None,
) -> None:
    if comment_source is None:
        return
    try:
        parsed = comment_source(table)
    except Exception as err:  # noqa: BLE001
        logger.warning(
            "Comment source unavailable for {}.{}: {}; "
            "run `sema ingest recover-comments` to restore later.",
            schema, table, err,
        )
        return
    for column, comment in parsed.column_comments.items():
        if comment:
            staging.execute(
                build_column_comment_sql(schema, table, column, comment)
            )
    if parsed.table_comment:
        staging.execute(
            build_table_comment_sql(schema, table, parsed.table_comment)
        )


def _default_comment_source(
    cache_dir: Path, study_id: str,
) -> CommentSource | None:
    study_dir = cache_dir / study_id
    if not study_dir.exists():
        logger.warning(
            "cBioPortal source cache not found at {}; migration will run "
            "without comments. Run `sema ingest recover-comments --study {}` "
            "later to restore.",
            study_dir, study_id,
        )
        return None
    try:
        from showcase.cbioportal_to_omop.comment_extract import (
            extract_study_comments,
        )
    except ImportError as err:
        logger.warning(
            "showcase parser not importable ({}); skipping comment recovery.",
            err,
        )
        return None
    parsed = extract_study_comments(study_dir)
    return lambda table: parsed.get(
        table, ParsedTableComments(table_comment=None, column_comments={})
    )


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

        comment_source = _default_comment_source(
            Path(config.cache_dir).expanduser(), study_id,
        )
        _rename_schema(
            staging, LEGACY_SCHEMA, target_schema,
            comment_source=comment_source,
        )
        _backfill_registry(staging, target_schema, study_id)
        logger.info("Migration complete: {} -> {}", LEGACY_SCHEMA, target_schema)
    finally:
        staging.close()


if __name__ == "__main__":
    sys.exit(main(standalone_mode=True))

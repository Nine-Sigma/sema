from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from sema.ingest.databricks_push_utils import (
    build_alter_column_comment_sql,
    build_alter_table_comment_sql,
)
from sema.ingest.study_registry import StudyRegistry
from sema.log import logger
from sema.models.config import IngestConfig

QueryFn = Callable[[str, list[str]], Sequence[Sequence[object]]]


class StudyNotRegisteredError(LookupError):
    """The given study_id is not present in the _sema_study_registry."""


class StudyCacheMissingError(LookupError):
    """The study is registered but its cache directory does not exist."""


class PartialOverrideError(ValueError):
    """Partial override flags supplied; bypass mode requires the full set."""


@dataclass(frozen=True)
class ParsedTableComments:
    table_comment: str | None
    column_comments: dict[str, str]


@dataclass(frozen=True)
class LiveTableComments:
    table_comment: str | None
    column_comments: dict[str, str | None]


@dataclass(frozen=True)
class RecoveryContext:
    study_id: str | None
    source_cache: Path
    target_catalog: str
    target_schema: str


@dataclass(frozen=True)
class ColumnUpdate:
    table: str
    column: str
    new_comment: str


@dataclass(frozen=True)
class TableUpdate:
    table: str
    new_comment: str


@dataclass(frozen=True)
class SkippedColumn:
    table: str
    column: str
    reason: str


@dataclass(frozen=True)
class FailedColumn:
    table: str
    column: str
    error: str


@dataclass(frozen=True)
class RecoveryPlan:
    catalog: str
    schema: str
    column_updates: list[ColumnUpdate]
    table_updates: list[TableUpdate]
    skipped_columns: list[SkippedColumn]


@dataclass(frozen=True)
class RecoveryReport:
    columns_updated: int
    columns_skipped: int
    columns_failed: int
    table_comments_updated: int
    failed: list[FailedColumn] = field(default_factory=list)
    skipped: list[SkippedColumn] = field(default_factory=list)


def build_recovery_plan(
    ctx: RecoveryContext,
    parsed: dict[str, ParsedTableComments],
    live: dict[str, LiveTableComments],
    *,
    force: bool = False,
) -> RecoveryPlan:
    column_updates: list[ColumnUpdate] = []
    table_updates: list[TableUpdate] = []
    skipped: list[SkippedColumn] = []
    for table in sorted(parsed):
        parsed_t = parsed[table]
        live_t = live.get(table)
        if live_t is None:
            for col in parsed_t.column_comments:
                skipped.append(SkippedColumn(table, col, "table_not_found"))
            continue
        _plan_column_updates(
            table, parsed_t, live_t, force, column_updates, skipped,
        )
        _plan_table_update(table, parsed_t, live_t, force, table_updates)
    return RecoveryPlan(
        catalog=ctx.target_catalog,
        schema=ctx.target_schema,
        column_updates=column_updates,
        table_updates=table_updates,
        skipped_columns=skipped,
    )


def _plan_column_updates(
    table: str,
    parsed_t: ParsedTableComments,
    live_t: LiveTableComments,
    force: bool,
    out_updates: list[ColumnUpdate],
    out_skipped: list[SkippedColumn],
) -> None:
    for column, new_text in parsed_t.column_comments.items():
        if not new_text:
            continue
        if column not in live_t.column_comments:
            out_skipped.append(SkippedColumn(table, column, "column_not_found"))
            continue
        existing = live_t.column_comments.get(column) or ""
        if existing and not force:
            out_skipped.append(
                SkippedColumn(table, column, "already_commented")
            )
            continue
        out_updates.append(
            ColumnUpdate(table=table, column=column, new_comment=new_text)
        )


def _plan_table_update(
    table: str,
    parsed_t: ParsedTableComments,
    live_t: LiveTableComments,
    force: bool,
    out_updates: list[TableUpdate],
) -> None:
    new_text = parsed_t.table_comment
    if not new_text:
        return
    existing = live_t.table_comment or ""
    if existing and not force:
        return
    out_updates.append(TableUpdate(table=table, new_comment=new_text))


def execute_recovery_plan(
    plan: RecoveryPlan,
    executor_fn: Callable[[str], None],
    *,
    dry_run: bool = False,
) -> RecoveryReport:
    failed: list[FailedColumn] = []
    columns_updated = 0
    table_comments_updated = 0
    for upd in plan.column_updates:
        sql = build_alter_column_comment_sql(
            plan.catalog, plan.schema, upd.table, upd.column, upd.new_comment,
        )
        if dry_run:
            print(sql)
            columns_updated += 1
            continue
        try:
            executor_fn(sql)
            columns_updated += 1
        except Exception as err:  # noqa: BLE001
            logger.warning(
                "ALTER COLUMN failed for {}.{}: {}", upd.table, upd.column, err,
            )
            failed.append(
                FailedColumn(table=upd.table, column=upd.column, error=str(err))
            )
    for tupd in plan.table_updates:
        sql = build_alter_table_comment_sql(
            plan.catalog, plan.schema, tupd.table, tupd.new_comment,
        )
        if dry_run:
            print(sql)
            table_comments_updated += 1
            continue
        try:
            executor_fn(sql)
            table_comments_updated += 1
        except Exception as err:  # noqa: BLE001
            logger.warning("COMMENT ON TABLE failed for {}: {}", tupd.table, err)
            failed.append(
                FailedColumn(table=tupd.table, column="<table>", error=str(err))
            )
    return RecoveryReport(
        columns_updated=columns_updated,
        columns_skipped=len(plan.skipped_columns),
        columns_failed=len(failed),
        table_comments_updated=table_comments_updated,
        failed=failed,
        skipped=list(plan.skipped_columns),
    )


def read_databricks_comments(
    catalog: str, schema: str, query_fn: QueryFn,
) -> dict[str, LiveTableComments]:
    column_rows = query_fn(
        "SELECT table_name, column_name, comment "
        "FROM system.information_schema.columns "
        "WHERE table_catalog = ? AND table_schema = ? "
        "ORDER BY table_name, ordinal_position",
        [catalog, schema],
    )
    table_rows = query_fn(
        "SELECT table_name, comment "
        "FROM system.information_schema.tables "
        "WHERE table_catalog = ? AND table_schema = ?",
        [catalog, schema],
    )
    columns_by_table: dict[str, dict[str, str | None]] = {}
    for row in column_rows:
        table, column, comment = row[0], row[1], row[2]
        columns_by_table.setdefault(str(table), {})[str(column)] = (
            None if comment is None else str(comment)
        )
    table_comments: dict[str, str | None] = {
        str(row[0]): (None if row[1] is None else str(row[1]))
        for row in table_rows
    }
    out: dict[str, LiveTableComments] = {}
    table_names = set(columns_by_table) | set(table_comments)
    for name in table_names:
        out[name] = LiveTableComments(
            table_comment=table_comments.get(name),
            column_comments=columns_by_table.get(name, {}),
        )
    return out


def resolve_recovery_context(
    *,
    study_id: str | None,
    registry: StudyRegistry,
    ingest_config: IngestConfig,
    source_cache_override: Path | None,
    target_catalog_override: str | None,
    target_schema_override: str | None,
) -> RecoveryContext:
    if (
        source_cache_override is not None
        and target_catalog_override is not None
        and target_schema_override is not None
    ):
        return RecoveryContext(
            study_id=study_id,
            source_cache=source_cache_override,
            target_catalog=target_catalog_override,
            target_schema=target_schema_override,
        )
    if study_id is None:
        raise PartialOverrideError(
            "Recovery requires either --study or the full override set "
            "(--source-cache, --target-catalog, --target-schema)."
        )
    schema = (
        target_schema_override
        if target_schema_override is not None
        else registry.find_schema_for_study(study_id)
    )
    if schema is None:
        raise StudyNotRegisteredError(
            f"Study {study_id!r} not in `_sema_study_registry`. "
            f"Run `sema ingest cbioportal --study {study_id}` first, or pass "
            "`--source-cache`, `--target-catalog`, and `--target-schema` "
            "explicitly."
        )
    cache_path = (
        source_cache_override
        if source_cache_override is not None
        else Path(ingest_config.cache_dir).expanduser() / study_id
    )
    if not cache_path.exists():
        raise StudyCacheMissingError(
            f"Source cache for study {study_id!r} not found at {cache_path}. "
            "Re-fetch the study via `sema ingest cbioportal` or pass "
            "`--source-cache <path>`."
        )
    catalog = (
        target_catalog_override
        if target_catalog_override is not None
        else ingest_config.databricks.catalog
    )
    return RecoveryContext(
        study_id=study_id,
        source_cache=cache_path,
        target_catalog=catalog,
        target_schema=schema,
    )

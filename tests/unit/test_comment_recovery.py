from __future__ import annotations

from pathlib import Path

import pytest

from sema.ingest.comment_recovery import (
    ColumnUpdate,
    LiveTableComments,
    ParsedTableComments,
    PartialOverrideError,
    RecoveryContext,
    StudyCacheMissingError,
    StudyNotRegisteredError,
    TableUpdate,
    build_recovery_plan,
    execute_recovery_plan,
    read_databricks_comments,
    resolve_recovery_context,
)
from sema.ingest.duckdb_staging import Staging
from sema.ingest.study_registry import StudyRegistry
from sema.models.config import (
    IngestConfig,
    IngestDatabricksTargetConfig,
)

pytestmark = pytest.mark.unit


def _ctx() -> RecoveryContext:
    return RecoveryContext(
        study_id="study_x",
        source_cache=Path("/tmp/cache/study_x"),
        target_catalog="workspace",
        target_schema="cbioportal_x",
    )


def _parsed_clinical() -> dict[str, ParsedTableComments]:
    return {
        "patient": ParsedTableComments(
            table_comment="cBioPortal clinical patient",
            column_comments={
                "PATIENT_ID": "Identifier to uniquely specify a patient.",
                "OS_STATUS": "Overall survival status.",
            },
        ),
        "clinical_supp_hypoxia": ParsedTableComments(
            table_comment="cBioPortal clinical_supp_hypoxia",
            column_comments={"BUFFA_HYPOXIA_SCORE": "Buffa hypoxia score."},
        ),
    }


def _live_blank() -> dict[str, LiveTableComments]:
    return {
        "patient": LiveTableComments(
            table_comment=None,
            column_comments={"PATIENT_ID": "", "OS_STATUS": None},  # type: ignore[dict-item]
        ),
        "clinical_supp_hypoxia": LiveTableComments(
            table_comment="",
            column_comments={"BUFFA_HYPOXIA_SCORE": ""},
        ),
    }


def test_plan_emits_one_alter_per_parsed_column_and_table_comment() -> None:
    plan = build_recovery_plan(_ctx(), _parsed_clinical(), _live_blank())
    assert len(plan.column_updates) == 3
    assert len(plan.table_updates) == 2
    assert plan.catalog == "workspace"
    assert plan.schema == "cbioportal_x"
    pairs = {(u.table, u.column) for u in plan.column_updates}
    assert pairs == {
        ("patient", "PATIENT_ID"),
        ("patient", "OS_STATUS"),
        ("clinical_supp_hypoxia", "BUFFA_HYPOXIA_SCORE"),
    }


def test_plan_idempotent_when_live_already_commented() -> None:
    parsed = _parsed_clinical()
    live = {
        "patient": LiveTableComments(
            table_comment="cBioPortal clinical patient",
            column_comments={
                "PATIENT_ID": "Identifier to uniquely specify a patient.",
                "OS_STATUS": "Overall survival status.",
            },
        ),
        "clinical_supp_hypoxia": LiveTableComments(
            table_comment="cBioPortal clinical_supp_hypoxia",
            column_comments={"BUFFA_HYPOXIA_SCORE": "Buffa hypoxia score."},
        ),
    }
    plan = build_recovery_plan(_ctx(), parsed, live)
    assert plan.column_updates == []
    assert plan.table_updates == []


def test_plan_force_overrides_existing_column_and_table_comments() -> None:
    parsed = _parsed_clinical()
    live = {
        "patient": LiveTableComments(
            table_comment="manual operator edit",
            column_comments={
                "PATIENT_ID": "manual edit",
                "OS_STATUS": "manual edit",
            },
        ),
        "clinical_supp_hypoxia": LiveTableComments(
            table_comment="manual edit",
            column_comments={"BUFFA_HYPOXIA_SCORE": "manual edit"},
        ),
    }
    plan = build_recovery_plan(_ctx(), parsed, live, force=True)
    assert len(plan.column_updates) == 3
    assert len(plan.table_updates) == 2


def test_plan_marks_missing_column_as_skipped_with_reason() -> None:
    parsed = {
        "patient": ParsedTableComments(
            table_comment=None,
            column_comments={
                "PATIENT_ID": "ok",
                "LEGACY_COL": "no longer in databricks",
            },
        ),
    }
    live = {
        "patient": LiveTableComments(
            table_comment=None,
            column_comments={"PATIENT_ID": ""},
        ),
    }
    plan = build_recovery_plan(_ctx(), parsed, live)
    assert ColumnUpdate(
        table="patient", column="PATIENT_ID", new_comment="ok",
    ) in plan.column_updates
    assert any(
        s.column == "LEGACY_COL" and s.reason == "column_not_found"
        for s in plan.skipped_columns
    )


def test_plan_skips_table_unknown_to_live() -> None:
    parsed = {
        "ghost": ParsedTableComments(
            table_comment="t", column_comments={"X": "x"},
        ),
    }
    plan = build_recovery_plan(_ctx(), parsed, {})
    assert plan.column_updates == []
    assert plan.table_updates == []
    assert any(
        s.table == "ghost" and s.reason == "table_not_found"
        for s in plan.skipped_columns
    )


def test_plan_preserves_existing_table_comment_unless_force() -> None:
    parsed = {
        "patient": ParsedTableComments(
            table_comment="parser value",
            column_comments={},
        ),
    }
    live = {
        "patient": LiveTableComments(
            table_comment="existing",
            column_comments={},
        ),
    }
    plan = build_recovery_plan(_ctx(), parsed, live)
    assert plan.table_updates == []
    forced = build_recovery_plan(_ctx(), parsed, live, force=True)
    assert forced.table_updates == [
        TableUpdate(table="patient", new_comment="parser value"),
    ]


def test_plan_skips_empty_parser_comment() -> None:
    parsed = {
        "patient": ParsedTableComments(
            table_comment=None,
            column_comments={"BLANK": ""},
        ),
    }
    live = {
        "patient": LiveTableComments(
            table_comment=None, column_comments={"BLANK": ""},
        ),
    }
    plan = build_recovery_plan(_ctx(), parsed, live)
    assert plan.column_updates == []


def test_execute_runs_executor_per_statement() -> None:
    plan = build_recovery_plan(_ctx(), _parsed_clinical(), _live_blank())
    executed: list[str] = []

    def executor(sql: str) -> None:
        executed.append(sql)

    report = execute_recovery_plan(plan, executor)
    assert report.columns_updated == 3
    assert report.table_comments_updated == 2
    assert report.columns_failed == 0
    assert len(executed) == 5
    assert all("ALTER TABLE" in s or "COMMENT ON TABLE" in s for s in executed)


def test_execute_dry_run_skips_executor() -> None:
    plan = build_recovery_plan(_ctx(), _parsed_clinical(), _live_blank())
    executed: list[str] = []

    def executor(sql: str) -> None:
        executed.append(sql)

    report = execute_recovery_plan(plan, executor, dry_run=True)
    assert executed == []
    assert report.columns_updated == 3
    assert report.table_comments_updated == 2


def test_execute_continues_after_per_statement_failure() -> None:
    plan = build_recovery_plan(_ctx(), _parsed_clinical(), _live_blank())
    calls: list[str] = []

    def executor(sql: str) -> None:
        calls.append(sql)
        if "OS_STATUS" in sql:
            raise RuntimeError("simulated transient")

    report = execute_recovery_plan(plan, executor)
    assert report.columns_failed == 1
    assert report.columns_updated == 2
    assert len(calls) == 5
    assert any(f.column == "OS_STATUS" for f in report.failed)


def test_execute_records_skipped_from_plan() -> None:
    parsed = {
        "patient": ParsedTableComments(
            table_comment=None,
            column_comments={"PATIENT_ID": "ok", "LEGACY_COL": "x"},
        ),
    }
    live = {
        "patient": LiveTableComments(
            table_comment=None, column_comments={"PATIENT_ID": ""},
        ),
    }
    plan = build_recovery_plan(_ctx(), parsed, live)
    report = execute_recovery_plan(plan, lambda sql: None)
    assert report.columns_skipped == 1
    assert report.columns_updated == 1


def test_read_databricks_comments_combines_columns_and_tables() -> None:
    column_rows = [
        ("patient", "PATIENT_ID", "Existing comment"),
        ("patient", "AGE", None),
        ("sample", "SAMPLE_ID", ""),
    ]
    table_rows = [("patient", "old patient comment"), ("sample", None)]

    def query_fn(sql: str, params: list[str]) -> list[tuple[str, ...]]:
        if "information_schema.columns" in sql:
            return column_rows  # type: ignore[return-value]
        if "information_schema.tables" in sql:
            return table_rows  # type: ignore[return-value]
        raise AssertionError(f"unexpected query: {sql}")

    result = read_databricks_comments("workspace", "cbioportal_x", query_fn)
    assert result["patient"].table_comment == "old patient comment"
    assert result["patient"].column_comments == {
        "PATIENT_ID": "Existing comment",
        "AGE": None,
    }
    assert result["sample"].column_comments == {"SAMPLE_ID": ""}
    assert result["sample"].table_comment is None


def _ingest_config(tmp_path: Path) -> IngestConfig:
    return IngestConfig(
        cache_dir=str(tmp_path / "cache"),
        databricks=IngestDatabricksTargetConfig(catalog="workspace"),
    )


def _registry_with(study_to_schema: dict[str, str], staging: Staging) -> StudyRegistry:
    registry = StudyRegistry(staging)
    for study, schema in study_to_schema.items():
        registry.register(
            schema_name=schema,
            original_study_id=study,
            source_type="cbioportal",
        )
    return registry


@pytest.fixture
def staging_db(tmp_path: Path) -> Staging:
    return Staging(str(tmp_path / "stg.duckdb"))


def test_resolve_context_registered_study_with_cache(
    tmp_path: Path, staging_db: Staging,
) -> None:
    registry = _registry_with({"study_x": "cbioportal_x"}, staging_db)
    config = _ingest_config(tmp_path)
    cache = Path(config.cache_dir).expanduser() / "study_x"
    cache.mkdir(parents=True)

    ctx = resolve_recovery_context(
        study_id="study_x", registry=registry, ingest_config=config,
        source_cache_override=None, target_catalog_override=None,
        target_schema_override=None,
    )
    assert ctx.study_id == "study_x"
    assert ctx.source_cache == cache
    assert ctx.target_catalog == "workspace"
    assert ctx.target_schema == "cbioportal_x"


def test_resolve_context_unregistered_study_raises(
    tmp_path: Path, staging_db: Staging,
) -> None:
    registry = _registry_with({}, staging_db)
    config = _ingest_config(tmp_path)
    with pytest.raises(StudyNotRegisteredError):
        resolve_recovery_context(
            study_id="ghost", registry=registry, ingest_config=config,
            source_cache_override=None, target_catalog_override=None,
            target_schema_override=None,
        )


def test_resolve_context_missing_cache_raises_distinct_error(
    tmp_path: Path, staging_db: Staging,
) -> None:
    registry = _registry_with({"study_x": "cbioportal_x"}, staging_db)
    config = _ingest_config(tmp_path)
    with pytest.raises(StudyCacheMissingError):
        resolve_recovery_context(
            study_id="study_x", registry=registry, ingest_config=config,
            source_cache_override=None, target_catalog_override=None,
            target_schema_override=None,
        )


def test_resolve_context_full_overrides_bypass_registry(
    tmp_path: Path, staging_db: Staging,
) -> None:
    registry = _registry_with({}, staging_db)
    config = _ingest_config(tmp_path)
    cache = tmp_path / "alt"
    cache.mkdir()

    ctx = resolve_recovery_context(
        study_id=None, registry=registry, ingest_config=config,
        source_cache_override=cache,
        target_catalog_override="custom_catalog",
        target_schema_override="cbioportal_alt",
    )
    assert ctx.source_cache == cache
    assert ctx.target_catalog == "custom_catalog"
    assert ctx.target_schema == "cbioportal_alt"
    assert ctx.study_id is None


def test_resolve_context_partial_overrides_without_study_raise(
    tmp_path: Path, staging_db: Staging,
) -> None:
    registry = _registry_with({}, staging_db)
    config = _ingest_config(tmp_path)
    with pytest.raises(PartialOverrideError):
        resolve_recovery_context(
            study_id=None, registry=registry, ingest_config=config,
            source_cache_override=tmp_path,
            target_catalog_override=None,
            target_schema_override="cbioportal_alt",
        )


def test_resolve_context_no_study_no_overrides_raises(
    tmp_path: Path, staging_db: Staging,
) -> None:
    registry = _registry_with({}, staging_db)
    config = _ingest_config(tmp_path)
    with pytest.raises(PartialOverrideError):
        resolve_recovery_context(
            study_id=None, registry=registry, ingest_config=config,
            source_cache_override=None, target_catalog_override=None,
            target_schema_override=None,
        )

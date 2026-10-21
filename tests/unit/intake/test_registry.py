"""Run registry: one SQLite row per ingestion run, listed and filtered for the CLI."""

from __future__ import annotations

import ast
import importlib
import sqlite3
import uuid
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.unit

FORBIDDEN_IMPORT_ROOTS = ("langgraph", "langchain")
START = datetime(2026, 10, 20, 9, 0, tzinfo=UTC)


def _stepping_clock() -> Callable[[], datetime]:
    ticks: Iterator[int] = iter(range(10_000))
    return lambda: START + timedelta(seconds=next(ticks))


def _registry(root: Path) -> Any:
    from sema.intake.registry import Registry

    return Registry(root / "registry.sqlite", clock=_stepping_clock())


def _as_datetime(value: Any) -> datetime:
    return datetime.fromisoformat(value) if isinstance(value, str) else value


def _row(registry: Any, run_id: str) -> Any:
    matches = [r for r in registry.list() if r.run_id == run_id]
    assert len(matches) == 1
    return matches[0]


def _new_id() -> str:
    return str(uuid.uuid4())


def test_registry_lives_in_a_sqlite_file_in_the_ledger_root(tmp_path: Path) -> None:
    run_id = _new_id()
    _registry(tmp_path).register(run_id, source="dict", stage="land")

    with sqlite3.connect(tmp_path / "registry.sqlite") as conn:
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master")]

    assert tables


def test_register_starts_a_running_row(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    run_id = _new_id()
    registry.register(run_id, source="dict", stage="land")

    row = _row(registry, run_id)

    assert row.run_id == run_id
    assert row.source == "dict"
    assert row.stage == "land"
    assert row.status == "running"
    assert row.approver is None
    assert (row.model_calls, row.tokens) == (0, 0)
    assert row.cost == pytest.approx(0.0)


def test_advance_moves_stage_and_updates_timestamp(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    run_id = _new_id()
    registry.register(run_id, source="dict", stage="land")

    registry.advance(run_id, "roles", "running")
    row = _row(registry, run_id)

    assert (row.stage, row.status) == ("roles", "running")
    assert _as_datetime(row.updated_at) > _as_datetime(row.started_at)


def test_advance_rejects_an_unknown_status(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    run_id = _new_id()
    registry.register(run_id, source="dict", stage="land")

    with pytest.raises(ValueError, match="paused"):
        registry.advance(run_id, "roles", "paused")


@pytest.mark.parametrize(
    "status",
    ["running", "waiting_review", "approved", "rejected", "failed", "complete"],
)
def test_advance_accepts_every_spec_status(tmp_path: Path, status: str) -> None:
    registry = _registry(tmp_path)
    run_id = _new_id()
    registry.register(run_id, source="dict", stage="land")

    registry.advance(run_id, "review", status)

    assert _row(registry, run_id).status == status


def test_count_call_increases_calls_tokens_and_cost(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    run_id = _new_id()
    registry.register(run_id, source="dict", stage="land")

    registry.count_call(run_id, 120, 0.01)
    registry.count_call(run_id, 80, 0.02)
    row = _row(registry, run_id)

    assert (row.model_calls, row.tokens) == (2, 200)
    assert row.cost == pytest.approx(0.03)


def test_list_returns_newest_first(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    ids = [_new_id() for _ in range(3)]
    for run_id in ids:
        registry.register(run_id, source="dict", stage="land")

    assert [r.run_id for r in registry.list()] == list(reversed(ids))


def _three_runs(registry: Any) -> list[str]:
    ids = [_new_id() for _ in range(3)]
    registry.register(ids[0], source="dict", stage="land")
    registry.register(ids[1], source="codes", stage="land")
    registry.register(ids[2], source="codes", stage="land")
    registry.advance(ids[1], "structure", "running")
    registry.advance(ids[2], "review", "waiting_review")
    return ids


def test_filter_by_status(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    ids = _three_runs(registry)

    assert [r.run_id for r in registry.list(status="waiting_review")] == [ids[2]]


def test_filter_by_stage(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    ids = _three_runs(registry)

    assert [r.run_id for r in registry.list(stage="structure")] == [ids[1]]


def test_filter_by_source(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    ids = _three_runs(registry)

    assert [r.run_id for r in registry.list(source="codes")] == [ids[2], ids[1]]


def test_filters_combine(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    ids = _three_runs(registry)

    assert [r.run_id for r in registry.list(source="codes", status="running")] == [ids[1]]


def test_rows_persist_across_registry_instances(tmp_path: Path) -> None:
    run_id = _new_id()
    _registry(tmp_path).register(run_id, source="dict", stage="land")

    assert _row(_registry(tmp_path), run_id).source == "dict"


def test_uuid4_run_id_is_kept_unchanged(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    run_id = _new_id()
    registry.register(run_id, source="dict", stage="land")

    assert uuid.UUID(_row(registry, run_id).run_id).version == 4


def test_duplicate_run_id_is_rejected_naming_the_id(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    run_id = _new_id()
    registry.register(run_id, source="dict", stage="land")

    with pytest.raises(ValueError, match=run_id):
        registry.register(run_id, source="dict", stage="land")
    assert len(registry.list()) == 1


def test_registry_imports_no_orchestration_framework() -> None:
    module = importlib.import_module("sema.intake.registry")
    assert module.__file__ is not None
    tree = ast.parse(Path(module.__file__).read_text())
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])

    assert not roots & set(FORBIDDEN_IMPORT_ROOTS)
    assert "sqlite3" in roots

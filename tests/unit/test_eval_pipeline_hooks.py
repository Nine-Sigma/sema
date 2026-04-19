"""Tests for eval hooks wired into the pipeline."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from sema.models.config import BuildConfig

pytestmark = pytest.mark.unit


class TestBuildConfigEvalFields:
    def test_defaults_are_inert(self) -> None:
        cfg = BuildConfig()
        assert cfg.eval_dump_dir is None
        assert cfg.eval_config_label == "run"
        assert cfg.slice_tables == []


class TestSliceFilter:
    def test_keeps_only_slice_tables(self) -> None:
        from sema.pipeline.orchestrate_utils import (
            _filter_work_items_to_slice,
        )

        work_items = [
            MagicMock(table_name="patient"),
            MagicMock(table_name="sample"),
            MagicMock(table_name="mutation"),
        ]
        kept = _filter_work_items_to_slice(
            work_items, slice_tables=["patient", "mutation"],
        )
        names = [w.table_name for w in kept]
        assert names == ["patient", "mutation"]

    def test_empty_slice_returns_all(self) -> None:
        from sema.pipeline.orchestrate_utils import (
            _filter_work_items_to_slice,
        )

        work_items = [
            MagicMock(table_name="patient"),
            MagicMock(table_name="sample"),
        ]
        kept = _filter_work_items_to_slice(
            work_items, slice_tables=[],
        )
        assert len(kept) == 2


class TestDumpHook:
    def test_writes_both_assertion_and_telemetry_files(
        self, tmp_path: Path,
    ) -> None:
        from sema.models.assertions import (
            Assertion,
            AssertionPredicate,
        )
        from datetime import datetime, timezone
        from sema.eval.pipeline_hook import dump_table_eval_outputs

        a = Assertion(
            id="x",
            subject_ref="unity://c/s/patient",
            predicate=AssertionPredicate.HAS_ENTITY_NAME,
            payload={"value": "Patient"},
            source="llm_interpretation",
            confidence=0.9,
            run_id="run-1",
            observed_at=datetime.now(timezone.utc),
        )
        telemetry = {"b_outcome": "B_SUCCESS", "raw_coverage_pct": 1.0}

        dump_table_eval_outputs(
            assertions=[a], telemetry=telemetry,
            table_ref="unity://c/s/patient",
            label="staged", output_dir=tmp_path,
        )

        assert (tmp_path / "patient__staged.json").exists()
        assert (
            tmp_path / "patient__staged__telemetry.json"
        ).exists()

    def test_no_telemetry_file_when_none(
        self, tmp_path: Path,
    ) -> None:
        from sema.eval.pipeline_hook import dump_table_eval_outputs

        dump_table_eval_outputs(
            assertions=[], telemetry=None,
            table_ref="unity://c/s/patient",
            label="staged", output_dir=tmp_path,
        )
        assert (tmp_path / "patient__staged.json").exists()
        assert not (
            tmp_path / "patient__staged__telemetry.json"
        ).exists()

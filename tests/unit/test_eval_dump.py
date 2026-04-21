"""Tests for assertion dump capture (task 5.4)."""
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from sema.models.assertions import (
    Assertion,
    AssertionPredicate,
    AssertionStatus,
)

pytestmark = pytest.mark.unit


def _make_assertion(
    subject_ref: str,
    predicate: AssertionPredicate,
    payload: dict | None = None,
    confidence: float = 0.9,
    source: str = "llm_interpretation",
    run_id: str = "run-001",
) -> Assertion:
    return Assertion(
        id=f"a-{subject_ref}-{predicate.value}",
        subject_ref=subject_ref,
        predicate=predicate,
        payload=payload or {},
        source=source,
        confidence=confidence,
        status=AssertionStatus.AUTO,
        run_id=run_id,
        observed_at=datetime(2026, 4, 13, 12, 0, 0, tzinfo=timezone.utc),
    )


def _sample_assertions() -> list[Assertion]:
    return [
        _make_assertion(
            "unity://cat.sch.patient",
            AssertionPredicate.HAS_ENTITY_NAME,
            {"value": "Patient"},
        ),
        _make_assertion(
            "unity://cat.sch.patient.gender",
            AssertionPredicate.HAS_PROPERTY_NAME,
            {"value": "gender"},
        ),
        _make_assertion(
            "unity://cat.sch.patient.gender",
            AssertionPredicate.HAS_SEMANTIC_TYPE,
            {"value": "demographic"},
        ),
        _make_assertion(
            "unity://cat.sch.patient.os_status",
            AssertionPredicate.HAS_DECODED_VALUE,
            {"value": "0:LIVING", "decoded": "Patient alive"},
        ),
    ]


class TestAssertionDumpCapture:
    """Dump assertions to JSON keyed by table, timestamp, config label."""

    def test_dump_creates_json_file(self, tmp_path: Path) -> None:
        from sema.eval.dump import dump_assertions

        assertions = _sample_assertions()
        out = dump_assertions(
            assertions=assertions,
            table_ref="unity://cat.sch.patient",
            config_label="baseline",
            output_dir=tmp_path,
        )
        assert out.exists()
        assert out.suffix == ".json"

    def test_dump_json_structure(self, tmp_path: Path) -> None:
        from sema.eval.dump import dump_assertions

        assertions = _sample_assertions()
        out = dump_assertions(
            assertions=assertions,
            table_ref="unity://cat.sch.patient",
            config_label="baseline",
            output_dir=tmp_path,
        )
        data = json.loads(out.read_text())
        assert data["table_ref"] == "unity://cat.sch.patient"
        assert data["config_label"] == "baseline"
        assert "timestamp" in data
        assert isinstance(data["assertions"], list)
        assert len(data["assertions"]) == 4

    def test_dump_assertion_fields(self, tmp_path: Path) -> None:
        from sema.eval.dump import dump_assertions

        assertions = _sample_assertions()
        out = dump_assertions(
            assertions=assertions,
            table_ref="unity://cat.sch.patient",
            config_label="staged",
            output_dir=tmp_path,
        )
        data = json.loads(out.read_text())
        first = data["assertions"][0]
        assert "subject_ref" in first
        assert "predicate" in first
        assert "payload" in first
        assert "confidence" in first
        assert "source" in first

    def test_dump_filename_contains_table_and_label(
        self, tmp_path: Path,
    ) -> None:
        from sema.eval.dump import dump_assertions

        out = dump_assertions(
            assertions=_sample_assertions(),
            table_ref="unity://cat.sch.patient",
            config_label="baseline",
            output_dir=tmp_path,
        )
        assert "patient" in out.name
        assert "baseline" in out.name

    def test_dump_preserves_assertion_ordering(
        self, tmp_path: Path,
    ) -> None:
        from sema.eval.dump import dump_assertions

        assertions = _sample_assertions()
        out = dump_assertions(
            assertions=assertions,
            table_ref="unity://cat.sch.patient",
            config_label="test",
            output_dir=tmp_path,
        )
        data = json.loads(out.read_text())
        predicates = [a["predicate"] for a in data["assertions"]]
        assert predicates == [
            "has_entity_name",
            "has_property_name",
            "has_semantic_type",
            "has_decoded_value",
        ]

    def test_dump_with_run_id(self, tmp_path: Path) -> None:
        from sema.eval.dump import dump_assertions

        out = dump_assertions(
            assertions=_sample_assertions(),
            table_ref="unity://cat.sch.patient",
            config_label="v1",
            output_dir=tmp_path,
            run_id="run-abc",
        )
        data = json.loads(out.read_text())
        assert data["run_id"] == "run-abc"

    def test_dump_empty_assertions(self, tmp_path: Path) -> None:
        from sema.eval.dump import dump_assertions

        out = dump_assertions(
            assertions=[],
            table_ref="unity://cat.sch.empty_table",
            config_label="test",
            output_dir=tmp_path,
        )
        data = json.loads(out.read_text())
        assert data["assertions"] == []
        assert data["table_ref"] == "unity://cat.sch.empty_table"


class TestLoadAssertionDump:
    """Load a previously-saved assertion dump."""

    def test_load_roundtrips(self, tmp_path: Path) -> None:
        from sema.eval.dump import dump_assertions, load_dump

        assertions = _sample_assertions()
        out = dump_assertions(
            assertions=assertions,
            table_ref="unity://cat.sch.patient",
            config_label="baseline",
            output_dir=tmp_path,
        )
        loaded = load_dump(out)
        assert loaded["table_ref"] == "unity://cat.sch.patient"
        assert len(loaded["assertions"]) == 4
        assert loaded["config_label"] == "baseline"

    def test_load_nonexistent_raises(self) -> None:
        from sema.eval.dump import load_dump

        with pytest.raises(FileNotFoundError):
            load_dump(Path("/does/not/exist.json"))

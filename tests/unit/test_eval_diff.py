"""Tests for structured assertion diff tool (task 5.5)."""
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def _make_dump(
    table_ref: str,
    assertions: list[dict],
    config_label: str = "test",
) -> dict:
    return {
        "table_ref": table_ref,
        "config_label": config_label,
        "timestamp": "2026-04-13T12:00:00Z",
        "run_id": "run-001",
        "assertions": assertions,
    }


def _assertion_dict(
    subject_ref: str,
    predicate: str,
    payload: dict | None = None,
    confidence: float = 0.9,
    source: str = "llm_interpretation",
) -> dict:
    return {
        "subject_ref": subject_ref,
        "predicate": predicate,
        "payload": payload or {},
        "confidence": confidence,
        "source": source,
    }


class TestAssertionDiff:
    """Compare two assertion dumps and report changes."""

    def test_identical_dumps_no_changes(self) -> None:
        from sema.eval.diff import diff_dumps

        a = _assertion_dict("t.col1", "has_property_name", {"value": "age"})
        dump_a = _make_dump("t", [a])
        dump_b = _make_dump("t", [a])
        result = diff_dumps(dump_a, dump_b)
        assert result["added"] == []
        assert result["removed"] == []
        assert result["changed"] == []

    def test_added_assertion(self) -> None:
        from sema.eval.diff import diff_dumps

        a1 = _assertion_dict("t.col1", "has_property_name", {"value": "age"})
        a2 = _assertion_dict(
            "t.col2", "has_semantic_type", {"value": "numeric"},
        )
        dump_a = _make_dump("t", [a1])
        dump_b = _make_dump("t", [a1, a2])
        result = diff_dumps(dump_a, dump_b)
        assert len(result["added"]) == 1
        assert result["added"][0]["subject_ref"] == "t.col2"
        assert result["removed"] == []

    def test_removed_assertion(self) -> None:
        from sema.eval.diff import diff_dumps

        a1 = _assertion_dict("t.col1", "has_property_name", {"value": "age"})
        a2 = _assertion_dict(
            "t.col2", "has_semantic_type", {"value": "numeric"},
        )
        dump_a = _make_dump("t", [a1, a2])
        dump_b = _make_dump("t", [a1])
        result = diff_dumps(dump_a, dump_b)
        assert result["added"] == []
        assert len(result["removed"]) == 1
        assert result["removed"][0]["subject_ref"] == "t.col2"

    def test_changed_payload(self) -> None:
        from sema.eval.diff import diff_dumps

        a_old = _assertion_dict(
            "t.col1", "has_property_name", {"value": "age"},
        )
        a_new = _assertion_dict(
            "t.col1", "has_property_name", {"value": "patient_age"},
        )
        dump_a = _make_dump("t", [a_old])
        dump_b = _make_dump("t", [a_new])
        result = diff_dumps(dump_a, dump_b)
        assert result["added"] == []
        assert result["removed"] == []
        assert len(result["changed"]) == 1
        change = result["changed"][0]
        assert change["subject_ref"] == "t.col1"
        assert change["predicate"] == "has_property_name"
        assert change["old_payload"] == {"value": "age"}
        assert change["new_payload"] == {"value": "patient_age"}

    def test_changed_confidence(self) -> None:
        from sema.eval.diff import diff_dumps

        a_old = _assertion_dict(
            "t.col1", "has_entity_name", {"value": "Patient"},
            confidence=0.9,
        )
        a_new = _assertion_dict(
            "t.col1", "has_entity_name", {"value": "Patient"},
            confidence=0.75,
        )
        dump_a = _make_dump("t", [a_old])
        dump_b = _make_dump("t", [a_new])
        result = diff_dumps(dump_a, dump_b)
        assert len(result["changed"]) == 1
        change = result["changed"][0]
        assert change["old_confidence"] == 0.9
        assert change["new_confidence"] == 0.75

    def test_multiple_changes(self) -> None:
        from sema.eval.diff import diff_dumps

        old_assertions = [
            _assertion_dict("t.col1", "has_property_name", {"value": "age"}),
            _assertion_dict("t.col2", "has_semantic_type", {"value": "id"}),
            _assertion_dict("t.col3", "has_alias", {"value": "gender"}),
        ]
        new_assertions = [
            _assertion_dict(
                "t.col1", "has_property_name", {"value": "patient_age"},
            ),
            _assertion_dict("t.col4", "has_decoded_value", {"value": "M"}),
        ]
        dump_a = _make_dump("t", old_assertions)
        dump_b = _make_dump("t", new_assertions)
        result = diff_dumps(dump_a, dump_b)
        assert len(result["added"]) == 1
        assert len(result["removed"]) == 2
        assert len(result["changed"]) == 1

    def test_summary_stats(self) -> None:
        from sema.eval.diff import diff_dumps

        a1 = _assertion_dict("t.col1", "has_property_name", {"value": "age"})
        a2 = _assertion_dict("t.col2", "has_semantic_type", {"value": "id"})
        dump_a = _make_dump("t", [a1])
        dump_b = _make_dump("t", [a1, a2])
        result = diff_dumps(dump_a, dump_b)
        summary = result["summary"]
        assert summary["added_count"] == 1
        assert summary["removed_count"] == 0
        assert summary["changed_count"] == 0
        assert summary["total_before"] == 1
        assert summary["total_after"] == 2

    def test_empty_dumps(self) -> None:
        from sema.eval.diff import diff_dumps

        dump_a = _make_dump("t", [])
        dump_b = _make_dump("t", [])
        result = diff_dumps(dump_a, dump_b)
        assert result["added"] == []
        assert result["removed"] == []
        assert result["changed"] == []

    def test_diff_from_files(self, tmp_path: Path) -> None:
        from sema.eval.diff import diff_dump_files

        a1 = _assertion_dict("t.col1", "has_property_name", {"value": "age"})
        a2 = _assertion_dict("t.col2", "has_semantic_type", {"value": "id"})
        file_a = tmp_path / "before.json"
        file_b = tmp_path / "after.json"
        file_a.write_text(json.dumps(_make_dump("t", [a1])))
        file_b.write_text(json.dumps(_make_dump("t", [a1, a2])))
        result = diff_dump_files(file_a, file_b)
        assert len(result["added"]) == 1

    def test_regression_flag(self) -> None:
        """Removing a previously-correct assertion is flagged."""
        from sema.eval.diff import diff_dumps

        entity = _assertion_dict(
            "t", "has_entity_name", {"value": "Patient"},
        )
        dump_a = _make_dump("t", [entity])
        dump_b = _make_dump("t", [])
        result = diff_dumps(dump_a, dump_b)
        assert len(result["removed"]) == 1
        assert any(
            r.get("regression_risk", False)
            for r in result["removed"]
        )

    def test_changed_source_not_flagged(self) -> None:
        """Source field change alone is not a semantic change."""
        from sema.eval.diff import diff_dumps

        a_old = _assertion_dict(
            "t.col1", "has_property_name", {"value": "age"},
            source="old_engine",
        )
        a_new = _assertion_dict(
            "t.col1", "has_property_name", {"value": "age"},
            source="staged_engine",
        )
        dump_a = _make_dump("t", [a_old])
        dump_b = _make_dump("t", [a_new])
        result = diff_dumps(dump_a, dump_b)
        # Source change alone should not count as a semantic change
        assert result["changed"] == []

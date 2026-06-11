"""US-004: vocabulary lifecycle runs once per build, not per table.

A later table's materialization must never deprecate vocabularies an
earlier table introduced (finding D). Deprecation now runs once at the
orchestrator over the union of every table's active vocabularies.
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from sema.graph.lifecycle_utils import (
    active_vocab_names,
    deprecate_stale_from_results,
    deprecate_stale_vocabularies,
)
from sema.models.assertions import (
    Assertion,
    AssertionPredicate,
    AssertionStatus,
)

pytestmark = pytest.mark.unit


def _vocab_match(
    value: str, status: AssertionStatus = AssertionStatus.AUTO,
) -> Assertion:
    return Assertion(
        id=f"a-{value}",
        subject_ref="databricks://ws/sch/tbl#col",
        predicate=AssertionPredicate.VOCABULARY_MATCH,
        payload={"value": value},
        source="test",
        confidence=0.9,
        status=status,
        run_id="run",
        observed_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


class TestActiveVocabNames:
    def test_collects_active_only(self) -> None:
        names = active_vocab_names([
            _vocab_match("ICD-10"),
            _vocab_match("SNOMED", AssertionStatus.REJECTED),
            _vocab_match("LOINC", AssertionStatus.SUPERSEDED),
        ])
        assert names == {"ICD-10"}

    def test_ignores_non_vocab_predicates(self) -> None:
        other = Assertion(
            id="x",
            subject_ref="databricks://ws/sch/tbl#col",
            predicate=AssertionPredicate.HAS_ENTITY_NAME,
            payload={"value": "Patient"},
            source="test",
            confidence=0.9,
            status=AssertionStatus.AUTO,
            run_id="run",
            observed_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        assert active_vocab_names([other]) == set()


class TestDeprecateStaleVocabularies:
    def test_no_op_when_empty(self) -> None:
        loader = MagicMock()
        deprecate_stale_vocabularies(loader, set())
        loader._run.assert_not_called()

    def test_runs_with_active_names(self) -> None:
        loader = MagicMock()
        deprecate_stale_vocabularies(loader, {"ICD-10"})
        loader._run.assert_called_once()
        call = loader._run.call_args
        query = call.args[0]
        assert "DEPRECATED" in query
        assert "ICD-10" in call.kwargs["active_names"]


class TestMaterializeUnifiedNoPerTableLifecycle:
    def test_materialize_does_not_deprecate(self) -> None:
        from sema.graph.materializer import materialize_unified

        loader = MagicMock()
        materialize_unified(
            loader, [_vocab_match("ICD-10")], source_schema="sch",
        )
        queries = [
            c.args[0] for c in loader._run.call_args_list if c.args
        ]
        assert not any("DEPRECATED" in q for q in queries)


class TestOrchestratorRunsLifecycleOnce:
    def test_union_keeps_every_tables_vocab(self) -> None:
        from sema.pipeline.build import TableResult

        results = [
            TableResult.success("A", active_vocabularies=["X"]),
            TableResult.success("B", active_vocabularies=["Y"]),
        ]
        loader = MagicMock()
        deprecate_stale_from_results(loader, results)
        loader._run.assert_called_once()
        active = loader._run.call_args.kwargs["active_names"]
        assert set(active) == {"X", "Y"}

    def test_no_active_vocabs_is_no_op(self) -> None:
        from sema.pipeline.build import TableResult

        loader = MagicMock()
        deprecate_stale_from_results(loader, [TableResult.success("A")])
        loader._run.assert_not_called()


class TestResumeCarriesActiveVocabularies:
    def test_resumed_table_contributes_vocab(self) -> None:
        from sema.pipeline.build import _try_resume

        loader = MagicMock()
        loader.has_assertions.return_value = True
        loader.load_assertions.return_value = [{
            "id": "a-1",
            "subject_ref": "databricks://ws/sch/tbl#col",
            "predicate": AssertionPredicate.VOCABULARY_MATCH.value,
            "payload": {"value": "ICD-10"},
            "source": "test",
            "confidence": 0.9,
            "status": AssertionStatus.AUTO.value,
            "run_id": "run",
            "observed_at": "2026-01-01T00:00:00+00:00",
        }]
        work_item = MagicMock()
        work_item.fqn = "ws.sch.tbl"
        work_item.schema = "sch"
        work_item.table_name = "tbl"

        with patch(
            "sema.graph.materializer.materialize_unified"
        ):
            result = _try_resume(work_item, loader)

        assert result is not None
        assert result.status == "skipped"
        assert result.active_vocabularies == ["ICD-10"]

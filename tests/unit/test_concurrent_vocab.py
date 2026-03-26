from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from unittest.mock import MagicMock, call, patch

import pytest

from sema.models.assertions import (
    Assertion,
    AssertionPredicate,
)

pytestmark = pytest.mark.unit


def _make_extraction_assertion(
    column_ref: str,
    values: list[dict[str, str]],
    run_id: str = "run-1",
) -> Assertion:
    return Assertion(
        id=f"ext-{column_ref}",
        subject_ref=column_ref,
        predicate=AssertionPredicate.HAS_TOP_VALUES,
        payload={"values": values},
        source="test",
        confidence=0.9,
        run_id=run_id,
        observed_at=datetime.now(timezone.utc),
    )


def _make_semantic_assertion(
    column_ref: str,
    raw: str,
    label: str,
    run_id: str = "run-1",
) -> Assertion:
    return Assertion(
        id=f"sem-{column_ref}-{raw}",
        subject_ref=column_ref,
        predicate=AssertionPredicate.HAS_DECODED_VALUE,
        payload={"raw": raw, "label": label},
        source="test",
        confidence=0.8,
        run_id=run_id,
        observed_at=datetime.now(timezone.utc),
    )


def _make_vocab_result(column_ref: str) -> list[Assertion]:
    return [
        Assertion(
            id=f"vocab-{column_ref}",
            subject_ref=column_ref,
            predicate=AssertionPredicate.VOCABULARY_MATCH,
            payload={"value": "ICD-10"},
            source="pattern_match",
            confidence=0.9,
            run_id="run-1",
            observed_at=datetime.now(timezone.utc),
        )
    ]


def test_vocabulary_columns_processed_concurrently() -> None:
    col_refs = [
        "unity://cat.sch.tbl.col_a",
        "unity://cat.sch.tbl.col_b",
        "unity://cat.sch.tbl.col_c",
    ]
    extraction_assertions = [
        _make_extraction_assertion(ref, [{"value": "A01"}])
        for ref in col_refs
    ]
    semantic_assertions: list[Assertion] = []

    mock_vocab = MagicMock()
    mock_vocab.process_column = MagicMock(
        side_effect=lambda ref, vals, dec: _make_vocab_result(ref)
    )

    with patch(
        "sema.pipeline.build_utils.ThreadPoolExecutor",
    ) as mock_executor_cls:
        mock_executor = MagicMock(spec=ThreadPoolExecutor)
        mock_executor_cls.return_value.__enter__ = MagicMock(
            return_value=mock_executor
        )
        mock_executor_cls.return_value.__exit__ = MagicMock(
            return_value=False
        )

        futures = []
        for ref in col_refs:
            future: Future[list[Assertion]] = Future()
            future.set_result(_make_vocab_result(ref))
            futures.append(future)

        mock_executor.submit = MagicMock(side_effect=futures)

        with patch(
            "sema.pipeline.build_utils.as_completed",
            return_value=iter(futures),
        ):
            from sema.pipeline.build_utils import (
                _run_vocabulary_alignment,
            )

            result = _run_vocabulary_alignment(
                extraction_assertions,
                semantic_assertions,
                MagicMock(table_name="tbl"),
                MagicMock(),
                "run-1",
                vocab_workers=4,
            )

        mock_executor_cls.assert_called_once_with(max_workers=4)
        assert mock_executor.submit.call_count == 3
        assert len(result) == 3


def test_concurrent_vocab_results_match_sequential() -> None:
    col_refs = [
        "unity://cat.sch.tbl.col_a",
        "unity://cat.sch.tbl.col_b",
    ]
    extraction_assertions = [
        _make_extraction_assertion(ref, [{"value": "X99"}])
        for ref in col_refs
    ]
    semantic_assertions: list[Assertion] = []

    vocab_results: dict[str, list[Assertion]] = {
        ref: _make_vocab_result(ref) for ref in col_refs
    }

    mock_vocab_engine = MagicMock()
    mock_vocab_engine.process_column = MagicMock(
        side_effect=lambda ref, vals, dec: vocab_results[ref]
    )

    sequential: list[Assertion] = []
    for ref in col_refs:
        sequential.extend(vocab_results[ref])

    with patch(
        "sema.pipeline.build_utils.VocabularyEngine",
        return_value=mock_vocab_engine,
    ):
        from sema.pipeline.build_utils import (
            _run_vocabulary_alignment,
        )

        concurrent_result = _run_vocabulary_alignment(
            extraction_assertions,
            semantic_assertions,
            MagicMock(table_name="tbl"),
            MagicMock(),
            "run-1",
            vocab_workers=2,
        )

    sequential_subjects = {a.subject_ref for a in sequential}
    concurrent_subjects = {a.subject_ref for a in concurrent_result}
    assert sequential_subjects == concurrent_subjects


def test_single_column_failure_doesnt_block_others() -> None:
    col_refs = [
        "unity://cat.sch.tbl.col_ok1",
        "unity://cat.sch.tbl.col_fail",
        "unity://cat.sch.tbl.col_ok2",
    ]
    extraction_assertions = [
        _make_extraction_assertion(ref, [{"value": "Z01"}])
        for ref in col_refs
    ]
    semantic_assertions: list[Assertion] = []

    def side_effect(ref: str, vals: list[str], dec: list[dict[str, str]] | None) -> list[Assertion]:
        if "col_fail" in ref:
            raise RuntimeError("LLM timeout")
        return _make_vocab_result(ref)

    mock_vocab_engine = MagicMock()
    mock_vocab_engine.process_column = MagicMock(side_effect=side_effect)

    with patch(
        "sema.pipeline.build_utils.VocabularyEngine",
        return_value=mock_vocab_engine,
    ):
        from sema.pipeline.build_utils import (
            _run_vocabulary_alignment,
        )

        result = _run_vocabulary_alignment(
            extraction_assertions,
            semantic_assertions,
            MagicMock(table_name="tbl"),
            MagicMock(),
            "run-1",
            vocab_workers=3,
        )

    result_subjects = {a.subject_ref for a in result}
    assert "unity://cat.sch.tbl.col_ok1" in result_subjects
    assert "unity://cat.sch.tbl.col_ok2" in result_subjects
    assert "unity://cat.sch.tbl.col_fail" not in result_subjects
    assert len(result) == 2

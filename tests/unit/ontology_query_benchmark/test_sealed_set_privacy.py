"""Ruling M on the sealed-set runner: no question text, SQL or row on an unsealed surface
(M1 to M3), arm S on the core query path (M4), and found values normalized by the
expected-answer writer's rule before scoring (M5). The entry under test and the neutral set
are described in `test_sealed_set_arms.py`.
"""

import datetime
import json
import re
import tempfile
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from sema.log import logger
from tests.unit.ontology_query_benchmark.sealed_set_env import (
    ARM_REPLIES,
    ARMS,
    LEAK_MARKED,
    LEAK_REPLY,
    LEAK_ROW,
    LEAK_SQL,
    LEAK_TEXT,
    MARKERS,
    SINK_CANARY,
    WIDE,
    WRITER_RULE,
    Reply,
    SealedRun,
    captured_logs,
    files_holding,
    mark_questions,
    materialize,
    question_texts,
    sealed_run,  # noqa: F401 - a fixture
    truncate_expected,
)
from tests.unit.ontology_query_benchmark.sealed_set_env import run_arms as _run
from tests.unit.ontology_query_benchmark.sealed_set_env import sealed_set_module as _sealed_set
from tests.unit.ontology_query_benchmark.sealed_set_env import writer_cell as _cell

pytestmark = pytest.mark.unit


def test_a_failing_lexical_query_logs_no_question_text() -> None:
    from sema.pipeline.retrieval import RetrievalEngine

    def failing_run(query: str, **params: Any) -> Any:
        raise RuntimeError(f"lexical query failed for {params}")

    driver = MagicMock()
    driver.session.return_value.__enter__.return_value.run.side_effect = failing_run
    with captured_logs() as logs:
        logger.trace(SINK_CANARY)
        RetrievalEngine(driver, embedder=None).retrieve(f"How many {LEAK_TEXT} rows?")

    assert SINK_CANARY in "".join(logs) and driver.session.called
    assert [line for line in logs if LEAK_TEXT.lower() in line.lower()] == []


def test_markers_in_a_wrong_a_flagged_and_a_refused_question_reach_no_unsealed_surface(
    frozen: Path, harness: SimpleNamespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "sealed_set"
    profile_path = materialize(root, frozen)
    mark_questions(root / "neutral", LEAK_MARKED, LEAK_TEXT)
    truncate_expected(root / "neutral", WIDE, keep=5)
    monkeypatch.setitem(ARM_REPLIES["N02"], "S", LEAK_REPLY)

    run = _run(harness, root, profile_path)

    assert run.receipt("S:N02:1")["outcome"] == "wrong"
    assert LEAK_ROW in json.dumps(run.receipt("S:N02:1")["answer"])
    assert LEAK_TEXT in json.dumps(run.model.calls) and SINK_CANARY in run.logs
    markers = (LEAK_TEXT, LEAK_SQL, LEAK_ROW)
    surfaces = {"stdout and stderr": run.output, "log": run.logs, "report": run.report}
    assert [(name, m) for name, text in surfaces.items() for m in markers if m in text] == []
    assert files_holding((root,), run.set_dir / "sealed", markers) == []


def test_a_dry_run_prints_only_slot_and_spend_lines_ids_counts_and_status_words(
    frozen: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    root = tmp_path / "sealed_set"
    profile_path = materialize(root, frozen)
    truncate_expected(root / "neutral", WIDE, keep=5)
    texts = question_texts(root / "neutral")

    with captured_logs() as logs:
        code = _sealed_set().main(["--profile", str(profile_path), "--arms", "A,S",
                                   "--repetitions", "1", "--dry-run"])  # fmt: skip

    out, err = capsys.readouterr()
    lines = out.splitlines()
    assert code == 0 and {"spend 0.00", f"slots {(len(texts) - 1) * 2}"} <= set(lines)
    word = re.compile(r"[a-z_]+|\d+(\.\d\d)?")
    assert all(t in texts or word.fullmatch(t) for line in lines for t in line.split()), lines
    leaked = [t for t in (*texts.values(), *MARKERS) for s in (out, err, *logs) if t in s]
    assert leaked == []


def test_a_run_writes_no_question_sql_or_row_outside_the_sealed_folder(
    frozen: Path, harness: SimpleNamespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, work, temp = tmp_path / "sealed_set", tmp_path / "cwd", tmp_path / "tmp"
    profile_path = materialize(root, frozen)
    work.mkdir()
    temp.mkdir()
    monkeypatch.chdir(work)
    monkeypatch.setenv("TMPDIR", str(temp))
    monkeypatch.setattr(tempfile, "tempdir", str(temp))

    run = _run(harness, root, profile_path)

    texts = tuple(question_texts(run.set_dir).values())
    needles = (*texts, *run.model.replies, *MARKERS)
    assert run.receipts and run.model.replies
    assert files_holding((root, work, temp), run.set_dir / "sealed", needles) == []


def test_model_sql_cannot_read_a_sealed_file(
    frozen: Path, harness: SimpleNamespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "sealed_set"
    profile_path = materialize(root, frozen)
    questions = root / "neutral" / "sealed" / "questions.jsonl"
    reply = Reply(f"SELECT content FROM read_text('{questions}')", False, False, 0)
    monkeypatch.setitem(ARM_REPLIES["N02"], "A", reply)

    run = _run(harness, root, profile_path)

    answer = json.dumps(run.receipt("A:N02:1").get("answer"))
    assert [t for t in question_texts(run.set_dir).values() if t in answer] == []


def test_main_gives_arm_s_the_core_path_provider(
    frozen: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from showcase.ontology_query_benchmark.core_s_provider import CoreSContextProvider
    from showcase.ontology_query_benchmark.sema_context import SContextProvider

    profile_path = materialize(tmp_path / "sealed_set", frozen)
    module, seen = _sealed_set(), {}
    monkeypatch.setenv("EMBEDDING_PROVIDER", "")
    monkeypatch.setattr(module, "require_clean_process", lambda *_, **__: None)
    monkeypatch.setattr(module, "model_client", lambda *_, **__: "mocked-model")
    monkeypatch.setattr(module, "run_sealed_set", lambda *_, **kwargs: seen.update(kwargs))

    code = module.main(["--profile", str(profile_path), "--arms", "A,S", "--repetitions", "1",
                        "--report", str(tmp_path / "report.md")])  # fmt: skip

    provider = seen["s_provider"]
    assert code == 0 and seen["model"] == "mocked-model"
    assert isinstance(provider, CoreSContextProvider)
    assert not isinstance(provider, SContextProvider)


def test_found_values_are_normalized_by_the_writers_rule_before_scoring(
    sealed_run: Callable[[], SealedRun],
) -> None:
    run = sealed_run()
    scored = {r["slot_id"]: (r["outcome"], r["strict"]) for r in run.receipts
              if r["question_id"] in WRITER_RULE}  # fmt: skip
    assert scored == {f"{arm}:{qid}:1": ("correct", True) for qid in WRITER_RULE for arm in ARMS}


@pytest.mark.parametrize(("value", "normal"), [
    (None, None), (True, True), ("001", "001"), (7, 7),
    (float("nan"), "nan"), (float("inf"), "inf"), (Decimal("-Infinity"), "-inf"),
    (5.0, 5), (Decimal("5.00"), 5), (1 / 3, 0.3333333333),
    (Decimal("428785.71429"), 428785.7143),
    (datetime.datetime(2024, 1, 1), "2024-01-01"),
    (datetime.datetime(2024, 1, 1, 5), "2024-01-01T05:00:00"),
    (datetime.datetime(2024, 1, 1, tzinfo=datetime.UTC), "2024-01-01T00:00:00+00:00"),
    (datetime.date(2024, 1, 2), "2024-01-02"), (datetime.time(1, 2), "01:02:00"),
    (datetime.timedelta(seconds=90), 90), (datetime.timedelta(milliseconds=1500), 1.5),
    (b"\x01\xff", "01ff"), ([1.0, {"k": Decimal("2.50")}], [1, {"k": 2.5}]),
    ((2.0, "a"), [2, "a"]),
])  # fmt: skip
def test_the_normalizer_follows_the_writers_rule(value: Any, normal: Any) -> None:
    assert _cell(value) == normal and type(_cell(value)) is type(normal)

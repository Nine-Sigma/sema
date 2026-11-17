"""Done-when 5 and 7's dry run: arms A and S on a sealed set over a DuckDB source.

The entry under test is `showcase.ontology_query_benchmark.sealed_set`:

- `main(argv) -> int`: `--profile <yaml> --arms A,S --repetitions <n> [--dry-run]
  [--report <md>]`. `--dry-run` prints the lines `slots <n>` (questions x arms x
  repetitions) and `spend 0.00` and makes no model call.
- `run_sealed_set(profile_path, arms, repetitions, *, model, s_provider, report) -> Any`:
  builds questions and contracts from the set's sealed files, a generated harness config and
  arm A caches from the profile's DuckDB file, one executor per arm through the DuckDB
  connector, and runs every slot with the existing `runner.schedule` and `runner.run`
  (called as module attributes). `model` is a harness model client
  (`complete(messages) -> Generation`); `s_provider` is an `SProvider`
  (`(question, is_denied) -> SContextLike`). Receipts go under the set's `sealed/` folder;
  the unsealed markdown report goes to `report`.

The report holds one table, `| cell | key | arm | strict | order_free | n | strict_rate |`,
one row per cell, key and arm: `class` (question class), `hop_class`, `hops` (hop count),
`joins` (joins written in the arm's final statement), `join` (`with` or `without`) and
`excluded` (keyed by question id; a question `scoring.json` excludes from the rate is counted
only there). Counts are attempts; `strict_rate` is strict / n with two decimals. A section
whose heading holds `Open decisions` lists each `scoring.json` flag on a line with its
question id and flag name. No question text, SQL or row reaches stdout, a log line or any
file outside `sealed/`.
"""

import importlib
import io
import json
import socket
from collections.abc import Callable
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import duckdb
import pytest

from sema.connectors.duckdb import DuckDBConnector
from sema.log import logger
from showcase.ontology_query_benchmark import runner
from tests.support.connector_fixture import write_fixture_database
from tests.unit.ontology_query_benchmark.sealed_set_env import (
    ARM_REPLIES,
    MARKERS,
    ROW_MARKER,
    SEALED_SET_FIXTURE,
    SQL_MARKERS,
    TEXT_MARKER,
    ScriptedModel,
    expected_cells,
    materialize,
    question_texts,
    read_jsonl,
    report_cells,
    stub_s_provider,
)

pytestmark = pytest.mark.unit

ARMS = ("A", "S")
REPORT_CELLS = ("class", "hop_class", "hops", "joins", "join", "excluded")
QUESTION_FIELDS = {"id": str, "text": str, "question_class": str, "hop_class": int,
                   "hops": int, "answer_columns": list, "difficulty": int,
                   "underspecified": bool}  # fmt: skip


def _sealed_set() -> ModuleType:
    return importlib.import_module("showcase.ontology_query_benchmark.sealed_set")


@dataclass(frozen=True)
class SealedRun:
    root: Path
    set_dir: Path
    report: str
    receipts: list[dict[str, Any]]
    output: str
    logs: str
    model: ScriptedModel
    opened: list[Path]
    calls: list[str]

    def receipt(self, slot_id: str) -> dict[str, Any]:
        return next(r for r in self.receipts if r["slot_id"] == slot_id)


def _spy(mp: pytest.MonkeyPatch, module: Any, name: str, calls: list[str]) -> None:
    original = getattr(module, name)

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        calls.append(name)
        return original(*args, **kwargs)

    mp.setattr(module, name, wrapper)


def _spy_connector(mp: pytest.MonkeyPatch, opened: list[Path]) -> None:
    original = DuckDBConnector.__init__

    def init(self: DuckDBConnector, config: Any, *args: Any, **kwargs: Any) -> None:
        opened.append(Path(config.path).resolve())
        original(self, config, *args, **kwargs)

    mp.setattr(DuckDBConnector, "__init__", init)


def _run(harness: SimpleNamespace, root: Path, profile_path: Path) -> SealedRun:
    sealed_set = _sealed_set()
    set_dir = root / "neutral"
    model = ScriptedModel(harness, question_texts(set_dir))
    output, logs, opened, calls = io.StringIO(), [], [], []
    sink = logger.add(lambda message: logs.append(str(message)), level="INFO")
    try:
        with pytest.MonkeyPatch.context() as mp, redirect_stdout(output), redirect_stderr(output):
            _spy(mp, runner, "schedule", calls)
            _spy(mp, runner, "run", calls)
            _spy_connector(mp, opened)
            sealed_set.run_sealed_set(profile_path, ARMS, 1, model=model,
                                      s_provider=stub_s_provider, report=root / "report.md")
    finally:
        logger.remove(sink)
    receipts = [r for path in sorted((set_dir / "sealed").rglob("receipts.jsonl"))
                for r in read_jsonl(path)]  # fmt: skip
    return SealedRun(root, set_dir, (root / "report.md").read_text(), receipts,
                     output.getvalue(), "".join(logs), model, opened, calls)  # fmt: skip


@pytest.fixture(scope="module")
def sealed_run(
    frozen: Path, harness: SimpleNamespace, tmp_path_factory: pytest.TempPathFactory
) -> Callable[[], SealedRun]:
    """One run of both arms over the neutral set, made by the first test that asks."""
    root = tmp_path_factory.mktemp("sealed") / "sealed_set"
    profile_path = materialize(root, frozen)
    done: list[SealedRun] = []

    def get() -> SealedRun:
        if not done:
            done.append(_run(harness, root, profile_path))
        return done[0]

    return get


def _cell(value: Any) -> Any:
    # shortcut: the writer's normalization for the cell types this fixture returns (int,
    # integral float, text, NULL); extend it if a question returns another type.
    return int(value) if isinstance(value, float) and value.is_integer() else value


def test_the_neutral_set_follows_the_sealed_file_contract(tmp_path: Path) -> None:
    set_dir = SEALED_SET_FIXTURE / "neutral"
    questions = read_jsonl(set_dir / "sealed" / "questions.jsonl")
    answers = {r["id"]: r for r in read_jsonl(set_dir / "sealed" / "expected_answers.jsonl")
               if r["author"] == "A"}  # fmt: skip
    sql_a = read_jsonl(set_dir / "sealed" / "reference_sql_a.jsonl")
    reference = {r["id"]: r["sql"] for r in sql_a}
    flags = json.loads((set_dir / "scoring.json").read_text())["questions"]
    con = duckdb.connect(str(write_fixture_database(tmp_path)), read_only=True)
    for q in questions:
        assert set(q) == set(QUESTION_FIELDS), q["id"]
        assert all(isinstance(q[k], kind) for k, kind in QUESTION_FIELDS.items()), q["id"]
        assert q["question_class"] in tuple("ABCDEF")
        assert (q["hop_class"] > 0) == (q["question_class"] == "F"), q["id"]
        answer = answers[q["id"]]
        assert answer["status"] == "ok" and answer["rows_truncated"] is False
        assert [c["name"] for c in q["answer_columns"]] == answer["columns"]
        rows = [[_cell(c) for c in row] for row in con.execute(reference[q["id"]]).fetchall()]
        assert rows == answer["rows"], f"{q['id']} expected rows are stale"
    con.close()
    assert {q["question_class"] for q in questions} == set("ABCDEF")
    assert len({q["hop_class"] for q in questions if q["question_class"] == "F"}) >= 2
    empty = [q for q, f in flags.items() if f.get("excluded_from_rate")]
    null_flagged = [q for q, f in flags.items() if f.get("null_equals_empty")]
    assert len(empty) == len(null_flagged) == 1 and answers[empty[0]]["rows"] == []
    assert None in answers[null_flagged[0]]["rows"][0]
    marked = [q["id"] for q in questions if TEXT_MARKER in q["text"]]
    assert len(marked) == 1 and ROW_MARKER in answers[marked[0]]["rows"][0]
    assert set(ARM_REPLIES) == set(answers)


def test_dry_run_prints_the_slot_count_and_zero_spend_and_calls_nothing(
    frozen: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def no_network(*_: Any, **__: Any) -> None:
        raise AssertionError("a dry run opened a network connection")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    profile_path = materialize(tmp_path / "sealed_set", frozen)
    questions = read_jsonl(tmp_path / "sealed_set" / "neutral" / "sealed" / "questions.jsonl")

    code = _sealed_set().main(["--profile", str(profile_path), "--arms", "A,S",
                               "--repetitions", "2", "--dry-run"])  # fmt: skip

    lines = [line.strip() for line in capsys.readouterr().out.splitlines()]
    assert code == 0
    assert f"slots {len(questions) * 2 * 2}" in lines and "spend 0.00" in lines
    assert not list(tmp_path.rglob("receipts.jsonl"))


def test_both_arms_run_through_the_existing_runner_and_the_duckdb_connector(
    sealed_run: Callable[[], SealedRun],
) -> None:
    run = sealed_run()
    ids = list(question_texts(run.set_dir))
    assert run.calls == ["schedule", "run"]
    assert (run.root / "fixcat.duckdb").resolve() in run.opened
    assert [r["slot_id"] for r in run.receipts] == [f"{a}:{q}:1" for q in ids for a in ARMS]
    assert {(r["generation_calls"], r["sql_executions"]) for r in run.receipts} == {(1, 1)}


def test_every_attempt_is_scored_strict_as_its_statement_deserves(
    sealed_run: Callable[[], SealedRun],
) -> None:
    run = sealed_run()
    scored = {r["slot_id"]: (r["outcome"], r["strict"]) for r in run.receipts}
    expected = {f"{arm}:{qid}:1": ("correct", True) if reply.strict else ("wrong", False)
                for qid, replies in ARM_REPLIES.items() for arm, reply in replies.items()}
    assert scored == expected


def test_null_equals_empty_holds_on_its_flagged_question_only(
    sealed_run: Callable[[], SealedRun],
) -> None:
    run = sealed_run()
    assert run.receipt("A:N07:1")["strict"] is True
    assert run.receipt("A:N08:1")["strict"] is False
    assert run.receipt("A:N08:1")["outcome"] == "wrong"


def test_an_empty_expected_result_scores(sealed_run: Callable[[], SealedRun]) -> None:
    run = sealed_run()
    assert (run.receipt("A:N11:1")["outcome"], run.receipt("A:N11:1")["strict"]) == (
        "correct", True)
    assert run.receipt("S:N11:1")["outcome"] == "wrong"


def test_the_report_carries_every_query_report_cell(
    sealed_run: Callable[[], SealedRun],
) -> None:
    run = sealed_run()
    expected = expected_cells(run.set_dir, ARMS)
    assert expected[("class", "F", "A")] == (3, 3, 4, "0.75")
    assert expected[("class", "F", "S")] == (2, 3, 4, "0.50")
    assert expected[("join", "with", "S")] == (2, 3, 4, "0.50")
    assert expected[("excluded", "N11", "A")] == (1, 1, 1, "1.00")
    reported = {k: v for k, v in report_cells(run.report).items() if k[0] in REPORT_CELLS}
    assert reported == expected


def test_the_report_shows_the_open_decisions(sealed_run: Callable[[], SealedRun]) -> None:
    run = sealed_run()
    lines = run.report.splitlines()
    start = next(i for i, line in enumerate(lines) if "Open decisions" in line)
    section = lines[start + 1:]
    flags = json.loads((run.set_dir / "scoring.json").read_text())["questions"]
    for qid, flagged in flags.items():
        for flag in (name for name, value in flagged.items() if value is True):
            assert any(qid in line and flag in line for line in section), (qid, flag)


def test_receipts_land_under_the_sets_sealed_folder(sealed_run: Callable[[], SealedRun]) -> None:
    run = sealed_run()
    sealed = run.set_dir / "sealed"
    files = list(run.root.rglob("receipts.jsonl"))
    assert files and all(path.is_relative_to(sealed) for path in files)


def test_planted_markers_reach_no_stdout_log_or_unsealed_file(
    sealed_run: Callable[[], SealedRun],
) -> None:
    run = sealed_run()
    receipts_text = json.dumps(run.receipts)
    assert ROW_MARKER in receipts_text and SQL_MARKERS[0] in receipts_text
    assert TEXT_MARKER in json.dumps(run.model.calls)
    sealed = run.set_dir / "sealed"
    unsealed = {str(p.relative_to(run.root)): p.read_bytes().decode("utf-8", "replace")
                for p in run.root.rglob("*") if p.is_file() and not p.is_relative_to(sealed)}
    surfaces = {"stdout": run.output, "log": run.logs, **unsealed}
    leaks = [(name, marker) for name, text in surfaces.items() for marker in MARKERS
             if marker.lower() in text.lower()]  # fmt: skip
    assert leaks == []

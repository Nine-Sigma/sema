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
question id and flag name. No question text, SQL or row reaches stdout, a log line at any
level or any file outside `sealed/`.

Scoring: expected rows are author A's. Each executed attempt is scored twice by
`scorer.score`, strict and with `order_free=True`; the receipt keeps the second verdict in
`order_free` beside `strict`, and the report's `order_free` counts come from it. A question
whose author A record has `rows_truncated` true is refused: none of its attempts is scored,
and the report names it on a line holding its id and the word `refused`. Every statement
executes through the connector's `Connector.execute_query_with_columns(query) ->
tuple[list[str], list[tuple[Any, ...]]]` (column names, rows), once per statement.
"""

import json
import socket
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import duckdb
import pytest

from sema.connectors.base import Connector
from sema.connectors.duckdb import DuckDBConnector
from sema.models.config import DuckDBConfig
from tests.support.connector_fixture import write_fixture_database
from tests.unit.ontology_query_benchmark.sealed_set_env import (
    ACRIS_PROFILE,
    ARM_REPLIES,
    ARMS,
    AUTHOR_SPLIT,
    CONNECTOR_READ,
    MARKERS,
    NEAR_MISS,
    ROW_MARKER,
    SEALED_SET_FIXTURE,
    SINK_CANARY,
    SQL_MARKERS,
    TEXT_MARKER,
    WIDE,
    ScriptedModel,
    SealedRun,
    dialects_named,
    expected_cells,
    materialize,
    question_texts,
    read_jsonl,
    rename_question_ids,
    report_cells,
    sealed_run,  # noqa: F401 - a fixture
    truncate_expected,
)
from tests.unit.ontology_query_benchmark.sealed_set_env import run_arms as _run
from tests.unit.ontology_query_benchmark.sealed_set_env import sealed_set_module as _sealed_set
from tests.unit.ontology_query_benchmark.sealed_set_env import writer_cell as _cell

pytestmark = pytest.mark.unit

REPORT_CELLS = ("class", "hop_class", "hops", "joins", "join", "excluded")
SCORED = ("correct", "wrong", "context_incomplete")
QUESTION_FIELDS = {"id": str, "text": str, "question_class": str, "hop_class": int,
                   "hops": int, "answer_columns": list, "difficulty": int,
                   "underspecified": bool}  # fmt: skip


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
    assert SINK_CANARY in run.logs, "the log sink must see every level from TRACE"
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


def test_order_free_is_the_scorers_order_free_verdict_on_every_scored_attempt(
    sealed_run: Callable[[], SealedRun],
) -> None:
    run = sealed_run()
    near, wide = run.receipt(f"S:{NEAR_MISS}:1"), run.receipt(f"S:{WIDE}:1")
    assert near["tolerant"] is True and near["order_free"] is False
    assert wide["order_free"] is True and len(wide["answer"]) < len(_rows_of(run, WIDE, "A"))
    assert {r["slot_id"]: r["order_free"] for r in run.receipts} == {
        f"{arm}:{qid}:1": reply.order_free
        for qid, replies in ARM_REPLIES.items() for arm, reply in replies.items()}
    scored = Counter(r["question_id"] for r in run.receipts if r["outcome"] in SCORED)
    asked = Counter(qid for qid, order_free in run.scores if order_free)
    assert scored and all(asked[qid] >= n for qid, n in scored.items()), (scored, asked)


def _rows_of(run: SealedRun, qid: str, author: str) -> list[Any]:
    path = run.set_dir / "sealed" / "expected_answers.jsonl"
    return next(r["rows"] for r in read_jsonl(path) if (r["id"], r["author"]) == (qid, author))


def test_expected_rows_are_author_as_when_author_b_disagrees(
    sealed_run: Callable[[], SealedRun],
) -> None:
    run = sealed_run()
    assert _rows_of(run, AUTHOR_SPLIT, "A") != _rows_of(run, AUTHOR_SPLIT, "B")
    assert (run.receipt(f"A:{AUTHOR_SPLIT}:1")["outcome"],
            run.receipt(f"A:{AUTHOR_SPLIT}:1")["strict"]) == ("correct", True)  # fmt: skip
    assert run.receipt(f"S:{AUTHOR_SPLIT}:1")["strict"] is False


def test_the_connector_read_returns_column_names_with_the_rows(tmp_path: Path) -> None:
    assert callable(getattr(Connector, CONNECTOR_READ, None))
    connector = DuckDBConnector(DuckDBConfig(path=str(write_fixture_database(tmp_path))))
    try:
        columns, rows = connector.execute_query_with_columns(
            "SELECT region_code, count(*) AS customers FROM shop.customer "
            "GROUP BY region_code ORDER BY region_code")
    finally:
        connector.close()
    assert (list(columns), [tuple(r) for r in rows]) == (
        ["region_code", "customers"], [("N", 66), ("S", 67), ("W", 67)])


def _squash(sql: str) -> str:
    return "".join(sql.lower().split())


def test_every_statement_executes_once_through_the_connector_read(
    sealed_run: Callable[[], SealedRun],
) -> None:
    run = sealed_run()
    assert len(run.model.replies) == len(run.receipts) == sum(
        r["sql_executions"] for r in run.receipts)
    for k, statement in enumerate(run.model.replies, start=1):
        hits = [sql for after, sql in run.reads
                if after == k and _squash(statement) in _squash(sql)]  # fmt: skip
        assert len(hits) == 1, (k, len(hits))


def test_a_truncated_expected_record_is_refused_by_id_and_status_word(
    frozen: Path, harness: SimpleNamespace, tmp_path: Path,
) -> None:
    root = tmp_path / "sealed_set"
    profile_path = materialize(root, frozen)
    truncate_expected(root / "neutral", WIDE, keep=5)

    run = _run(harness, root, profile_path, arms=("A",))

    assert [r["slot_id"] for r in run.receipts
            if r["question_id"] == WIDE and r["outcome"] in SCORED] == []  # fmt: skip
    assert WIDE not in {qid for qid, _ in run.scores}
    assert run.receipt("A:N01:1")["outcome"] == "correct"
    assert any(WIDE in line and "refused" in line for line in run.report.splitlines())


def test_renamed_question_ids_give_the_same_per_rule_results(
    sealed_run: Callable[[], SealedRun], frozen: Path, harness: SimpleNamespace, tmp_path: Path,
) -> None:
    neutral = sealed_run()
    root = tmp_path / "sealed_set"
    profile_path = materialize(root, frozen)
    renamed = rename_question_ids(root / "neutral")
    back = {new: old for old, new in renamed.items()}

    decoy = _run(harness, root, profile_path)

    def verdicts(run: SealedRun, to_neutral: dict[str, str]) -> dict[tuple[str, str], Any]:
        return {(r["arm"], to_neutral.get(r["question_id"], r["question_id"])):
                (r["outcome"], r["strict"], r.get("order_free")) for r in run.receipts}  # fmt: skip

    assert verdicts(decoy, back) == verdicts(neutral, {})
    assert report_cells(decoy.report) == {
        (cell, renamed[key] if cell == "excluded" else key, arm): value
        for (cell, key, arm), value in report_cells(neutral.report).items()}
    flags = json.loads((root / "neutral" / "scoring.json").read_text())["questions"]
    lines = decoy.report.splitlines()
    assert all(any(qid in line and flag in line for line in lines)
               for qid, flagged in flags.items() for flag in flagged if flagged[flag] is True)
    assert not any(old in decoy.report for old in renamed)


def test_the_openrouter_client_speaks_the_harness_model_protocol(harness: SimpleNamespace) -> None:
    from showcase.ontology_query_benchmark import runner_live_utils as live

    sent: list[tuple[str, dict[str, Any], dict[str, str], float]] = []

    def post(url: str, body: dict[str, Any], headers: dict[str, str], timeout: float) -> Any:
        sent.append((url, body, headers, timeout))
        return {"choices": [{"message": {"content": "```sql\nSELECT 1\n```"}}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 3}}  # fmt: skip

    client = live.OpenRouterClient(harness, "vendor/model-x", "key-1", max_tokens=99,
                                   timeout=5.0, post=post)  # fmt: skip
    generation = client.complete([{"role": "user", "content": "q"}])

    ((url, body, headers, timeout),) = sent
    assert url == "https://openrouter.ai/api/v1/chat/completions"
    assert body == {"model": "vendor/model-x", "max_tokens": 99,
                    "messages": [{"role": "user", "content": "q"}]}  # fmt: skip
    assert headers["Authorization"] == "Bearer key-1" and timeout == 5.0
    assert isinstance(generation, harness.models.Generation)
    assert (generation.text, generation.prompt_tokens, generation.completion_tokens) == (
        "```sql\nSELECT 1\n```", 12, 3)


def test_a_runner_error_receipt_records_order_free_false(
    frozen: Path, harness: SimpleNamespace, tmp_path: Path,
) -> None:
    def failing(question: str, is_denied: Any) -> Any:
        raise RuntimeError("provider down")

    root = tmp_path / "sealed_set"
    profile_path = materialize(root, frozen)
    _sealed_set().run_sealed_set(profile_path, ("S",), 1, model=ScriptedModel(harness, {}),
                                 s_provider=failing, report=tmp_path / "report.md")  # fmt: skip

    receipts = read_jsonl(next((root / "neutral" / "sealed").rglob("receipts.jsonl")))
    assert receipts and {(r["outcome"], r.get("order_free", "absent")) for r in receipts} == {
        ("runner_error", False)}


def test_a_statement_past_the_row_cap_or_the_timeout_returns_promptly_with_its_status_word(
    harness: SimpleNamespace, tmp_path: Path,
) -> None:
    from showcase.ontology_query_benchmark.profile import load_sealed_profile
    from showcase.ontology_query_benchmark.runner_live_utils import ConnectorExecutor

    profiles = (SEALED_SET_FIXTURE / "profile.yaml", ACRIS_PROFILE)
    assert all(load_sealed_profile(p).limits.statement_timeout_s > 0 for p in profiles)
    connector = DuckDBConnector(DuckDBConfig(path=str(write_fixture_database(tmp_path))))
    executor, started = ConnectorExecutor(harness, connector, 1.0), time.monotonic()
    try:
        done = [executor.execute(sql, 10) for sql in (
            "SELECT i FROM range(1000000000) r(i)",
            "SELECT sum(i * i) AS n FROM range(20000000000) r(i)",
            "SELECT count(*) AS n FROM shop.customer")]  # fmt: skip
    finally:
        connector.close()
    assert time.monotonic() - started < 15
    assert [(e.ok, e.error_class) for e in done] == [
        (False, "row_cap"), (False, "timeout"), (True, None)]
    assert done[2].rows == ((200,),)


def test_both_arms_prompts_name_the_executors_dialect_and_no_other(
    frozen: Path, harness: SimpleNamespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    framed, build = [], harness.workflow.build_messages
    monkeypatch.setattr(harness.workflow, "build_messages", lambda context, *rest: (
        framed.append(context.dialect) or build(context, *rest)))  # fmt: skip
    run = _run(harness, tmp_path / "sealed_set", materialize(tmp_path / "sealed_set", frozen))
    prompts = ["\n".join(str(m["content"]) for m in call) for call in run.model.calls]
    assert len(framed) == len(prompts) == len(run.receipts) > 0
    assert {frozenset(dialects_named(text)) for text in (*framed, *prompts)} == {
        frozenset({DuckDBConnector.sql_dialect})}

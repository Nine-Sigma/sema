import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from showcase.partner_timbr_benchmark import runner, runner_utils, scorer
from showcase.partner_timbr_benchmark.routing import RoutingError

pytestmark = pytest.mark.unit

Q01_SQL = "```sql\nSELECT count(DISTINCT mrn) AS patients FROM main.s.person\n```"

@dataclass(frozen=True)
class FakeSContext:
    text: str
    identity: str
    findings: tuple[str, ...] = ()


class FakeModel:
    def __init__(self, harness: SimpleNamespace, replies: list[str], prompt: int = 900,
                 completion: int = 40) -> None:  # fmt: skip
        self.harness, self.replies = harness, replies
        self.prompt, self.completion = prompt, completion
        self.calls: list[list[dict[str, Any]]] = []

    def complete(self, messages: list[dict[str, Any]]) -> Any:
        self.calls.append(messages)
        text = self.replies[min(len(self.calls), len(self.replies)) - 1]
        return self.harness.models.Generation(text, self.prompt, self.completion, 0.01)


class FakeExecutor:
    def __init__(self, harness: SimpleNamespace, rows: tuple[tuple[Any, ...], ...]) -> None:
        self.harness, self.rows = harness, rows
        self.sqls: list[str] = []

    def execute(self, sql: str, row_limit: int) -> Any:
        self.sqls.append(sql)
        return self.harness.models.Execution(True, ("patients",), self.rows, 0.01)


def s_provider(findings: tuple[str, ...] = (), text: str = "TABLE main.s.person\n  mrn string"):
    return lambda question: FakeSContext(text, "s-identity-1", findings)


def settings(max_cost: float = 100.0) -> runner.RunSettings:
    prices = runner_utils.Prices(input_usd_per_1m=2.0, output_usd_per_1m=10.0)
    return runner.RunSettings(runner_utils.Limits(), prices, max_cost)


def run_q01(harness: SimpleNamespace, frozen: Path, model: FakeModel, executor: FakeExecutor,
            provider: Callable[[str], FakeSContext]) -> dict[str, Any]:  # fmt: skip
    runtimes = runner_utils.arm_runtimes(harness, frozen, {"S": executor}, provider)
    question = runner_utils.public_questions(harness, frozen, ("Q01",))[0]
    contract = scorer.load_contracts(runner_utils.question_paths(frozen))["Q01"]
    slot = runner.Slot(0, "S", "Q01", 1)
    return runner.run_slot(slot, question, runtimes["S"], model, contract, harness, settings())


def test_schedule_is_441_unique_deterministic_slots(frozen: Path) -> None:
    ids = runner_utils.question_ids(frozen)
    slots = runner.schedule(ids, ("A", "B", "S"), 3)
    assert len(ids) == 49 and "Q14" not in ids
    assert len(slots) == 441 == len({s.slot_id for s in slots})
    assert slots == runner.schedule(ids, ("A", "B", "S"), 3)
    assert [s.index for s in slots] == list(range(441))


def test_dry_run_writes_the_slot_list_and_makes_no_calls(frozen: Path, tmp_path: Path) -> None:
    code = runner.main(["--frozen", str(frozen), "--run-dir", str(tmp_path), "--dry-run"])
    lines = (tmp_path / "slots.jsonl").read_text().splitlines()
    assert code == 0 and len(lines) == 441
    assert not (tmp_path / "receipts.jsonl").exists()
    runner.main(["--frozen", str(frozen), "--run-dir", str(tmp_path / "m"), "--dry-run",
                 "--ids", "Q01", "--arms", "S", "--reps", "1"])  # fmt: skip
    lines = (tmp_path / "m/slots.jsonl").read_text().splitlines()
    assert [json.loads(line)["slot_id"] for line in lines] == ["S:Q01:1"]


def test_questions_reach_the_harness_without_gold(harness: SimpleNamespace, frozen: Path) -> None:
    question = runner_utils.public_questions(harness, frozen, ("Q01",))[0]
    assert question.gold_answer is None and question.gold_sql == ""


def test_s_slot_is_scored_by_the_typed_scorer_with_a_full_receipt(
    harness: SimpleNamespace, frozen: Path
) -> None:
    model, executor = FakeModel(harness, [Q01_SQL]), FakeExecutor(harness, ((939376,),))
    receipt = run_q01(harness, frozen, model, executor, s_provider())
    assert receipt["outcome"] == "correct" and receipt["strict"] is True
    assert receipt["first_sql"] == receipt["final_sql"] == executor.sqls[0]
    assert receipt["context_identity"] == "s-identity-1"
    assert (receipt["generation_calls"], receipt["sql_executions"]) == (1, 1)
    assert (receipt["prompt_tokens"], receipt["completion_tokens"]) == (900, 40)
    assert receipt["earliest_failure_stage"] is None
    assert "939376" not in json.dumps(model.calls)


def test_wrong_value_is_wrong_at_the_scoring_stage(harness: SimpleNamespace, frozen: Path) -> None:
    model, executor = FakeModel(harness, [Q01_SQL]), FakeExecutor(harness, ((5,),))
    receipt = run_q01(harness, frozen, model, executor, s_provider())
    assert (receipt["outcome"], receipt["earliest_failure_stage"]) == ("wrong", "scoring")


def test_completeness_findings_withhold_credit(harness: SimpleNamespace, frozen: Path) -> None:
    model, executor = FakeModel(harness, [Q01_SQL]), FakeExecutor(harness, ((939376,),))
    receipt = run_q01(harness, frozen, model, executor, s_provider(("missing:gender",)))
    assert receipt["outcome"] == "context_incomplete" and receipt["strict"] is False
    assert receipt["raw_match"] is True and receipt["earliest_failure_stage"] == "context"


def test_truncated_transport_is_not_credited(harness: SimpleNamespace, frozen: Path) -> None:
    model = FakeModel(harness, [Q01_SQL])
    executor = FakeExecutor(harness, tuple((939376,) for _ in range(1000)))
    receipt = run_q01(harness, frozen, model, executor, s_provider())
    assert receipt["outcome"] == "wrong" and receipt["score_reason"] == "incomplete"


def test_pre_call_estimate_over_the_ceiling_fails_closed(
    harness: SimpleNamespace, frozen: Path
) -> None:
    model, executor = FakeModel(harness, [Q01_SQL]), FakeExecutor(harness, ((939376,),))
    receipt = run_q01(harness, frozen, model, executor, s_provider(text="x" * 40_000))
    assert (receipt["outcome"], receipt["overflow"]) == ("overflow", "pre_call")
    assert model.calls == [] and receipt["generation_calls"] == 0


@pytest.mark.parametrize(("prompt", "completion"), [(16_001, 40), (900, 8_193)])
def test_endpoint_usage_over_a_ceiling_marks_overflow(
    harness: SimpleNamespace, frozen: Path, prompt: int, completion: int
) -> None:
    model = FakeModel(harness, [Q01_SQL], prompt=prompt, completion=completion)
    executor = FakeExecutor(harness, ((939376,),))
    receipt = run_q01(harness, frozen, model, executor, s_provider())
    assert (receipt["outcome"], receipt["overflow"]) == ("overflow", "post_call")
    assert executor.sqls == [] and receipt["generation_calls"] == 1


def test_harness_owns_at_most_two_repairs(harness: SimpleNamespace, frozen: Path) -> None:
    model = FakeModel(harness, ["```sql\nDELETE FROM main.s.person\n```"])
    executor = FakeExecutor(harness, ((939376,),))
    receipt = run_q01(harness, frozen, model, executor, s_provider())
    assert len(model.calls) == receipt["generation_calls"] == 3 and receipt["repairs"] == 2
    assert receipt["outcome"] == "validation_failure"
    assert receipt["earliest_failure_stage"] == "validation" and executor.sqls == []


def test_a_and_b_contexts_build_from_the_frozen_caches(
    harness: SimpleNamespace, frozen: Path
) -> None:
    executors = {"A": FakeExecutor(harness, ()), "B": FakeExecutor(harness, ())}
    runtimes = runner_utils.arm_runtimes(harness, frozen, executors, None)
    a = runtimes["A"].context("How many living patients are female?")
    b = runtimes["B"].context("How many living patients are female?")
    assert (a.arm, b.arm) == ("A", "B") and a.text and b.text
    assert runtimes["B"].known_concepts and runtimes["A"].known_concepts == ()
    assert {r.max_repairs for r in runtimes.values()} == {2}
    assert {r.row_limit for r in runtimes.values()} == {1000}


def _run(harness: SimpleNamespace, frozen: Path, out: Path, model: FakeModel,
         run_settings: runner.RunSettings) -> dict[str, Any]:  # fmt: skip
    slots = runner.schedule(["Q01"], ("S",), 3)
    executors = {"S": FakeExecutor(harness, ((939376,),))}
    runtimes = runner_utils.arm_runtimes(harness, frozen, executors, s_provider())
    questions = {q.id: q for q in runner_utils.public_questions(harness, frozen, ("Q01",))}
    contracts = scorer.load_contracts(runner_utils.question_paths(frozen))
    return runner.run(slots, questions, runtimes, model, contracts, harness, run_settings, out)


def _receipts(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_resume_never_repeats_a_recorded_slot(harness: SimpleNamespace, frozen: Path,
                                              tmp_path: Path) -> None:  # fmt: skip
    out = tmp_path / "receipts.jsonl"
    out.write_text(json.dumps({"slot_id": "S:Q01:1", "outcome": "wrong", "cost_usd": 0.0}) + "\n")
    model = FakeModel(harness, [Q01_SQL])
    _run(harness, frozen, out, model, settings())
    assert len(model.calls) == 2
    assert [r["slot_id"] for r in _receipts(out)] == ["S:Q01:1", "S:Q01:2", "S:Q01:3"]


def test_cost_ceiling_stops_and_marks_the_rest_unrun(harness: SimpleNamespace, frozen: Path,
                                                     tmp_path: Path) -> None:  # fmt: skip
    out = tmp_path / "receipts.jsonl"
    model = FakeModel(harness, [Q01_SQL])
    summary = _run(harness, frozen, out, model, settings(max_cost=0.001))
    outcomes = [r["outcome"] for r in _receipts(out)]
    assert outcomes == ["correct", "unrun", "unrun"] and len(model.calls) == 1
    assert summary["stopped"] == "cost_ceiling"


def test_b_executor_refuses_timbr_side_llm_tools() -> None:
    calls: list[str] = []
    executor = SimpleNamespace(call_tool=lambda tool, arguments: calls.append(tool) or {})
    guarded = runner_utils.guard_mcp_tools(executor)
    guarded.call_tool("execute_sql", {})
    with pytest.raises(RoutingError):
        guarded.call_tool("query_data", {})
    assert calls == ["execute_sql"]


def test_timbr_env_file_is_read_without_touching_the_environment(tmp_path: Path) -> None:
    env = tmp_path / "poc.env"
    lines = ("TIMBR_TOKEN='tok'", "DATABRICKS_HOST=https://elsewhere", "# c",
             "TIMBR_VERIFY_SSL=false")
    env.write_text("\n".join(lines) + "\n")
    before = dict(os.environ)
    values = runner_utils.read_env_keys(env, ("TIMBR_TOKEN", "TIMBR_VERIFY_SSL", "TIMBR_CA_BUNDLE"))
    assert values == {"TIMBR_TOKEN": "tok", "TIMBR_VERIFY_SSL": "false"}
    assert dict(os.environ) == before

import dataclasses
import itertools
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from showcase.ontology_query_benchmark import runner, runner_utils, scorer
from showcase.ontology_query_benchmark.profile import Limits, Profile
from showcase.ontology_query_benchmark.routing import RoutingError
from tests.unit.ontology_query_benchmark.conftest import PROFILE_PATH

pytestmark = pytest.mark.unit

R01_SQL = "```sql\nSELECT count(DISTINCT rid) AS people FROM main.s.person\n```"
GOLD = 42

@dataclass(frozen=True)
class FakeSContext:
    text: str
    identity: str
    findings: tuple[str, ...] = ()


class FakeModel:
    def __init__(self, harness: SimpleNamespace, replies: list[str], prompt: int | None = 900,
                 completion: int | None = 40) -> None:  # fmt: skip
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
        return self.harness.models.Execution(True, ("people",), self.rows, 0.01)


def s_provider(findings: tuple[str, ...] = (), text: str = "TABLE main.s.person\n  rid string"):
    return lambda question: FakeSContext(text, "s-identity-1", findings)


PRICES = runner_utils.Prices(input_usd_per_1m=2.0, output_usd_per_1m=10.0)


def settings(profile: Profile, max_cost: float = 100.0,
             limits: Limits | None = None) -> runner.RunSettings:  # fmt: skip
    return runner.RunSettings(limits or profile.limits, PRICES, max_cost, max_wall_s=3600.0)


def runtimes_for(harness: SimpleNamespace, frozen: Path, profile: Profile,
                 executors: dict[str, Any], provider: Any) -> dict[str, Any]:  # fmt: skip
    return runner_utils.arm_runtimes(harness, frozen, profile.harness, executors, provider,
                                     profile.limits)  # fmt: skip


def run_r01(harness: SimpleNamespace, frozen: Path, profile: Profile, model: FakeModel,
            executor: FakeExecutor, provider: Callable[[str], FakeSContext],
            limits: Limits | None = None) -> dict[str, Any]:  # fmt: skip
    runtimes = runtimes_for(harness, frozen, profile, {"S": executor}, provider)
    question = runner_utils.public_questions(harness, frozen, profile.harness, ("R01",))[0]
    paths = runner_utils.question_paths(frozen, profile.harness)
    contract = scorer.load_contracts(paths, profile.harness.retired_questions)["R01"]
    slot = runner.Slot(0, "S", "R01", 1)
    return runner.run_slot(slot, question, runtimes["S"], model, contract, harness,
                           settings(profile, limits=limits))  # fmt: skip


def test_schedule_is_unique_deterministic_slots(frozen: Path, profile: Profile) -> None:
    ids = runner_utils.question_ids(frozen, profile.harness)
    slots = runner.schedule(ids, ("A", "B", "S"), 3)
    assert ids == ["R01", "R02", "H01"]
    assert len(slots) == 27 == len({s.slot_id for s in slots})
    assert slots == runner.schedule(ids, ("A", "B", "S"), 3)
    assert [s.index for s in slots] == list(range(27))


def test_dry_run_writes_the_slot_list_and_makes_no_calls(frozen: Path, tmp_path: Path) -> None:
    base = ["--profile", str(PROFILE_PATH), "--frozen", str(frozen)]
    code = runner.main([*base, "--run-dir", str(tmp_path), "--dry-run"])
    lines = (tmp_path / "slots.jsonl").read_text().splitlines()
    assert code == 0 and len(lines) == 27
    assert not (tmp_path / "receipts.jsonl").exists()
    runner.main([*base, "--run-dir", str(tmp_path / "m"), "--dry-run",
                 "--ids", "R01", "--arms", "S", "--reps", "1"])  # fmt: skip
    lines = (tmp_path / "m/slots.jsonl").read_text().splitlines()
    assert [json.loads(line)["slot_id"] for line in lines] == ["S:R01:1"]


def test_questions_reach_the_harness_without_gold(
    harness: SimpleNamespace, frozen: Path, profile: Profile
) -> None:
    question = runner_utils.public_questions(harness, frozen, profile.harness, ("R01",))[0]
    assert question.gold_answer is None and question.gold_sql == ""


def test_s_slot_is_scored_by_the_typed_scorer_with_a_full_receipt(
    harness: SimpleNamespace, frozen: Path, profile: Profile
) -> None:
    model, executor = FakeModel(harness, [R01_SQL]), FakeExecutor(harness, ((GOLD,),))
    receipt = run_r01(harness, frozen, profile, model, executor, s_provider())
    assert receipt["outcome"] == "correct" and receipt["strict"] is True
    assert receipt["first_sql"] == receipt["final_sql"] == executor.sqls[0]
    assert receipt["context_identity"] == "s-identity-1"
    assert (receipt["generation_calls"], receipt["sql_executions"]) == (1, 1)
    assert (receipt["prompt_tokens"], receipt["completion_tokens"]) == (900, 40)
    assert receipt["earliest_failure_stage"] is None
    assert str(GOLD) not in json.dumps(model.calls)


def test_wrong_value_is_wrong_at_the_scoring_stage(
    harness: SimpleNamespace, frozen: Path, profile: Profile
) -> None:
    model, executor = FakeModel(harness, [R01_SQL]), FakeExecutor(harness, ((5,),))
    receipt = run_r01(harness, frozen, profile, model, executor, s_provider())
    assert (receipt["outcome"], receipt["earliest_failure_stage"]) == ("wrong", "scoring")


def test_completeness_findings_withhold_credit(
    harness: SimpleNamespace, frozen: Path, profile: Profile
) -> None:
    model, executor = FakeModel(harness, [R01_SQL]), FakeExecutor(harness, ((GOLD,),))
    receipt = run_r01(harness, frozen, profile, model, executor, s_provider(("missing:gender",)))
    assert receipt["outcome"] == "context_incomplete" and receipt["strict"] is False
    assert receipt["raw_match"] is True and receipt["earliest_failure_stage"] == "context"


def test_truncated_transport_is_not_credited(
    harness: SimpleNamespace, frozen: Path, profile: Profile
) -> None:
    model = FakeModel(harness, [R01_SQL])
    executor = FakeExecutor(harness, tuple((GOLD,) for _ in range(1000)))
    receipt = run_r01(harness, frozen, profile, model, executor, s_provider())
    assert receipt["outcome"] == "wrong" and receipt["score_reason"] == "incomplete"


def test_repair_request_over_the_ceiling_fails_closed_before_the_call(
    harness: SimpleNamespace, frozen: Path, profile: Profile
) -> None:
    long_invalid = "```sql\nDELETE FROM main.s.person\n```\n" + "y" * 40_000
    model = FakeModel(harness, [long_invalid])
    executor = FakeExecutor(harness, ((GOLD,),))
    receipt = run_r01(harness, frozen, profile, model, executor, s_provider())
    assert (receipt["outcome"], receipt["overflow"]) == ("overflow", "pre_call")
    assert len(model.calls) == receipt["generation_calls"] == 1


@pytest.mark.parametrize(("prompt", "completion"), [(16_001, 40), (900, 8_193)])
def test_endpoint_usage_over_a_ceiling_marks_overflow(
    harness: SimpleNamespace, frozen: Path, profile: Profile, prompt: int, completion: int
) -> None:
    model = FakeModel(harness, [R01_SQL], prompt=prompt, completion=completion)
    executor = FakeExecutor(harness, ((GOLD,),))
    receipt = run_r01(harness, frozen, profile, model, executor, s_provider())
    assert (receipt["outcome"], receipt["overflow"]) == ("overflow", "post_call")
    assert executor.sqls == [] and receipt["generation_calls"] == 1


def test_harness_owns_at_most_two_repairs(
    harness: SimpleNamespace, frozen: Path, profile: Profile
) -> None:
    model = FakeModel(harness, ["```sql\nDELETE FROM main.s.person\n```"])
    executor = FakeExecutor(harness, ((GOLD,),))
    receipt = run_r01(harness, frozen, profile, model, executor, s_provider())
    assert len(model.calls) == receipt["generation_calls"] == 3 and receipt["repairs"] == 2
    assert receipt["outcome"] == "validation_failure"
    assert receipt["earliest_failure_stage"] == "validation" and executor.sqls == []


def test_a_and_b_contexts_build_from_the_frozen_caches(
    harness: SimpleNamespace, frozen: Path, profile: Profile
) -> None:
    executors = {"A": FakeExecutor(harness, ()), "B": FakeExecutor(harness, ())}
    runtimes = runtimes_for(harness, frozen, profile, executors, None)
    a = runtimes["A"].context("How many living patients are female?")
    b = runtimes["B"].context("How many living patients are female?")
    assert (a.arm, b.arm) == ("A", "B") and a.text and b.text
    assert runtimes["B"].known_concepts and runtimes["A"].known_concepts == ()
    assert {r.max_repairs for r in runtimes.values()} == {2}
    assert {r.row_limit for r in runtimes.values()} == {1000}


def _run(
    harness: SimpleNamespace, frozen: Path, profile: Profile, out: Path, model: FakeModel,
    run_settings: runner.RunSettings, **clock: Callable[[], float],
) -> dict[str, Any]:
    slots = runner.schedule(["R01"], ("S",), 3)
    executors = {"S": FakeExecutor(harness, ((GOLD,),))}
    runtimes = runtimes_for(harness, frozen, profile, executors, s_provider())
    questions, contracts = _questions_and_contracts(harness, frozen, profile)
    return runner.run(
        slots, questions, runtimes, model, contracts, harness, run_settings, out, **clock
    )


def _questions_and_contracts(
    harness: SimpleNamespace, frozen: Path, profile: Profile
) -> tuple[dict[str, Any], dict[str, Any]]:
    layout = profile.harness
    questions = {q.id: q for q in runner_utils.public_questions(harness, frozen, layout, ("R01",))}
    paths = runner_utils.question_paths(frozen, layout)
    return questions, scorer.load_contracts(paths, layout.retired_questions)


def _receipts(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_resume_never_repeats_a_recorded_slot(
    harness: SimpleNamespace, frozen: Path, profile: Profile, tmp_path: Path
) -> None:
    out = tmp_path / "receipts.jsonl"
    out.write_text(json.dumps({"slot_id": "S:R01:1", "outcome": "wrong", "cost_usd": 0.0}) + "\n")
    model = FakeModel(harness, [R01_SQL])
    _run(harness, frozen, profile, out, model, settings(profile))
    assert len(model.calls) == 2
    assert [r["slot_id"] for r in _receipts(out)] == ["S:R01:1", "S:R01:2", "S:R01:3"]


def test_cost_ceiling_stops_and_marks_the_rest_unrun(
    harness: SimpleNamespace, frozen: Path, profile: Profile, tmp_path: Path
) -> None:
    out = tmp_path / "receipts.jsonl"
    model = FakeModel(harness, [R01_SQL])
    summary = _run(harness, frozen, profile, out, model, settings(profile, max_cost=0.001))
    outcomes = [r["outcome"] for r in _receipts(out)]
    assert outcomes == ["correct", "unrun", "unrun"] and len(model.calls) == 1
    assert summary["stopped"] == "cost_ceiling"


def test_time_ceiling_stops_and_marks_the_rest_unrun(
    harness: SimpleNamespace, frozen: Path, profile: Profile, tmp_path: Path
) -> None:
    out, ticks = tmp_path / "receipts.jsonl", itertools.count(0.0, 6.0)
    model = FakeModel(harness, [R01_SQL])
    run_settings = dataclasses.replace(settings(profile), max_wall_s=10.0)
    summary = _run(harness, frozen, profile, out, model, run_settings, clock=lambda: next(ticks))
    receipts = _receipts(out)
    assert [r["outcome"] for r in receipts] == ["correct", "unrun", "unrun"]
    assert [r.get("reason") for r in receipts[1:]] == ["time_ceiling"] * 2
    assert summary["stopped"] == "time_ceiling" and len(model.calls) == 1


def test_b_executor_refuses_engine_side_llm_tools(profile: Profile) -> None:
    calls: list[str] = []
    executor = SimpleNamespace(call_tool=lambda tool, arguments: calls.append(tool) or {})
    guarded = runner_utils.guard_mcp_tools(executor, profile.timbr)
    guarded.call_tool("execute_sql", {})
    with pytest.raises(RoutingError):
        guarded.call_tool("query_data", {})
    assert calls == ["execute_sql"]


def test_engine_env_file_is_read_without_touching_the_environment(tmp_path: Path) -> None:
    env = tmp_path / "engine.env"
    lines = ("TOY_TOKEN='tok'", "DATABRICKS_HOST=https://elsewhere", "# c",
             "TOY_VERIFY_SSL=false  # local CA")
    env.write_text("\n".join(lines) + "\n")
    before = dict(os.environ)
    values = runner_utils.read_env_keys(env, ("TOY_TOKEN", "TOY_VERIFY_SSL", "TOY_CA_BUNDLE"))
    assert values == {"TOY_TOKEN": "tok", "TOY_VERIFY_SSL": "false"}
    assert dict(os.environ) == before


class RaisingExecutor(FakeExecutor):
    def execute(self, sql: str, row_limit: int) -> Any:
        self.sqls.append(sql)
        raise ConnectionError("warehouse unreachable")


def test_a_raising_context_provider_becomes_a_runner_error_receipt(
    harness: SimpleNamespace, frozen: Path, profile: Profile, tmp_path: Path
) -> None:
    def broken(question: str) -> FakeSContext:
        raise RuntimeError("index not loaded")

    slots = runner.schedule(["R01"], ("S",), 2)
    runtimes = runtimes_for(harness, frozen, profile, {"S": FakeExecutor(harness, ())}, broken)
    questions, contracts = _questions_and_contracts(harness, frozen, profile)
    out = tmp_path / "receipts.jsonl"
    model = FakeModel(harness, [R01_SQL])
    runner.run(slots, questions, runtimes, model, contracts, harness, settings(profile), out)
    receipts = _receipts(out)
    assert [r["outcome"] for r in receipts] == ["runner_error", "runner_error"]
    assert {r["earliest_failure_stage"] for r in receipts} == {"context"}
    assert "index not loaded" in receipts[0]["error_message"] and model.calls == []


def test_a_raising_executor_becomes_a_runner_error_at_execution(
    harness: SimpleNamespace, frozen: Path, profile: Profile
) -> None:
    model, executor = FakeModel(harness, [R01_SQL]), RaisingExecutor(harness, ())
    receipt = run_r01(harness, frozen, profile, model, executor, s_provider())
    assert (receipt["outcome"], receipt["earliest_failure_stage"]) == ("runner_error", "execution")
    assert (receipt["generation_calls"], receipt["sql_executions"]) == (1, 1)


def test_first_prompt_over_the_repair_reserved_limit_is_a_context_overflow(
    harness: SimpleNamespace, frozen: Path, profile: Profile
) -> None:
    limits = profile.limits
    assert limits.first_prompt_ceiling == 13_934
    text = "x" * int(14_800 * limits.guard_chars_per_token)
    model, executor = FakeModel(harness, [R01_SQL]), FakeExecutor(harness, ((GOLD,),))
    receipt = run_r01(harness, frozen, profile, model, executor, s_provider(text=text))
    assert (receipt["outcome"], receipt["overflow"]) == ("overflow", "context")
    assert model.calls == [] and receipt["earliest_failure_stage"] == "context"


def test_guard_sends_a_request_that_only_the_conservative_ratio_would_refuse(
    harness: SimpleNamespace, frozen: Path, profile: Profile
) -> None:
    model, executor = FakeModel(harness, [R01_SQL]), FakeExecutor(harness, ((GOLD,),))
    receipt = run_r01(harness, frozen, profile, model, executor, s_provider(text="x" * 30_500))
    assert len(model.calls) == 1 and receipt["outcome"] == "correct"
    limits = profile.limits
    lenient = runner_utils.estimate_request_tokens(model.calls[0], 2.334)
    conservative = runner_utils.estimate_request_tokens(model.calls[0], 2.08)
    assert lenient <= limits.first_prompt_ceiling < conservative
    assert receipt["request_estimates"] == [[lenient, conservative]]


def test_a_wider_frozen_ceiling_admits_a_larger_first_prompt(
    harness: SimpleNamespace, frozen: Path, profile: Profile
) -> None:
    model, executor = FakeModel(harness, [R01_SQL]), FakeExecutor(harness, ((GOLD,),))
    wide = profile.limits.model_copy(update={"input_ceiling": 36_000})
    assert wide.first_prompt_ceiling == 33_934
    provider = s_provider(text="x" * 40_000)
    receipt = run_r01(harness, frozen, profile, model, executor, provider, wide)
    assert len(model.calls) == 1 and receipt["outcome"] == "correct"


def test_a_repair_may_use_the_whole_input_ceiling(harness: SimpleNamespace, profile: Profile) -> None:
    limits = profile.limits
    large = [{"role": "user", "content": "x" * int(15_000 * limits.guard_chars_per_token)}]
    lenient = runner_utils.estimate_request_tokens(large, limits.guard_chars_per_token)
    assert limits.first_prompt_ceiling < lenient <= limits.input_ceiling
    model, meter = FakeModel(harness, [R01_SQL]), runner_utils.Meter()
    guarded = runner_utils.GuardedModel(model, meter, limits, PRICES)
    guarded.complete([{"role": "user", "content": "q"}])
    guarded.complete(large)
    assert len(model.calls) == 2 and meter.overflow is None
    first_only = runner_utils.GuardedModel(model, runner_utils.Meter(), limits, PRICES)
    with pytest.raises(runner_utils.TokenOverflow):
        first_only.complete(large)


def test_first_call_usage_under_the_ceiling_is_not_overflow(
    harness: SimpleNamespace, frozen: Path, profile: Profile
) -> None:
    prompt = 15_000
    assert profile.limits.first_prompt_ceiling < prompt <= profile.limits.input_ceiling
    model, executor = FakeModel(harness, [R01_SQL], prompt=prompt), FakeExecutor(harness, ((GOLD,),))
    receipt = run_r01(harness, frozen, profile, model, executor, s_provider())
    assert (receipt["outcome"], receipt["overflow"]) == ("correct", None)


def test_unreported_usage_is_charged_at_the_conservative_estimate_and_output_ceiling(
    harness: SimpleNamespace, frozen: Path, profile: Profile
) -> None:
    model = FakeModel(harness, [R01_SQL], prompt=None, completion=None)
    executor = FakeExecutor(harness, ((GOLD,),))
    receipt = run_r01(harness, frozen, profile, model, executor, s_provider())
    conservative, output = receipt["request_estimates"][0][1], profile.limits.output_ceiling
    assert receipt["usage_reported"] is False
    assert (receipt["prompt_tokens"], receipt["completion_tokens"]) == (conservative, output)
    assert receipt["cost_usd"] == round((conservative * 2.0 + output * 10.0) / 1e6, 6)

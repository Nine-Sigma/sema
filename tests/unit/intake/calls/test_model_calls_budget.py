"""Intake model-call path: one invoke is one call, priced from the table, capped per run."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from sema.intake.calls.ledger_utils import prompt_hash
from sema.llm_client import LLMStageError
from sema.log import logger
from tests.unit.intake.model_calls_env import (
    INPUT_PER_1K,
    MODEL,
    OUTPUT_PER_1K,
    PROMPT,
    STAGE,
    Choice_v1,
    WiredRun,
    choice,
    fake_client,
    intake_config,
    numbered_prompt,
    record_in,
    stats,
    wired_run,
)

pytestmark = pytest.mark.unit

BUDGET_REASON = "budget_exceeded"
STOP_STAGE = "review"
STOP_STATUS = "waiting_review"
FAILED_CALL_KIND = "call_failed"


def _cost(prompt_tokens: int, completion_tokens: int) -> float:
    return prompt_tokens / 1000 * INPUT_PER_1K + completion_tokens / 1000 * OUTPUT_PER_1K


def _stage_error() -> LLMStageError:
    return LLMStageError(
        table_ref="",
        stage_name=STAGE,
        step_errors=[
            ("structured_output", ValueError("schema rejected by provider")),
            ("plain_invoke", ValueError("response was not valid JSON")),
        ],
    )


@pytest.fixture
def warnings() -> Iterator[list[str]]:
    lines: list[str] = []
    sink_id = logger.add(lambda message: lines.append(str(message)), level="WARNING")
    yield lines
    logger.remove(sink_id)


def test_tokens_and_cost_are_counted_once_per_invoke(tmp_path: Path) -> None:
    run = wired_run(tmp_path, fake_client(per_call_stats=stats(1000, 500)))

    run.calls.call(STAGE, PROMPT, Choice_v1)
    row = run.row()

    assert (row.model_calls, row.tokens) == (1, 1500)
    assert row.cost == pytest.approx(_cost(1000, 500))
    assert run.call_record(prompt_hash(MODEL, PROMPT, "Choice_v1"))["tokens"] == 1500


def test_counters_sum_over_invokes(tmp_path: Path) -> None:
    run = wired_run(tmp_path, fake_client(per_call_stats=stats(200, 100)))

    for index in range(3):
        run.calls.call(STAGE, numbered_prompt(index), Choice_v1)
    row = run.row()

    assert (row.model_calls, row.tokens) == (3, 900)
    assert row.cost == pytest.approx(3 * _cost(200, 100))


def test_a_model_missing_from_the_price_table_costs_nothing_and_warns(
    tmp_path: Path, warnings: list[str]
) -> None:
    run = wired_run(tmp_path, fake_client(), intake_config(prices={}))

    run.calls.call(STAGE, PROMPT, Choice_v1)
    row = run.row()

    assert (row.model_calls, row.tokens) == (1, 1500)
    assert row.cost == pytest.approx(0.0)
    assert any(MODEL in line for line in warnings)


def _assert_stopped_for_budget(run: WiredRun, calls: int) -> None:
    row = run.row()
    assert row.model_calls == calls
    assert (row.stage, row.status, row.reason) == (STOP_STAGE, STOP_STATUS, BUDGET_REASON)


def test_the_call_cap_stops_the_next_call(tmp_path: Path) -> None:
    from sema.intake.calls.model_calls import BudgetExceededError

    run = wired_run(tmp_path, fake_client(), intake_config(max_calls=3))
    for index in range(3):
        run.calls.call(STAGE, numbered_prompt(index), Choice_v1)

    with pytest.raises(BudgetExceededError):
        run.calls.call(STAGE, numbered_prompt(3), Choice_v1)

    assert run.client.invoke.call_count == 3
    assert len(run.call_files()) == 3
    _assert_stopped_for_budget(run, calls=3)


def test_the_cost_cap_stops_the_next_call(tmp_path: Path) -> None:
    from sema.intake.calls.model_calls import BudgetExceededError

    per_call = stats(4000, 1000)
    assert _cost(4000, 1000) > 0.01
    config = intake_config(max_cost=0.01)
    run = wired_run(tmp_path, fake_client(per_call_stats=per_call), config)
    run.calls.call(STAGE, numbered_prompt(0), Choice_v1)

    with pytest.raises(BudgetExceededError):
        run.calls.call(STAGE, numbered_prompt(1), Choice_v1)

    assert run.client.invoke.call_count == 1
    _assert_stopped_for_budget(run, calls=1)


def test_a_replay_is_served_after_the_cap_is_reached(tmp_path: Path) -> None:
    run = wired_run(tmp_path, fake_client([choice("b")]), intake_config(max_calls=1))
    run.calls.call(STAGE, PROMPT, Choice_v1)

    assert run.calls.call(STAGE, PROMPT, Choice_v1) == choice("b")
    run.client.invoke.assert_called_once()
    row = run.row()
    assert (row.status, row.reason) == ("running", None)


def test_replays_are_free(tmp_path: Path) -> None:
    run = wired_run(tmp_path, fake_client(), intake_config(max_calls=1))
    prompts = [numbered_prompt(index) for index in range(10)]
    for prompt in prompts:
        record_in(run.ledger.directory, prompt, choice("b"))

    results = [run.calls.call(STAGE, prompt, Choice_v1) for prompt in prompts]

    assert results == [choice("b")] * 10
    run.client.invoke.assert_not_called()
    row = run.row()
    assert (row.model_calls, row.tokens, row.cost) == (0, 0, 0.0)
    assert len(run.events("replay")) == 10


def test_a_stage_error_is_recorded_with_its_step_errors_and_counts_one_call(
    tmp_path: Path,
) -> None:
    error = _stage_error()
    run = wired_run(tmp_path, fake_client([error], per_call_stats=stats(300, 0)))

    with pytest.raises(LLMStageError) as raised:
        run.calls.call(STAGE, PROMPT, Choice_v1)

    assert raised.value is error
    run.client.invoke.assert_called_once()
    row = run.row()
    assert (row.model_calls, row.tokens) == (1, 300)
    (failed,) = run.events(FAILED_CALL_KIND)
    assert failed["stage"] == STAGE
    assert failed["payload"]["prompt_hash"] == prompt_hash(MODEL, PROMPT, "Choice_v1")
    assert failed["payload"]["step_errors"] == [
        {"step": "structured_output", "error": "schema rejected by provider"},
        {"step": "plain_invoke", "error": "response was not valid JSON"},
    ]
    assert run.call_files() == []


def test_a_retry_after_a_stage_error_reaches_the_provider_again(tmp_path: Path) -> None:
    run = wired_run(tmp_path, fake_client([_stage_error(), choice("a")]))
    with pytest.raises(LLMStageError):
        run.calls.call(STAGE, PROMPT, Choice_v1)

    assert run.calls.call(STAGE, PROMPT, Choice_v1) == choice("a")
    assert run.client.invoke.call_count == 2
    assert run.row().model_calls == 2


def test_a_failed_call_counts_toward_the_call_cap(tmp_path: Path) -> None:
    from sema.intake.calls.model_calls import BudgetExceededError

    run = wired_run(tmp_path, fake_client([_stage_error()]), intake_config(max_calls=1))
    with pytest.raises(LLMStageError):
        run.calls.call(STAGE, PROMPT, Choice_v1)

    with pytest.raises(BudgetExceededError):
        run.calls.call(STAGE, numbered_prompt(1), Choice_v1)

    run.client.invoke.assert_called_once()
    _assert_stopped_for_budget(run, calls=1)

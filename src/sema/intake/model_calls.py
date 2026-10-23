"""The one path every intake model call takes: replay, else one capped, counted invoke.

A call is keyed by `prompt_hash(model, prompt, output model name)`. The run ledger is
checked first, then the read-only recordings directory; a recording hit is copied into
the run ledger so the run stays self-contained. Only a miss on both reaches the provider,
after the prompt is checked against ``MAX_PROMPT_TOKENS`` and the registry's call and cost
counters against the configured caps.
"""

from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel

from sema.cli_factories import _get_llm
from sema.intake.ledger import REPLAY_KIND, RunLedger
from sema.intake.ledger_utils import prompt_hash
from sema.intake.model_calls_utils import (
    call_cost,
    call_payload,
    prompt_tokens,
    read_recording,
    recorded_agreement,
    step_errors,
    total_tokens,
)
from sema.intake.registry import Registry
from sema.llm_client import InvocationStats, LLMClient, LLMStageError
from sema.log import logger
from sema.models.config import IntakeModelConfig, LLMConfig

T = TypeVar("T", bound=BaseModel)
IntakeClient = LLMClient

CALL_KIND = "call"
CALL_FAILED_KIND = "call_failed"
CALL_REFUSED_KIND = "call_refused"
# Model-agnostic policy, below common context windows; never derived from a fixture.
MAX_PROMPT_TOKENS = 100_000
BUDGET_STOP_STAGE = "review"
BUDGET_STOP_STATUS = "waiting_review"
BUDGET_REASON = "budget_exceeded"


class IntakeProviderNotSetError(RuntimeError):
    """Intake has no provider of its own; it never falls back to the build default."""


class BudgetExceededError(LLMStageError):
    """The run reached its call or cost cap; the run stops at Review.

    A refused call fails like an exhausted one, so every caller that tolerates a failed
    call leaves its entries open, and a recomputation is refused the same way.
    """

    def __init__(self, run_id: str, stage: str) -> None:
        super().__init__(run_id, stage, [])
        self.args = (f"run {run_id} reached its model-call budget at stage {stage}",)


class PromptBudgetError(LLMStageError):
    """A prompt over ``MAX_PROMPT_TOKENS`` after compaction; no provider call is made."""

    def __init__(self, stage: str, tokens: int) -> None:
        super().__init__("", stage, [])
        message = f"the prompt of {tokens} tokens exceeds the model-call budget of"
        self.args = (f"{message} {MAX_PROMPT_TOKENS} tokens",)


def fits(prompt: str, output_model: type[BaseModel]) -> bool:
    return prompt_tokens(prompt, output_model) <= MAX_PROMPT_TOKENS


def build_model_calls(
    config: IntakeModelConfig, ledger: RunLedger, registry: Registry, run_id: str
) -> ModelCalls:
    return ModelCalls(build_intake_client(config), ledger, registry, run_id, config)


def build_intake_client(config: IntakeModelConfig) -> IntakeClient:
    """The provider client the intake config names; intake never uses the build default."""
    if config.provider is None:
        raise IntakeProviderNotSetError(
            "set INTAKE_MODEL_PROVIDER for intake; the build pipeline's provider is not used"
        )
    llm = _get_llm(
        LLMConfig(
            provider=config.provider,
            model=config.model,
            api_key=config.api_key,
            base_url=config.base_url,
            use_structured_output=config.use_structured_output,
            request_timeout=config.request_timeout,
        )
    )
    return LLMClient(llm, use_structured_output=config.use_structured_output)


def model_calls_for(
    config: IntakeModelConfig,
    ledger: RunLedger,
    registry: Registry,
    run_id: str,
    client: IntakeClient | None,
) -> ModelCalls:
    """The call path over a given client, else over one built from the intake config."""
    if client is None:
        return build_model_calls(config, ledger, registry, run_id)
    return ModelCalls(client, ledger, registry, run_id, config)


class ModelCalls:
    def __init__(
        self,
        client: LLMClient,
        ledger: RunLedger,
        registry: Registry,
        run_id: str,
        config: IntakeModelConfig,
    ) -> None:
        self.client = client
        self._ledger = ledger
        self._registry = registry
        self._run_id = run_id
        self._config = config
        self._warned_unpriced = False
        self.refused = False
        # A replay follows its recording, whatever the environment says (S5 decision 17).
        recordings = config.recordings_dir
        self.agreement = (
            config.agreement if recordings is None else recorded_agreement(recordings)
        )

    def call_key(self, prompt: str, output_model: type[BaseModel]) -> str:
        return prompt_hash(self._config.model, prompt, output_model.__name__)

    def call(self, stage: str, prompt: str, output_model: type[T]) -> T:
        schema_name = output_model.__name__
        recorded = self._replay(stage, prompt, schema_name)
        if recorded is not None:
            return output_model.model_validate(recorded)
        self._check_size(stage, prompt, output_model)
        self._check_budget(stage)
        return self._invoke(stage, prompt, output_model)

    def _replay(self, stage: str, prompt: str, schema_name: str) -> dict[str, object] | None:
        model = self._config.model
        recorded = self._ledger.replay(model, prompt, schema_name, stage=stage)
        if recorded is not None:
            return recorded
        key = prompt_hash(model, prompt, schema_name)
        recording = read_recording(self._config.recordings_dir, key)
        if recording is None:
            return None
        output = recording["output"]
        self._ledger.record_call(model, prompt, schema_name, output, recording["tokens"])
        self._ledger.event(stage, REPLAY_KIND, call_payload(key, model, schema_name))
        return dict(output)

    def _check_size(self, stage: str, prompt: str, output_model: type[BaseModel]) -> None:
        tokens = prompt_tokens(prompt, output_model)
        if tokens <= MAX_PROMPT_TOKENS:
            return
        key = self.call_key(prompt, output_model)
        payload = call_payload(key, self._config.model, output_model.__name__)
        self._ledger.event(stage, CALL_REFUSED_KIND, {**payload, "prompt_tokens": tokens})
        raise PromptBudgetError(stage, tokens)

    def _check_budget(self, stage: str) -> None:
        row = self._registry.get(self._run_id)
        config = self._config
        if row.model_calls < config.max_calls and row.cost < config.max_cost:
            return
        logger.warning(
            f"Run {self._run_id} stopped at {stage}: {row.model_calls} calls "
            f"(cap {config.max_calls}), cost {row.cost:.4f} (cap {config.max_cost})"
        )
        self._registry.advance(
            self._run_id, BUDGET_STOP_STAGE, BUDGET_STOP_STATUS, reason=BUDGET_REASON
        )
        self.refused = True
        raise BudgetExceededError(self._run_id, stage)

    def _invoke(self, stage: str, prompt: str, output_model: type[T]) -> T:
        model = self._config.model
        schema_name = output_model.__name__
        key = self.call_key(prompt, output_model)
        try:
            output = self.client.invoke(prompt, output_model, stage_name=stage)
        except LLMStageError as error:
            self._count(self.client.last_stats)
            payload = {**call_payload(key, model, schema_name), "step_errors": step_errors(error)}
            self._ledger.event(stage, CALL_FAILED_KIND, payload)
            raise
        tokens = self._count(self.client.last_stats)
        self._ledger.record_call(
            model, prompt, schema_name, output.model_dump(mode="json"), tokens
        )
        payload = {**call_payload(key, model, schema_name), "path": self.client.last_path}
        self._ledger.event(stage, CALL_KIND, payload)
        return output

    def _count(self, stats: InvocationStats) -> int:
        tokens = total_tokens(stats)
        self._registry.count_call(self._run_id, tokens, self._cost(stats))
        return tokens

    def _cost(self, stats: InvocationStats) -> float:
        price = self._config.prices.get(self._config.model)
        if price is not None:
            return call_cost(stats, price)
        if not self._warned_unpriced:
            logger.warning(
                f"No price for model {self._config.model}; its calls count at cost 0"
            )
            self._warned_unpriced = True
        return 0.0

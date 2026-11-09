"""Adapters between the frozen Timbr harness and the benchmark runner.

The harness package is imported from the frozen copy at run time and never copied here.
Gold never reaches the harness: questions are rebuilt with gold fields emptied, and only
the typed scorer reads gold. Arm S reuses the arm A dialect and validation path.
"""

import argparse
import dataclasses
import hashlib
import importlib
import json
import math
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Protocol

from showcase.ontology_query_benchmark.routing import require_mcp_tool

HARNESS_PACKAGE = "timbr_demographics_poc"
HARNESS_MODULES = ("benchmark_models", "benchmark_workflow", "benchmark_clients",
                   "benchmark_config_utils", "benchmark_context_utils",
                   "benchmark_kg_slice_utils")  # fmt: skip
W4S_CONFIG = "config/benchmark/cancer_plus_demographics_v2_bench_L_W4S.json"
QUESTION_DIR = "runs/v3_20261021_rw/ab_questions"
QUESTION_FILES = ("regular_live.json", "temporal_live.json", "hierarchy_live.json")
CACHE_DIR = "runs/v3_20261021_rw/ab_w4s/cache"
KG_CACHE = "kg_cancer_plus_demographics_v2.json"
PUBLIC_QUESTION_KEYS = ("id", "class", "question", "result_shape", "needs", "precision",
                        "tolerance", "category_columns")  # fmt: skip
OVERFLOW_FINDING = "overflow"


class SContextLike(Protocol):
    @property
    def text(self) -> str: ...
    @property
    def identity(self) -> str: ...
    @property
    def findings(self) -> tuple[str, ...]: ...


@dataclass(frozen=True)
class Limits:
    """Frozen per run (manifest.json); input_ceiling and repair_reserve come from the CLI."""

    input_ceiling: int = 16_000
    # Two repair turns of 1,033 tokens, the largest Oct 21 repair (fit.calibrate).
    repair_reserve: int = 2_066
    output_ceiling: int = 8_192
    # Highest and lowest chars/prompt_tokens over 186 matched Oct 21 first attempts. The
    # guard refuses only a certain overflow (fewest tokens); the other is recorded.
    guard_chars_per_token: float = 2.334
    conservative_chars_per_token: float = 2.08
    row_limit: int = 1_000
    max_repairs: int = 2

    @property
    def first_prompt_ceiling(self) -> int:
        return self.input_ceiling - self.repair_reserve


@dataclass(frozen=True)
class Prices:
    input_usd_per_1m: float
    output_usd_per_1m: float


@dataclass(frozen=True)
class Knobs:
    budget_tokens: int | None = None
    values_top_k: int | None = None
    description_cap_chars: int | None = None


# Frozen by the coordinator in run-settings.md; Knobs() keeps the Oct 21 W4S values.
BENCHMARK_KNOBS = Knobs(budget_tokens=6_000, values_top_k=5, description_cap_chars=160)


def estimate_tokens_from_chars(chars: int, messages: int, chars_per_token: float) -> int:
    return math.ceil(chars / chars_per_token) + 4 * messages


def add_knob_args(parser: argparse.ArgumentParser) -> None:
    for name, default in zip(("budget-tokens", "values-top-k", "description-cap-chars"),
                             dataclasses.astuple(BENCHMARK_KNOBS), strict=True):  # fmt: skip
        parser.add_argument(f"--{name}", type=int, default=default,
                            help=f"common override for every arm (default {default})")


def knobs_from(args: argparse.Namespace) -> Knobs:
    return Knobs(args.budget_tokens, args.values_top_k, args.description_cap_chars)


def add_limit_args(parser: argparse.ArgumentParser) -> None:
    defaults = Limits()
    parser.add_argument("--input-ceiling", type=int, default=defaults.input_ceiling,
                        help="whole-request input ceiling for every arm")
    parser.add_argument("--repair-reserve", type=int, default=defaults.repair_reserve,
                        help="tokens kept free for two repairs on the first prompt")


def limits_from(args: argparse.Namespace) -> Limits:
    return Limits(input_ceiling=args.input_ceiling, repair_reserve=args.repair_reserve)


def cap_values(values: dict[str, Any], top_k: int) -> dict[str, Any]:
    """The values cache as a smaller ``values_top_k`` would have built it: each profile keeps
    its first ``top_k`` rows (the cache query orders by rows descending, then value)."""
    relations = {
        relation: {**entry, "values": {
            column: {**profile, "values": (profile.get("values") or [])[:top_k]}
            for column, profile in (entry.get("values") or {}).items()
        }}
        for relation, entry in (values.get("relations") or {}).items()
    }  # fmt: skip
    return {**values, "relations": relations}


class TokenOverflow(RuntimeError):
    pass


@dataclass
class Meter:
    """Counters for one slot; created per slot, never shared between slots."""

    generation_calls: int = 0
    sql_executions: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    usage_reported: bool = True
    cost_usd: float = 0.0
    overflow: str | None = None
    context_identity: str | None = None
    failed_stage: str | None = None
    findings: tuple[str, ...] = ()
    last_execution: Any = None
    request_estimates: list[list[int]] = field(default_factory=list)


def load_harness(frozen: Path) -> SimpleNamespace:
    src = str(frozen / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    modules = {n: importlib.import_module(f"{HARNESS_PACKAGE}.{n}") for n in HARNESS_MODULES}
    loaded_from = Path(modules["benchmark_workflow"].__file__ or "").resolve()
    if not loaded_from.is_relative_to(Path(src).resolve()):
        raise RuntimeError(f"harness imported from {loaded_from}, not the frozen copy {src}")
    return SimpleNamespace(
        models=modules["benchmark_models"], workflow=modules["benchmark_workflow"],
        clients=modules["benchmark_clients"], config=modules["benchmark_config_utils"],
        context=modules["benchmark_context_utils"], kg=modules["benchmark_kg_slice_utils"],
    )  # fmt: skip


def question_paths(frozen: Path) -> list[Path]:
    return [frozen / QUESTION_DIR / name for name in QUESTION_FILES]


def _question_entries(frozen: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for path in question_paths(frozen):
        raw = json.loads(path.read_text())
        for entry in raw["questions"]:
            entries.append({k: entry[k] for k in PUBLIC_QUESTION_KEYS if k in entry})
    return entries


def question_ids(frozen: Path) -> list[str]:
    return [str(entry["id"]) for entry in _question_entries(frozen)]


def public_questions(
    harness: SimpleNamespace, frozen: Path, ids: tuple[str, ...] = ()
) -> list[Any]:
    """Harness Question records in file order with gold_answer None and no gold SQL."""
    chosen = [e for e in _question_entries(frozen) if not ids or e["id"] in ids]
    missing = set(ids) - {e["id"] for e in chosen}
    if missing:
        raise ValueError(f"unknown question ids: {', '.join(sorted(missing))}")
    precision = harness.config.precision_and_tolerance
    return [
        harness.models.Question(
            id=str(e["id"]), question_class=str(e["class"]), question=str(e["question"]),
            result_shape=e["result_shape"], gold_answer=None, gold_sql="",
            needs=tuple(e.get("needs") or ()),
            precision=precision(e.get("precision"), e.get("tolerance"))[0],
            tolerance=precision(e.get("precision"), e.get("tolerance"))[1],
            category_columns=tuple(e.get("category_columns") or ()),
        )  # fmt: skip
        for e in chosen
    ]


class SArmContext:
    """Adapt the S provider's SContext to the harness ArmContext; keep the last one."""

    def __init__(self, harness: SimpleNamespace, provider: Callable[[str], SContextLike],
                 budget: int) -> None:  # fmt: skip
        self.harness, self.provider, self.budget = harness, provider, budget
        self.last: SContextLike | None = None

    def __call__(self, question: str) -> Any:
        supplied = self.provider(question)
        self.last = supplied
        overflow = any(f.split(":")[0] == OVERFLOW_FINDING for f in supplied.findings)
        tokens = self.harness.context.estimate_tokens(supplied.text)
        return self.harness.models.ArmContext(
            "S", self.harness.context.ARM_A_DIALECT, supplied.text, tokens, self.budget, overflow
        )


def arm_runtimes(harness: SimpleNamespace, frozen: Path, executors: Mapping[str, Any],
                 s_provider: Callable[[str], SContextLike] | None,
                 limits: Limits = Limits(), knobs: Knobs = Knobs()) -> dict[str, Any]:  # fmt: skip
    """One ArmRuntime per arm in ``executors``, all with W4S rules and matched limits."""
    config = _with_knobs(harness.config.load_bench_config(frozen / W4S_CONFIG), knobs)
    policy, budget = config.policy, config.budget_tokens
    contexts: dict[str, Callable[[str], Any]] = {}
    known: dict[str, tuple[str, ...]] = {"A": (), "S": ()}
    if "A" in executors:
        schema = harness.config.load_json(frozen / CACHE_DIR / "schema.json")
        values = harness.config.load_json(frozen / CACHE_DIR / "values.json")
        if knobs.values_top_k is not None:
            values = cap_values(values, knobs.values_top_k)
        contexts["A"] = lambda _q: harness.context.arm_a_context(
            schema, config.relations, policy, budget, values
        )
    if "B" in executors:
        kg = harness.config.load_json(frozen / CACHE_DIR / KG_CACHE)
        contexts["B"] = lambda q: harness.kg.arm_b_context(
            kg, q, row_concepts=config.row_concepts, always_concepts=config.always_concepts,
            policy=policy, budget=budget, description_cap=config.description_cap_chars,
        )  # fmt: skip
        known["B"] = tuple(sorted(harness.kg.kg_sections(kg, policy)[0]))
    if "S" in executors:
        if s_provider is None:
            raise ValueError("arm S needs an S context provider")
        contexts["S"] = SArmContext(harness, s_provider, budget)
    return {
        arm: harness.workflow.ArmRuntime(
            arm, contexts[arm], executor, policy, known[arm], limits.row_limit,
            limits.max_repairs, config.near_miss_pct, conventions=config.conventions,
            sql_rules=config.sql_rules, sql_checks=config.sql_checks,
        )  # fmt: skip
        for arm, executor in executors.items()
    }


def _with_knobs(config: Any, knobs: Knobs) -> Any:
    if knobs.values_top_k is not None and not 0 <= knobs.values_top_k <= config.values_top_k:
        raise ValueError(f"values_top_k {knobs.values_top_k} is outside 0..{config.values_top_k}, "
                         "the frozen values cache")  # fmt: skip
    changes = {"budget_tokens": knobs.budget_tokens,
               "description_cap_chars": knobs.description_cap_chars}  # fmt: skip
    return dataclasses.replace(config, **{k: v for k, v in changes.items() if v is not None})


def estimate_request_tokens(messages: list[dict[str, Any]], chars_per_token: float) -> int:
    chars = sum(len(str(m.get("content") or "")) for m in messages)
    return estimate_tokens_from_chars(chars, len(messages), chars_per_token)


class GuardedModel:
    """Enforce the whole-request ceilings around one model client and meter its use."""

    def __init__(self, inner: Any, meter: Meter, limits: Limits, prices: Prices) -> None:
        self.inner, self.meter, self.limits, self.prices = inner, meter, limits, prices

    def complete(self, messages: list[dict[str, Any]]) -> Any:
        lenient = estimate_request_tokens(messages, self.limits.guard_chars_per_token)
        estimate = estimate_request_tokens(messages, self.limits.conservative_chars_per_token)
        self.meter.request_estimates.append([lenient, estimate])
        first = self.meter.generation_calls == 0
        ceiling = self.limits.first_prompt_ceiling if first else self.limits.input_ceiling
        if lenient > ceiling:
            self.meter.overflow = "context" if first else "pre_call"
            raise TokenOverflow(f"request certainly exceeds {ceiling}: estimate {lenient}")
        self.meter.generation_calls += 1
        generation = self.inner.complete(messages)
        prompt, completion = generation.prompt_tokens, generation.completion_tokens
        if prompt is None or completion is None:
            self.meter.usage_reported = False
        # shortcut: missing usage is charged at the estimate and the output ceiling.
        prompt_n = estimate if prompt is None else prompt
        completion_n = self.limits.output_ceiling if completion is None else completion
        self.meter.prompt_tokens += prompt_n
        self.meter.completion_tokens += completion_n
        self.meter.cost_usd += (prompt_n * self.prices.input_usd_per_1m
                                + completion_n * self.prices.output_usd_per_1m) / 1e6  # fmt: skip
        if prompt_n > self.limits.input_ceiling or completion_n > self.limits.output_ceiling:
            self.meter.overflow = "post_call"
            raise TokenOverflow(f"endpoint usage {prompt_n}/{completion_n} exceeds a ceiling")
        return generation


class MeteredExecutor:
    def __init__(self, inner: Any, meter: Meter) -> None:
        self.inner, self.meter = inner, meter

    def execute(self, sql: str, row_limit: int) -> Any:
        self.meter.sql_executions += 1
        self.meter.failed_stage = "execution"
        execution = self.inner.execute(sql, row_limit)
        self.meter.failed_stage = None
        self.meter.last_execution = execution
        return execution


def metered_context(context: Callable[[str], Any], meter: Meter) -> Callable[[str], Any]:
    def build(question: str) -> Any:
        meter.failed_stage = "context"
        built = context(question)
        meter.failed_stage = None
        supplied: SContextLike | None = getattr(context, "last", None)
        if supplied is None:
            meter.context_identity = hashlib.sha256(built.text.encode()).hexdigest()
        else:
            meter.context_identity, meter.findings = supplied.identity, tuple(supplied.findings)
        return built

    return build


def load_s_provider(spec: str) -> Callable[[str], SContextLike]:
    module, _, function = spec.partition(":")
    if not module or not function:
        raise ValueError(f"S context provider must be module:function, got {spec!r}")
    provider: Callable[[str], SContextLike] = getattr(importlib.import_module(module), function)
    return provider


def guard_mcp_tools(executor: Any) -> Any:
    """Route every Timbr tools/call through the arm B allowlist."""
    call_tool = executor.call_tool

    def guarded(tool: str, arguments: dict[str, Any]) -> Any:
        return call_tool(require_mcp_tool(tool), arguments)

    executor.call_tool = guarded
    return executor


def read_env_keys(path: Path, keys: tuple[str, ...]) -> dict[str, str]:
    """Read the named KEY=VALUE lines from an env file without exporting anything."""
    values: dict[str, str] = {}
    for line in Path(path).read_text().splitlines():
        name, sep, value = line.strip().removeprefix("export ").partition("=")
        if sep and name.strip() in keys:
            values[name.strip()] = value.split(" #", 1)[0].strip().strip("'\"")
    return values


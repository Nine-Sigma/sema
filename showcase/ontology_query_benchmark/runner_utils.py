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

from showcase.ontology_query_benchmark.profile import Harness, Knobs, Limits, Timbr
from showcase.ontology_query_benchmark.routing import require_mcp_tool

HARNESS_MODULES = ("benchmark_models", "benchmark_workflow", "benchmark_clients",
                   "benchmark_config_utils", "benchmark_context_utils",
                   "benchmark_kg_slice_utils")  # fmt: skip
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
class Prices:
    input_usd_per_1m: float
    output_usd_per_1m: float


def estimate_tokens_from_chars(chars: int, messages: int, chars_per_token: float) -> int:
    return math.ceil(chars / chars_per_token) + 4 * messages


KNOB_ARGS = ("budget_tokens", "values_top_k", "description_cap_chars")
LIMIT_ARGS = ("input_ceiling", "repair_reserve")


def add_knob_args(parser: argparse.ArgumentParser) -> None:
    for name in KNOB_ARGS:
        parser.add_argument(f"--{name.replace('_', '-')}", type=int,
                            help="common override for every arm (default: the profile knobs)")


def knobs_from(args: argparse.Namespace, profile_knobs: Knobs) -> Knobs:
    return profile_knobs.model_copy(update=_given(args, KNOB_ARGS))


def add_limit_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-ceiling", type=int,
                        help="whole-request input ceiling for every arm (default: profile)")
    parser.add_argument("--repair-reserve", type=int,
                        help="tokens kept free for two repairs (default: profile)")


def limits_from(args: argparse.Namespace, profile_limits: Limits) -> Limits:
    return profile_limits.model_copy(update=_given(args, LIMIT_ARGS))


def _given(args: argparse.Namespace, names: tuple[str, ...]) -> dict[str, int]:
    return {name: getattr(args, name) for name in names if getattr(args, name) is not None}


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


def load_harness(frozen: Path, layout: Harness) -> SimpleNamespace:
    src = str(frozen / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    modules = {n: importlib.import_module(f"{layout.package}.{n}") for n in HARNESS_MODULES}
    loaded_from = Path(modules["benchmark_workflow"].__file__ or "").resolve()
    if not loaded_from.is_relative_to(Path(src).resolve()):
        raise RuntimeError(f"harness imported from {loaded_from}, not the frozen copy {src}")
    return SimpleNamespace(
        models=modules["benchmark_models"], workflow=modules["benchmark_workflow"],
        clients=modules["benchmark_clients"], config=modules["benchmark_config_utils"],
        context=modules["benchmark_context_utils"], kg=modules["benchmark_kg_slice_utils"],
    )  # fmt: skip


def question_paths(frozen: Path, layout: Harness) -> list[Path]:
    return [frozen / layout.question_dir / name for name in layout.question_files]


def _question_entries(frozen: Path, layout: Harness) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for path in question_paths(frozen, layout):
        raw = json.loads(path.read_text())
        for entry in raw["questions"]:
            entries.append({k: entry[k] for k in PUBLIC_QUESTION_KEYS if k in entry})
    return entries


def question_ids(frozen: Path, layout: Harness) -> list[str]:
    return [str(entry["id"]) for entry in _question_entries(frozen, layout)]


def public_questions(
    harness: SimpleNamespace, frozen: Path, layout: Harness, ids: tuple[str, ...] = ()
) -> list[Any]:
    """Harness Question records in file order with gold_answer None and no gold SQL."""
    chosen = [e for e in _question_entries(frozen, layout) if not ids or e["id"] in ids]
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


def arm_runtimes(harness: SimpleNamespace, frozen: Path, layout: Harness,
                 executors: Mapping[str, Any], s_provider: Callable[[str], SContextLike] | None,
                 limits: Limits, knobs: Knobs = Knobs()) -> dict[str, Any]:  # fmt: skip
    """One ArmRuntime per arm in ``executors``, all with the config rules and matched limits."""
    config = _with_knobs(harness.config.load_bench_config(frozen / layout.config), knobs)
    policy, budget = config.policy, config.budget_tokens
    contexts: dict[str, Callable[[str], Any]] = {}
    known: dict[str, tuple[str, ...]] = {"A": (), "S": ()}
    if "A" in executors:
        schema = harness.config.load_json(frozen / layout.cache_dir / "schema.json")
        values = harness.config.load_json(frozen / layout.cache_dir / "values.json")
        if knobs.values_top_k is not None:
            values = cap_values(values, knobs.values_top_k)
        contexts["A"] = lambda _q: harness.context.arm_a_context(
            schema, config.relations, policy, budget, values
        )
    if "B" in executors:
        kg = harness.config.load_json(frozen / layout.cache_dir / layout.kg_cache)
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


def guard_mcp_tools(executor: Any, timbr: Timbr) -> Any:
    """Route every MCP tools/call through the profile's arm B allowlist."""
    call_tool = executor.call_tool

    def guarded(tool: str, arguments: dict[str, Any]) -> Any:
        return call_tool(require_mcp_tool(tool, timbr), arguments)

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



class SourceVersionError(RuntimeError):
    """A pinned source relation is unreadable or no longer at its pinned version."""


def version_reader(executor: Any) -> Callable[[str], int]:
    """A relation's current Delta version, read through an arm's SQL executor."""

    def read(relation: str) -> int:
        execution = executor.execute(f"DESCRIBE HISTORY {relation} LIMIT 1", 1)
        if not execution.ok or not execution.rows:
            raise SourceVersionError(f"cannot read the version of {relation}: {execution.message}")
        return int(execution.rows[0][execution.columns.index("version")])

    return read


def source_drift(pins: Mapping[str, int], read_version: Callable[[str], int]) -> str | None:
    for relation, pinned in pins.items():
        current = read_version(relation)
        if current != pinned:
            return f"{relation} is at version {current}, pinned at {pinned}"
    return None

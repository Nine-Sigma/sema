"""Adapters between the frozen Timbr harness and the benchmark runner.

The harness package is imported from the frozen copy at run time and never copied here.
Gold never reaches the harness: questions are rebuilt with gold fields emptied, and only
the typed scorer reads gold. Arm S reuses the arm A dialect and validation path.
"""

import hashlib
import importlib
import json
import math
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Protocol

from showcase.partner_timbr_benchmark.routing import (
    SQL_WAREHOUSE_ID,
    TEST_PROFILE,
    TIMBR_MCP_URL,
    TIMBR_ONTOLOGY,
    RoutingError,
    require_chat_endpoint,
    require_mcp_tool,
    require_test_host,
    require_timbr_mcp_url,
)

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
CHAT_ENDPOINT = "databricks-claude-sonnet-5"
MODEL_TIMEOUT_S = 300.0


class SContextLike(Protocol):
    @property
    def text(self) -> str: ...
    @property
    def identity(self) -> str: ...
    @property
    def findings(self) -> tuple[str, ...]: ...


@dataclass(frozen=True)
class Limits:
    input_ceiling: int = 16_000
    output_ceiling: int = 8_192
    # Measured on the Oct 21 receipts: endpoint tokens were 1.77-2.09x the chars/4 estimate.
    chars_per_token: float = 1.8
    row_limit: int = 1_000
    max_repairs: int = 2


@dataclass(frozen=True)
class Prices:
    input_usd_per_1m: float
    output_usd_per_1m: float


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
    findings: tuple[str, ...] = ()
    last_execution: Any = None


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
                 limits: Limits = Limits()) -> dict[str, Any]:  # fmt: skip
    """One ArmRuntime per arm in ``executors``, all with W4S rules and matched limits."""
    config = harness.config.load_bench_config(frozen / W4S_CONFIG)
    policy, budget = config.policy, config.budget_tokens
    contexts: dict[str, Callable[[str], Any]] = {}
    known: dict[str, tuple[str, ...]] = {"A": (), "S": ()}
    if "A" in executors:
        schema = harness.config.load_json(frozen / CACHE_DIR / "schema.json")
        values = harness.config.load_json(frozen / CACHE_DIR / "values.json")
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


def estimate_request_tokens(messages: list[dict[str, Any]], chars_per_token: float) -> int:
    chars = sum(len(str(m.get("content") or "")) for m in messages)
    return math.ceil(chars / chars_per_token) + 4 * len(messages)


class GuardedModel:
    """Enforce the whole-request ceilings around one model client and meter its use."""

    def __init__(self, inner: Any, meter: Meter, limits: Limits, prices: Prices) -> None:
        self.inner, self.meter, self.limits, self.prices = inner, meter, limits, prices

    def complete(self, messages: list[dict[str, Any]]) -> Any:
        estimate = estimate_request_tokens(messages, self.limits.chars_per_token)
        if estimate > self.limits.input_ceiling:
            self.meter.overflow = "pre_call"
            raise TokenOverflow(f"request estimate {estimate} > {self.limits.input_ceiling}")
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
        execution = self.inner.execute(sql, row_limit)
        self.meter.last_execution = execution
        return execution


def metered_context(context: Callable[[str], Any], meter: Meter) -> Callable[[str], Any]:
    def build(question: str) -> Any:
        built = context(question)
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
            values[name.strip()] = value.strip().strip("'\"")
    return values


def live_clients(harness: SimpleNamespace, arms: list[str], timbr_env: Path | None,
                 limits: Limits) -> tuple[Any, dict[str, Any]]:  # fmt: skip
    """The TEST serving model and one executor per arm; any routing mismatch raises."""
    from databricks import sql
    from databricks.sdk.core import Config

    profile = Config(profile=TEST_PROFILE)
    host, token = require_test_host(profile.host or ""), profile.token
    if not token:
        raise RoutingError(f"profile {TEST_PROFILE} has no token")
    clients = harness.clients
    model = clients.DatabricksServingClient(
        host, token, require_chat_endpoint(CHAT_ENDPOINT), max_tokens=limits.output_ceiling,
        temperature=None, timeout=MODEL_TIMEOUT_S,
    )  # fmt: skip

    def connect() -> Any:
        return sql.connect(server_hostname=host.removeprefix("https://"),
                           http_path=f"/sql/1.0/warehouses/{SQL_WAREHOUSE_ID}",
                           access_token=token)  # fmt: skip

    executors: dict[str, Any] = {arm: clients.DatabricksSqlExecutor(connect)
                                 for arm in arms if arm in ("A", "S")}  # fmt: skip
    if "B" in arms:
        if timbr_env is None:
            raise ValueError("arm B needs --timbr-env (the original POC env file)")
        env = read_env_keys(timbr_env, ("TIMBR_TOKEN", "TIMBR_CA_BUNDLE", "TIMBR_VERIFY_SSL"))
        if not env.get("TIMBR_TOKEN"):
            raise ValueError(f"TIMBR_TOKEN missing in {timbr_env}")
        verify: bool | str = env.get("TIMBR_CA_BUNDLE") or env.get("TIMBR_VERIFY_SSL") != "false"
        executor = clients.TimbrMcpExecutor(require_timbr_mcp_url(TIMBR_MCP_URL),
                                            env["TIMBR_TOKEN"], verify, TIMBR_ONTOLOGY,
                                            timeout=MODEL_TIMEOUT_S)  # fmt: skip
        executors["B"] = guard_mcp_tools(executor)
    return model, executors

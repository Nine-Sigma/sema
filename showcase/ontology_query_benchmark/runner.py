"""Run the A/B/S benchmark matrix on a frozen harness with typed scoring.

python -m showcase.ontology_query_benchmark.runner --profile <profile.yaml>
    --frozen <closure dir> --run-dir <dir>
    [--arms A B S] [--reps 3] [--ids ID1,ID2] [--dry-run]
    [--timbr-env <ontology engine env file>]
    [--neo4j-uri U --s-manifest M --s-projection P --s-projection-receipt R]  # arm S
    [--s-concept-spec C]   # arm S, default: the profile's concept spec
    [--settings settings.json]   # {"prices": {...usd_per_1m}, "max_cost_usd": 60, "max_wall_s": 10800}
    [--budget-tokens N] [--values-top-k K] [--description-cap-chars C]  # default: profile
    [--input-ceiling N] [--repair-reserve N]   # default: profile; frozen in manifest.json

Slots run in a fixed order (rep, question, arm). Each slot appends one receipt to
receipts.jsonl; a slot whose latest receipt has an outcome other than ``unrun`` is never
run again. An exception from a context provider or executor becomes a ``runner_error``
receipt and the run continues. Spend and elapsed time are checked before each slot: when
either reaches its ceiling every remaining slot is written ``unrun`` with the reason, so a
run can overshoot a ceiling by at most one slot (three generations). Each receipt records
its slot's ``wall_s``; elapsed time is the wall time earlier invocations recorded plus the
time since this invocation started.

Every source relation the profile pins must be at its pinned version, read with the
profile's version query through arm A's (else arm S's) SQL executor: a mismatch refuses the run before the first slot, and
a change seen after any slot writes every remaining slot ``unrun`` with ``source_drift``, and
a failed read after a slot (its receipt already written) does the same with
``source_check_failed``.
"""

import argparse
import dataclasses
import json
import os
import sys
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from sema.log import logger
from showcase.ontology_query_benchmark import runner_utils as ru
from showcase.ontology_query_benchmark import scorer
from showcase.ontology_query_benchmark.routing import require_clean_process
from showcase.ontology_query_benchmark.runner_live_utils import (
    add_s_args,
    live_clients,
    open_s_provider,
)
from showcase.ontology_query_benchmark.sema_context import SContextProvider
from showcase.ontology_query_benchmark.profile import Knobs, Limits, load_profile
from showcase.ontology_query_benchmark.runner_utils import Meter, Prices

ARMS = ("A", "B", "S")
STAGES = ("context", "generation", "validation", "execution", "scoring")
GENERATION_ERRORS = frozenset({"model_error", "no_sql"})
EXECUTION_ERRORS = frozenset({"sql_error", "infrastructure", "permission", "timeout"})
OUTCOME_STAGES = {"unsupported": "generation", "refusal": "generation", "wrong": "scoring"}


@dataclass(frozen=True)
class Slot:
    index: int
    arm: str
    question_id: str
    rep: int

    @property
    def slot_id(self) -> str:
        return f"{self.arm}:{self.question_id}:{self.rep}"


@dataclass(frozen=True)
class RunSettings:
    limits: Limits
    prices: Prices
    max_cost_usd: float
    max_wall_s: float


def schedule(ids: Sequence[str], arms: Sequence[str], reps: int) -> tuple[Slot, ...]:
    order = [(rep, qid, arm) for rep in range(1, reps + 1) for qid in ids for arm in arms]
    return tuple(Slot(i, arm, qid, rep) for i, (rep, qid, arm) in enumerate(order))


def run_slot(slot: Slot, question: Any, runtime: Any, model: Any, contract: scorer.Contract,
             harness: SimpleNamespace, settings: RunSettings) -> dict[str, Any]:  # fmt: skip
    meter = Meter()
    metered = dataclasses.replace(
        runtime,
        context=ru.metered_context(runtime.context, meter),
        executor=ru.MeteredExecutor(runtime.executor, meter),
    )
    guarded = ru.GuardedModel(model, meter, settings.limits, settings.prices)
    try:
        record = harness.workflow.run_question(question, metered, guarded, rep=slot.rep)
    except Exception as exc:  # noqa: BLE001 - a failing provider or executor ends only its slot
        logger.warning("{} runner_error: {}", slot.slot_id, type(exc).__name__)
        return _runner_error(slot, meter, harness.workflow.sanitize_message(str(exc)))
    outcome, verdict = _classify(record, meter, contract, settings.limits.row_limit)
    attempts = record["attempts"]
    return {
        **_slot_fields(slot), "outcome": outcome,
        "strict": bool(verdict and verdict.strict and outcome == "correct"),
        "tolerant": bool(verdict and verdict.tolerant and not meter.findings),
        "raw_match": verdict.strict if verdict else None,
        "score_reason": verdict.reason if verdict else None,
        "first_sql": attempts[0].get("sql") if attempts else None, "final_sql": record["sql"],
        "context_identity": meter.context_identity, "findings": list(meter.findings),
        "context_tokens_estimate": record["context_tokens"],
        "generation_calls": meter.generation_calls, "sql_executions": meter.sql_executions,
        "repairs": record["repairs"], "prompt_tokens": meter.prompt_tokens,
        "completion_tokens": meter.completion_tokens, "usage_reported": meter.usage_reported,
        "overflow": meter.overflow or ("context" if outcome == "overflow" else None),
        "cost_usd": round(meter.cost_usd, 6), "latency_s": record["latency_s"],
        "error_class": record["error_class"], "answer": record["answer"],
        "earliest_failure_stage": _earliest_stage(record, meter, outcome),
        "request_estimates": meter.request_estimates,
    }  # fmt: skip


def _runner_error(slot: Slot, meter: Meter, message: str) -> dict[str, Any]:
    return {**_slot_fields(slot), "outcome": "runner_error", "strict": False, "tolerant": False,
            "error_message": message, "earliest_failure_stage": meter.failed_stage or "runner",
            "context_identity": meter.context_identity, "findings": list(meter.findings),
            "generation_calls": meter.generation_calls, "sql_executions": meter.sql_executions,
            "prompt_tokens": meter.prompt_tokens, "completion_tokens": meter.completion_tokens,
            "usage_reported": meter.usage_reported, "overflow": meter.overflow,
            "cost_usd": round(meter.cost_usd, 6),
            "request_estimates": meter.request_estimates}  # fmt: skip


def _slot_fields(slot: Slot) -> dict[str, Any]:
    return {"slot_id": slot.slot_id, "index": slot.index, "arm": slot.arm,
            "question_id": slot.question_id, "rep": slot.rep}  # fmt: skip


def _classify(record: dict[str, Any], meter: Meter, contract: scorer.Contract,
              row_limit: int) -> tuple[str, scorer.Verdict | None]:  # fmt: skip
    if meter.overflow or record["outcome"] == "context_overflow":
        return "overflow", None
    executed = record["outcome"] in ("correct", "wrong") and record["error_class"] is None
    if not executed or meter.last_execution is None:
        return record["outcome"], None
    rows = meter.last_execution.rows
    complete = bool(record["complete"]) and len(rows) < row_limit
    verdict = scorer.score(contract, rows, complete=complete)
    if verdict.strict and meter.findings:
        return "context_incomplete", verdict
    return ("correct" if verdict.strict else "wrong"), verdict


def _earliest_stage(record: dict[str, Any], meter: Meter, outcome: str) -> str | None:
    if outcome == "correct":
        return None
    stages = {"context"} if meter.findings or outcome == "overflow" else set()
    errors = [a.get("error_class") for a in record["attempts"]] + [record["error_class"]]
    for error in filter(None, errors):
        stages.add("generation" if error in GENERATION_ERRORS
                   else "execution" if error in EXECUTION_ERRORS else "validation")  # fmt: skip
    if outcome in OUTCOME_STAGES:
        stages.add(OUTCOME_STAGES[outcome])
    return min(stages, key=STAGES.index) if stages else None


def recorded(receipts: Path) -> tuple[set[str], float, float]:
    """Slot ids whose latest receipt has an outcome, and the spend and wall time so far."""
    latest: dict[str, dict[str, Any]] = {}
    spent = wall = 0.0
    if receipts.is_file():
        for line in receipts.read_text().splitlines():
            entry = json.loads(line)
            latest[entry["slot_id"]] = entry
            spent += float(entry.get("cost_usd") or 0.0)
            wall += float(entry.get("wall_s") or 0.0)
    return {sid for sid, e in latest.items() if e.get("outcome") != "unrun"}, spent, wall


def run(slots: Sequence[Slot], questions: Mapping[str, Any], runtimes: Mapping[str, Any],
        model: Any, contracts: Mapping[str, scorer.Contract], harness: SimpleNamespace,
        settings: RunSettings, receipts: Path, *, pins: Mapping[str, int],
        read_version: Callable[[str], int],
        clock: Callable[[], float] = time.monotonic) -> dict[str, Any]:  # fmt: skip
    drift = ru.source_drift(pins, read_version)
    if drift:
        raise ru.SourceVersionError(f"refusing to start: {drift}")
    done, spent, prior_wall = recorded(receipts)
    start = clock()
    stopped: str | None = None
    source_stop: str | None = None
    outcomes: Counter[str] = Counter()
    receipts.parent.mkdir(parents=True, exist_ok=True)
    with receipts.open("a") as handle:
        for slot in slots:
            if slot.slot_id in done:
                continue
            began = clock()
            stopped = source_stop or _stop_reason(spent, prior_wall + began - start, settings)
            if stopped:
                entry = {**_slot_fields(slot), "outcome": "unrun", "reason": stopped}
            else:
                entry = run_slot(slot, questions[slot.question_id], runtimes[slot.arm], model,
                                 contracts[slot.question_id], harness, settings)  # fmt: skip
                spent += entry["cost_usd"]
                entry["wall_s"] = round(clock() - began, 6)
            outcomes[entry["outcome"]] += 1
            handle.write(json.dumps(entry, sort_keys=True, default=str) + "\n")
            handle.flush()
            logger.info("{} {} spent=${:.4f}", slot.slot_id, entry["outcome"], spent)
            if not stopped:
                source_stop, drift = _source_check(pins, read_version)
                if drift:
                    logger.warning("{} after {}: {}", source_stop, slot.slot_id, drift)
    return {
        "outcomes": dict(outcomes),
        "spent_usd": round(spent, 6),
        "stopped": stopped,
        "source_drift": drift,
    }


def _source_check(
    pins: Mapping[str, int], read_version: Callable[[str], int]
) -> tuple[str | None, str | None]:
    """The stop reason and its detail after a slot: a changed or an unreadable source."""
    try:
        drift = ru.source_drift(pins, read_version)
    except ru.SourceVersionError as exc:
        return "source_check_failed", str(exc)
    return ("source_drift" if drift else None), drift


def _stop_reason(spent: float, elapsed: float, settings: RunSettings) -> str | None:
    if spent >= settings.max_cost_usd:
        return "cost_ceiling"
    if elapsed >= settings.max_wall_s:
        return "time_ceiling"
    return None


def _write_slots(run_dir: Path, slots: Sequence[Slot]) -> None:
    text = "".join(json.dumps(dataclasses.asdict(s) | {"slot_id": s.slot_id}) + "\n" for s in slots)
    path = run_dir / "slots.jsonl"
    if path.is_file() and path.read_text() != text:
        raise ValueError(f"{path} holds a different schedule; use a fresh run dir")
    run_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _write_manifest(
    run_dir: Path, frozen: Path, knobs: Knobs, limits: Limits, pins: Mapping[str, int]
) -> None:
    manifest = {"closure_id": frozen.resolve().name, "knobs": knobs.model_dump(),
                "limits": limits.model_dump(), "source_versions": dict(pins)}  # fmt: skip
    path = run_dir / "manifest.json"
    if path.is_file() and json.loads(path.read_text()) != manifest:
        raise ValueError(
            f"{path} pins a different closure, knobs, limits or source versions; "
            "use a fresh run dir"
        )
    path.write_text(json.dumps(manifest, indent=1, sort_keys=True))


def _parse(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--ids", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--timbr-env", type=Path)
    parser.add_argument("--settings", type=Path)
    ru.add_knob_args(parser)
    ru.add_limit_args(parser)
    add_s_args(parser)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    profile = load_profile(args.profile)
    layout = profile.harness
    wanted = tuple(i.strip() for i in args.ids.split(",") if i.strip())
    ids = [i for i in ru.question_ids(args.frozen, layout) if not wanted or i in wanted]
    if set(wanted) - set(ids):
        raise ValueError(f"unknown question ids: {sorted(set(wanted) - set(ids))}")
    slots = schedule(ids, args.arms, args.reps)
    _write_slots(args.run_dir, slots)
    knobs, limits = ru.knobs_from(args, profile.knobs), ru.limits_from(args, profile.limits)
    _write_manifest(args.run_dir, args.frozen, knobs, limits, profile.source_versions)
    if args.dry_run:
        print(f"{len(slots)} slots written to {args.run_dir / 'slots.jsonl'}; no calls made")
        return 0
    raw = json.loads(args.settings.read_text())
    settings = RunSettings(limits, Prices(**raw["prices"]), float(raw["max_cost_usd"]),
                           float(raw["max_wall_s"]))
    s_open: AbstractContextManager[SContextProvider | None] = (
        open_s_provider(args, profile, limits) if "S" in args.arms else nullcontext()
    )
    harness = ru.load_harness(args.frozen, layout)
    require_clean_process(os.environ, sys.modules, profile.routing)
    questions = {q.id: q for q in ru.public_questions(harness, args.frozen, layout, tuple(ids))}
    contracts = scorer.load_contracts(ru.question_paths(args.frozen, layout),
                                      layout.retired_questions)
    with s_open as provider:
        model, executors = live_clients(harness, args.arms, args.timbr_env, limits, profile)
        runtimes = ru.arm_runtimes(harness, args.frozen, layout, executors, provider, limits,
                                   knobs)
        sql = next((executors[arm] for arm in ("A", "S") if arm in executors), None)
        if sql is None:
            raise ValueError("source versions are read through arm A or S; include one in --arms")
        summary = run(slots, questions, runtimes, model, contracts, harness, settings,
                      args.run_dir / "receipts.jsonl", pins=profile.source_versions,
                      read_version=ru.version_reader(sql, profile.sources))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

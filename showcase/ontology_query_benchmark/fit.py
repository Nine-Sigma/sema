"""Offline fit report: whole-request token estimates per arm and question, zero calls.

python -m showcase.ontology_query_benchmark.fit --frozen <closure dir>
    [--receipts <harness>/runs/v3_20261021_rw/ab_w4s] [--budget-tokens N]
    [--values-top-k K] [--description-cap-chars C] [--s-context module:function] [--out f]
    [--input-ceiling 16000] [--repair-reserve 2066]
    [--sweep]   # every SWEEP_GRID point (knobs x ceiling) instead of one setting

Calibration rebuilds each Oct 21 first-generation prompt from the frozen config and caches
(only shards whose config, question and cache hashes match) and divides its characters by
the endpoint's prompt_tokens. Each request is estimated at the min (conservative), median
and max (lenient) ratio against --input-ceiling minus --repair-reserve, the runner's
first-prompt limit. A repair turn is the prompt_tokens growth between consecutive attempts.
"""

import argparse
import dataclasses
from dataclasses import dataclass
import itertools
import json
import statistics
from collections.abc import Callable, Iterable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from showcase.ontology_query_benchmark import runner_utils as ru
from showcase.ontology_query_benchmark.freeze import HARNESS_ROOT, sha256_file
from showcase.ontology_query_benchmark.runner_utils import Knobs

RECEIPTS = HARNESS_ROOT / "runs/v3_20261021_rw/ab_w4s"
SWEEP_GRID = tuple(itertools.product((16_000, 8_000, 6_000, 5_000), (25, 10, 5, 0), (160, 80),
                                     (16_000, 36_000)))


@dataclass(frozen=True)
class Ratios:
    """Characters per endpoint prompt token from fit.calibrate: min, median and max."""

    conservative: float
    median: float
    lenient: float


def stats(values: Iterable[float]) -> dict[str, float]:
    found = sorted(values)
    if not found:
        return {"n": 0}
    return {"n": len(found), "min": found[0], "median": statistics.median(found),
            "max": found[-1]}  # fmt: skip


def _first_messages(harness: SimpleNamespace, runtime: Any, question: str) -> Any:
    context = runtime.context(question)
    messages = harness.workflow.build_messages(
        context, question, runtime.conventions, runtime.sql_rules
    )
    return context, messages


def _chars(messages: list[dict[str, Any]]) -> int:
    return sum(len(str(m.get("content") or "")) for m in messages)


def calibrate(harness: SimpleNamespace, frozen: Path, receipts: Path) -> dict[str, Any]:
    cache = frozen / ru.CACHE_DIR
    expected = {"A": sha256_file(cache / "schema.json"), "B": sha256_file(cache / ru.KG_CACHE)}
    config_hash = sha256_file(frozen / ru.W4S_CONFIG)
    question_hashes = {sha256_file(p) for p in ru.question_paths(frozen)}
    runtimes = ru.arm_runtimes(harness, frozen, {"A": None, "B": None}, None)
    texts = {q.id: q.question for q in ru.public_questions(harness, frozen)}
    chars: dict[tuple[str, str], int] = {}
    ratios: list[float] = []
    turns: list[float] = []
    skipped, used = 0, {}
    for summary in sorted(receipts.glob("*/shards/*/summary.json")):
        manifest = json.loads(summary.read_text())["manifest"]
        arm = manifest.get("arm")
        if (arm not in expected or manifest.get("config_sha256") != config_hash
                or manifest.get("questions_sha256") not in question_hashes
                or manifest.get("context_cache_sha256") != expected[arm]):  # fmt: skip
            skipped += 1
            continue
        results = summary.parent / "results.jsonl"
        used[str(results.relative_to(receipts))] = sha256_file(results)
        for line in results.read_text().splitlines():
            entry = json.loads(line)
            prompts = [a.get("prompt_tokens") for a in entry["attempts"]]
            turns += [b - a for a, b in zip(prompts, prompts[1:], strict=False) if a and b]
            if not prompts or not prompts[0] or entry["question_id"] not in texts:
                continue
            key = (arm, entry["question_id"])
            if key not in chars:
                chars[key] = _chars(_first_messages(harness, runtimes[arm], texts[key[1]])[1])
            ratios.append(chars[key] / prompts[0])
    return {"ratio": stats(ratios), "records": len(ratios), "skipped_shards": skipped,
            "repair_turn_tokens": stats(turns), "receipt_files": used}  # fmt: skip


def fit_report(harness: SimpleNamespace, frozen: Path, knobs: Knobs, *, ratios: Ratios,
               limits: ru.Limits, s_provider: Callable[[str], Any] | None) -> dict[str, Any]:
    """Per arm and question: first-prompt estimates at the three ratios against the limits.

    certain_overflow mirrors the runner guard (lenient estimate over the first-prompt limit,
    or an overflowing context); likely_fits uses the median ratio against the same limit.
    """
    arms: dict[str, None] = {"A": None, "B": None} | ({"S": None} if s_provider else {})
    runtimes = ru.arm_runtimes(harness, frozen, arms, s_provider, knobs=knobs)
    first_limit = limits.first_prompt_ceiling
    rows = []
    for arm, runtime in runtimes.items():
        for question in ru.public_questions(harness, frozen):
            context, messages = _first_messages(harness, runtime, question.question)
            chars, count = _chars(messages), len(messages)
            low, mid, high = (ru.estimate_tokens_from_chars(chars, count, r)
                              for r in (ratios.lenient, ratios.median, ratios.conservative))
            usable = not context.overflow
            rows.append({"arm": arm, "question_id": question.id,
                         "context_overflow": context.overflow, "request_chars": chars,
                         "estimated_tokens": high, "estimate_median": mid,
                         "estimate_lenient": low,
                         "fits_first": usable and high <= limits.input_ceiling,
                         "fits_two_repairs": usable and high <= first_limit,
                         "certain_overflow": not usable or low > first_limit,
                         "likely_fits": usable and mid <= first_limit})  # fmt: skip
    summary = {arm: _arm_summary([r for r in rows if r["arm"] == arm]) for arm in runtimes}
    return {"knobs": dataclasses.asdict(knobs), "ratios": dataclasses.asdict(ratios),
            "input_ceiling": limits.input_ceiling, "repair_reserve": limits.repair_reserve,
            "arms": summary, "rows": rows}  # fmt: skip


FLAGS = ("context_overflow", "fits_first", "fits_two_repairs", "certain_overflow", "likely_fits")


def _arm_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {"questions": len(rows), **{flag: sum(r[flag] for r in rows) for flag in FLAGS},
            "estimated_tokens": stats(r["estimated_tokens"] for r in rows),
            "max_median": max(r["estimate_median"] for r in rows),
            "max_lenient": max(r["estimate_lenient"] for r in rows)}  # fmt: skip


def sweep(harness: SimpleNamespace, frozen: Path, grid: Iterable[tuple[Knobs, int]], *,
          ratios: Ratios, limits: ru.Limits,
          s_provider: Callable[[str], Any] | None = None) -> list[dict[str, Any]]:  # fmt: skip
    """Per (knobs, input ceiling) point and arm: the fit counts and the largest estimates."""
    rows = []
    for knobs, ceiling in grid:
        point = dataclasses.replace(limits, input_ceiling=ceiling)
        report = fit_report(harness, frozen, knobs, ratios=ratios, limits=point,
                            s_provider=s_provider)  # fmt: skip
        arms = {arm: {**{flag: a[flag] for flag in FLAGS},
                      "max_tokens": a["estimated_tokens"]["max"], "max_median": a["max_median"],
                      "max_lenient": a["max_lenient"]}
                for arm, a in report["arms"].items()}  # fmt: skip
        rows.append({"knobs": report["knobs"], "input_ceiling": ceiling, "arms": arms})
    return rows


def _parse(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--receipts", type=Path, default=RECEIPTS)
    parser.add_argument("--s-context")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--sweep", action="store_true", help="report the SWEEP_GRID instead")
    ru.add_knob_args(parser)
    ru.add_limit_args(parser)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    harness = ru.load_harness(args.frozen)
    calibration = calibrate(harness, args.frozen, args.receipts)
    if not calibration["records"]:
        raise ValueError("no Oct 21 receipt matches the frozen inputs; cannot calibrate")
    provider = ru.load_s_provider(args.s_context) if args.s_context else None
    ratio = calibration["ratio"]
    ratios = Ratios(ratio["min"], ratio["median"], ratio["max"])
    limits = ru.limits_from(args)
    head = {"ratios": dataclasses.asdict(ratios), "limits": dataclasses.asdict(limits),
            "calibration": {k: calibration[k] for k in ("ratio", "records", "skipped_shards",
                                                         "repair_turn_tokens")}}  # fmt: skip
    if args.sweep:
        grid = [(Knobs(*knobs), ceiling) for *knobs, ceiling in SWEEP_GRID]
        body: dict[str, Any] = {"sweep": sweep(harness, args.frozen, grid, ratios=ratios,
                                               limits=limits, s_provider=provider)}
    else:
        report = fit_report(harness, args.frozen, ru.knobs_from(args), ratios=ratios,
                            limits=limits, s_provider=provider)  # fmt: skip
        body = {"knobs": report["knobs"], "arms": report["arms"], "rows": report["rows"]}
    result = head | body | {"receipt_files": calibration["receipt_files"]}
    if args.out:
        args.out.write_text(json.dumps(result, indent=1, sort_keys=True, default=str))
    print(json.dumps({k: v for k, v in result.items() if k not in ("rows", "receipt_files")},
                     indent=1, sort_keys=True))  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

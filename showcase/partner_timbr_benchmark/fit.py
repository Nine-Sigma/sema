"""Offline fit report: whole-request token estimates per arm and question, zero calls.

python -m showcase.partner_timbr_benchmark.fit --frozen <closure dir>
    [--receipts <harness>/runs/v3_20261021_rw/ab_w4s] [--budget-tokens N]
    [--values-top-k K] [--description-cap-chars C] [--s-context module:function] [--out f]
    [--sweep]   # every SWEEP_GRID point instead of one knob set

Calibration rebuilds each Oct 21 first-generation prompt from the frozen config and caches
(only shards whose config, question and cache hashes match) and divides its characters by
the endpoint's prompt_tokens. Estimates use the lowest ratio (the most tokens per
character). A repair turn is the prompt_tokens growth between consecutive attempts.
"""

import argparse
import dataclasses
import itertools
import json
import statistics
from collections.abc import Callable, Iterable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from showcase.partner_timbr_benchmark import runner_utils as ru
from showcase.partner_timbr_benchmark.freeze import HARNESS_ROOT, sha256_file
from showcase.partner_timbr_benchmark.runner_utils import Knobs

RECEIPTS = HARNESS_ROOT / "runs/v3_20261021_rw/ab_w4s"
SWEEP_GRID = tuple(itertools.product((16_000, 8_000, 6_000, 5_000), (25, 10, 5, 0), (160, 80)))


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


def fit_report(harness: SimpleNamespace, frozen: Path, knobs: Knobs, *, chars_per_token: float,
               repair_reserve: int, s_provider: Callable[[str], Any] | None,
               input_ceiling: int = ru.Limits().input_ceiling) -> dict[str, Any]:  # fmt: skip
    arms: dict[str, None] = {"A": None, "B": None} | ({"S": None} if s_provider else {})
    runtimes = ru.arm_runtimes(harness, frozen, arms, s_provider, knobs=knobs)
    rows = []
    for arm, runtime in runtimes.items():
        for question in ru.public_questions(harness, frozen):
            context, messages = _first_messages(harness, runtime, question.question)
            chars = _chars(messages)
            estimate = ru.estimate_tokens_from_chars(chars, len(messages), chars_per_token)
            usable = not context.overflow
            rows.append({"arm": arm, "question_id": question.id,
                         "context_overflow": context.overflow, "request_chars": chars,
                         "estimated_tokens": estimate,
                         "fits_first": usable and estimate <= input_ceiling,
                         "fits_two_repairs": usable
                         and estimate + 2 * repair_reserve <= input_ceiling})  # fmt: skip
    summary = {arm: _arm_summary([r for r in rows if r["arm"] == arm]) for arm in runtimes}
    return {"knobs": dataclasses.asdict(knobs), "chars_per_token": chars_per_token,
            "repair_reserve": repair_reserve, "input_ceiling": input_ceiling,
            "arms": summary, "rows": rows}  # fmt: skip


def _arm_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {"questions": len(rows),
            "context_overflow": sum(r["context_overflow"] for r in rows),
            "fits_first": sum(r["fits_first"] for r in rows),
            "fits_two_repairs": sum(r["fits_two_repairs"] for r in rows),
            "estimated_tokens": stats(r["estimated_tokens"] for r in rows)}  # fmt: skip


def sweep(harness: SimpleNamespace, frozen: Path, grid: Iterable[Knobs], *,
          chars_per_token: float, repair_reserve: int,
          s_provider: Callable[[str], Any] | None = None) -> list[dict[str, Any]]:  # fmt: skip
    """Per grid point and arm: questions that fit first and with two repairs, max estimate."""
    rows = []
    for knobs in grid:
        report = fit_report(harness, frozen, knobs, chars_per_token=chars_per_token,
                            repair_reserve=repair_reserve, s_provider=s_provider)  # fmt: skip
        arms = {arm: {"fits_first": a["fits_first"], "fits_two_repairs": a["fits_two_repairs"],
                      "context_overflow": a["context_overflow"],
                      "max_tokens": a["estimated_tokens"]["max"]}
                for arm, a in report["arms"].items()}  # fmt: skip
        rows.append({"knobs": report["knobs"], "arms": arms})
    return rows


def _parse(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--receipts", type=Path, default=RECEIPTS)
    parser.add_argument("--s-context")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--sweep", action="store_true", help="report the SWEEP_GRID instead")
    ru.add_knob_args(parser)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    harness = ru.load_harness(args.frozen)
    calibration = calibrate(harness, args.frozen, args.receipts)
    if not calibration["records"]:
        raise ValueError("no Oct 21 receipt matches the frozen inputs; cannot calibrate")
    provider = ru.load_s_provider(args.s_context) if args.s_context else None
    ratio = calibration["ratio"]["min"]
    reserve = int(calibration["repair_turn_tokens"].get("max", 0))
    head = {"chars_per_token": ratio, "repair_reserve": reserve,
            "calibration": {k: calibration[k] for k in ("ratio", "records", "skipped_shards",
                                                         "repair_turn_tokens")}}  # fmt: skip
    if args.sweep:
        grid = [Knobs(*point) for point in SWEEP_GRID]
        body: dict[str, Any] = {"sweep": sweep(harness, args.frozen, grid, chars_per_token=ratio,
                                               repair_reserve=reserve, s_provider=provider)}
    else:
        report = fit_report(harness, args.frozen, ru.knobs_from(args), chars_per_token=ratio,
                            repair_reserve=reserve, s_provider=provider)  # fmt: skip
        body = {"knobs": report["knobs"], "arms": report["arms"], "rows": report["rows"]}
    result = head | body | {"receipt_files": calibration["receipt_files"]}
    if args.out:
        args.out.write_text(json.dumps(result, indent=1, sort_keys=True, default=str))
    print(json.dumps({k: v for k, v in result.items() if k not in ("rows", "receipt_files")},
                     indent=1, sort_keys=True))  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

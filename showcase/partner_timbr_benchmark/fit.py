"""Offline fit report: whole-request token estimates per arm and question, zero calls.

python -m showcase.partner_timbr_benchmark.fit --frozen <closure dir>
    [--receipts <harness>/runs/v3_20261021_rw/ab_w4s] [--budget-tokens N]
    [--values-top-k K] [--description-cap-chars C] [--s-context module:function] [--out f]

Calibration rebuilds each Oct 21 first-generation prompt from the frozen config and caches
(only shards whose config, question and cache hashes match) and divides its characters by
the endpoint's prompt_tokens. Estimates use the lowest ratio (the most tokens per
character). A repair turn is the prompt_tokens growth between consecutive attempts.
"""

import argparse
import dataclasses
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
            rows.append({"arm": arm, "question_id": question.id,
                         "context_overflow": context.overflow, "request_chars": chars,
                         "estimated_tokens": estimate, "fits_first": estimate <= input_ceiling,
                         "fits_two_repairs": estimate + 2 * repair_reserve <= input_ceiling})
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--receipts", type=Path, default=RECEIPTS)
    parser.add_argument("--s-context")
    parser.add_argument("--out", type=Path)
    ru.add_knob_args(parser)
    args = parser.parse_args(argv)
    harness = ru.load_harness(args.frozen)
    calibration = calibrate(harness, args.frozen, args.receipts)
    if not calibration["records"]:
        raise ValueError("no Oct 21 receipt matches the frozen inputs; cannot calibrate")
    provider = ru.load_s_provider(args.s_context) if args.s_context else None
    reserve = int(calibration["repair_turn_tokens"].get("max", 0))
    report = fit_report(harness, args.frozen, ru.knobs_from(args),
                        chars_per_token=calibration["ratio"]["min"], repair_reserve=reserve,
                        s_provider=provider)  # fmt: skip
    report["calibration"] = calibration
    text = json.dumps(report, indent=1, sort_keys=True, default=str)
    if args.out:
        args.out.write_text(text)
    print(json.dumps({k: report[k] for k in ("knobs", "chars_per_token", "repair_reserve",
                                             "arms")} | {"calibration": {k: calibration[k]
                     for k in ("ratio", "records", "skipped_shards", "repair_turn_tokens")}},
                     indent=1, sort_keys=True))  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

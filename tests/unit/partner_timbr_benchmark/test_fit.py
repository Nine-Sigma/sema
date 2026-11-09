import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from showcase.partner_timbr_benchmark import fit, freeze, runner, runner_utils
from showcase.partner_timbr_benchmark.runner_utils import Knobs

pytestmark = pytest.mark.unit

def first_prompt_chars(harness: SimpleNamespace, frozen: Path, arm: str, qid: str,
                       knobs: Knobs = Knobs()) -> int:  # fmt: skip
    runtime = runner_utils.arm_runtimes(harness, frozen, {arm: None}, None, knobs=knobs)[arm]
    question = runner_utils.public_questions(harness, frozen, (qid,))[0]
    context = runtime.context(question.question)
    messages = harness.workflow.build_messages(
        context, question.question, runtime.conventions, runtime.sql_rules
    )
    return sum(len(m["content"]) for m in messages)


def write_shard(root: Path, frozen: Path, arm: str, records: list[dict[str, Any]],
                config_hash: str | None = None) -> None:  # fmt: skip
    cache = frozen / runner_utils.CACHE_DIR
    shard = root / "regular_live" / "shards" / f"arm_{arm}_0"
    shard.mkdir(parents=True)
    manifest = {
        "arm": arm,
        "config_sha256": config_hash or freeze.sha256_file(frozen / runner_utils.W4S_CONFIG),
        "questions_sha256": freeze.sha256_file(runner_utils.question_paths(frozen)[0]),
        "context_cache_sha256": freeze.sha256_file(
            cache / ("schema.json" if arm == "A" else runner_utils.KG_CACHE)
        ),
    }
    (shard / "summary.json").write_text(json.dumps({"manifest": manifest}))
    (shard / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))


def record(qid: str, prompts: list[int]) -> dict[str, Any]:
    attempts = [{"attempt": i + 1, "prompt_tokens": p} for i, p in enumerate(prompts)]
    return {"question_id": qid, "arm": "A", "attempts": attempts}


def test_calibration_divides_rebuilt_prompt_chars_by_endpoint_tokens(
    harness: SimpleNamespace, frozen: Path, tmp_path: Path
) -> None:
    write_shard(tmp_path, frozen, "A", [record("Q01", [17_000, 21_000, 24_500])])
    write_shard(tmp_path, frozen, "B", [record("Q01", [28_000])], config_hash="stale")
    result = fit.calibrate(harness, frozen, tmp_path)
    chars = first_prompt_chars(harness, frozen, "A", "Q01")
    assert result["ratio"]["min"] == result["ratio"]["max"] == pytest.approx(chars / 17_000)
    assert result["records"] == 1 and result["skipped_shards"] == 1
    assert result["repair_turn_tokens"]["max"] == 4_000
    assert result["repair_turn_tokens"]["min"] == 3_500


def test_common_knobs_reshape_contexts_through_the_harness_builders(
    harness: SimpleNamespace, frozen: Path
) -> None:
    base_a = first_prompt_chars(harness, frozen, "A", "Q01")
    base_b = first_prompt_chars(harness, frozen, "B", "Q01")
    assert first_prompt_chars(harness, frozen, "A", "Q01", Knobs(values_top_k=3)) < base_a
    assert first_prompt_chars(harness, frozen, "B", "Q01", Knobs(budget_tokens=4_000)) < base_b
    small_cap = Knobs(description_cap_chars=40)
    assert first_prompt_chars(harness, frozen, "B", "Q01", small_cap) < base_b
    runtime = runner_utils.arm_runtimes(harness, frozen, {"A": None}, None,
                                        knobs=Knobs(budget_tokens=2_000))["A"]  # fmt: skip
    assert runtime.context("q").overflow is True


def test_values_top_k_cannot_exceed_the_frozen_cache(harness: SimpleNamespace,
                                                     frozen: Path) -> None:  # fmt: skip
    with pytest.raises(ValueError, match="values_top_k"):
        runner_utils.arm_runtimes(harness, frozen, {"A": None}, None, knobs=Knobs(values_top_k=26))


def test_fit_report_covers_every_arm_question_pair_with_the_guard_estimate(
    harness: SimpleNamespace, frozen: Path
) -> None:
    report = fit.fit_report(harness, frozen, Knobs(), chars_per_token=2.0,
                            repair_reserve=1_000, s_provider=None)  # fmt: skip
    rows = report["rows"]
    assert {r["arm"] for r in rows} == {"A", "B"} and len(rows) == 98
    q01 = next(r for r in rows if r["arm"] == "A" and r["question_id"] == "Q01")
    chars = first_prompt_chars(harness, frozen, "A", "Q01")
    assert q01["estimated_tokens"] == runner_utils.estimate_tokens_from_chars(chars, 2, 2.0)
    assert q01["fits_first"] == (q01["estimated_tokens"] <= 16_000)
    assert q01["fits_two_repairs"] == (q01["estimated_tokens"] + 2_000 <= 16_000)
    assert report["arms"]["A"]["questions"] == 49


def test_fit_report_includes_s_when_a_provider_is_given(harness: SimpleNamespace,
                                                         frozen: Path) -> None:  # fmt: skip
    provider = lambda q: SimpleNamespace(text="T" * 100, identity="i", findings=())  # noqa: E731
    report = fit.fit_report(harness, frozen, Knobs(), chars_per_token=2.0,
                            repair_reserve=0, s_provider=provider)  # fmt: skip
    assert report["arms"]["S"]["questions"] == 49


def test_run_manifest_pins_knobs_and_refuses_a_different_resume(
    frozen: Path, tmp_path: Path
) -> None:
    args = ["--frozen", str(frozen), "--run-dir", str(tmp_path), "--dry-run"]
    runner.main([*args, "--budget-tokens", "5000"])
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["knobs"] == {"budget_tokens": 5000, "values_top_k": None,
                                 "description_cap_chars": None}  # fmt: skip
    assert manifest["closure_id"] == frozen.name
    with pytest.raises(ValueError, match="manifest"):
        runner.main([*args, "--budget-tokens", "6000"])

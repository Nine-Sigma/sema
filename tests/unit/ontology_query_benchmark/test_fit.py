import functools
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from showcase.ontology_query_benchmark import fit, freeze, runner, runner_utils
from showcase.ontology_query_benchmark.profile import Knobs, Profile
from tests.unit.ontology_query_benchmark.conftest import PROFILE_PATH, write_settings

pytestmark = pytest.mark.unit


def runtime_for(harness: SimpleNamespace, frozen: Path, profile: Profile, arm: str,
                knobs: Knobs = Knobs()) -> Any:  # fmt: skip
    return runner_utils.arm_runtimes(harness, frozen, profile.harness, {arm: None}, None,
                                     profile.limits, knobs)[arm]  # fmt: skip


def first_prompt_chars(harness: SimpleNamespace, frozen: Path, profile: Profile, arm: str,
                       qid: str, knobs: Knobs = Knobs()) -> int:  # fmt: skip
    runtime = runtime_for(harness, frozen, profile, arm, knobs)
    question = runner_utils.public_questions(harness, frozen, profile.harness, (qid,))[0]
    context = runtime.context(question.question)
    messages = harness.workflow.build_messages(
        context, question.question, runtime.conventions, runtime.sql_rules
    )
    return sum(len(m["content"]) for m in messages)


def write_shard(root: Path, frozen: Path, profile: Profile, arm: str,
                records: list[dict[str, Any]], config_hash: str | None = None) -> None:  # fmt: skip
    layout = profile.harness
    cache = frozen / layout.cache_dir
    shard = root / "regular_live" / "shards" / f"arm_{arm}_0"
    shard.mkdir(parents=True)
    manifest = {
        "arm": arm,
        "config_sha256": config_hash or freeze.sha256_file(frozen / layout.config),
        "questions_sha256": freeze.sha256_file(runner_utils.question_paths(frozen, layout)[0]),
        "context_cache_sha256": freeze.sha256_file(
            cache / ("schema.json" if arm == "A" else layout.kg_cache)
        ),
    }
    (shard / "summary.json").write_text(json.dumps({"manifest": manifest}))
    (shard / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))


def record(qid: str, prompts: list[int]) -> dict[str, Any]:
    attempts = [{"attempt": i + 1, "prompt_tokens": p} for i, p in enumerate(prompts)]
    return {"question_id": qid, "arm": "A", "attempts": attempts}


def test_calibration_divides_rebuilt_prompt_chars_by_endpoint_tokens(
    harness: SimpleNamespace, frozen: Path, profile: Profile, tmp_path: Path
) -> None:
    write_shard(tmp_path, frozen, profile, "A", [record("R01", [17_000, 21_000, 24_500])])
    write_shard(tmp_path, frozen, profile, "B", [record("R01", [28_000])], config_hash="stale")
    result = fit.calibrate(harness, frozen, profile.harness, tmp_path, profile.limits)
    chars = first_prompt_chars(harness, frozen, profile, "A", "R01")
    assert result["ratio"]["min"] == result["ratio"]["max"] == pytest.approx(chars / 17_000)
    assert result["records"] == 1 and result["skipped_shards"] == 1
    assert result["repair_turn_tokens"]["max"] == 4_000
    assert result["repair_turn_tokens"]["min"] == 3_500


def test_common_knobs_reshape_contexts_through_the_harness_builders(
    harness: SimpleNamespace, frozen: Path, profile: Profile
) -> None:
    chars = functools.partial(first_prompt_chars, harness, frozen, profile)
    base_a, base_b = chars("A", "R01"), chars("B", "R01")
    assert chars("A", "R01", Knobs(values_top_k=3)) < base_a
    assert chars("B", "R01", Knobs(budget_tokens=100)) < base_b
    assert chars("B", "R01", Knobs(description_cap_chars=40)) < base_b
    runtime = runtime_for(harness, frozen, profile, "A", Knobs(budget_tokens=20))
    assert runtime.context("q").overflow is True


def test_values_top_k_cannot_exceed_the_frozen_cache(
    harness: SimpleNamespace, frozen: Path, profile: Profile
) -> None:
    with pytest.raises(ValueError, match="values_top_k"):
        runtime_for(harness, frozen, profile, "A", Knobs(values_top_k=26))


def test_values_top_k_zero_drops_the_a_values_block(
    harness: SimpleNamespace, frozen: Path, profile: Profile
) -> None:
    text = runtime_for(harness, frozen, profile, "A", Knobs(values_top_k=0)).context("q").text
    assert "TABLE " in text and "\nVALUES " not in text


RATIOS = fit.Ratios(conservative=2.0, median=2.2, lenient=2.3)


def test_sweep_reports_fit_counts_per_arm_for_each_knob_and_ceiling(
    harness: SimpleNamespace, frozen: Path, profile: Profile
) -> None:
    grid = [(Knobs(budget_tokens=16_000, values_top_k=25, description_cap_chars=160), 16_000),
            (Knobs(budget_tokens=6_000, values_top_k=0, description_cap_chars=80), 36_000)]
    rows = fit.sweep(harness, frozen, profile.harness, grid, ratios=RATIOS,
                     limits=profile.limits)  # fmt: skip
    assert [(r["knobs"]["budget_tokens"], r["input_ceiling"]) for r in rows] == [
        (16_000, 16_000), (6_000, 36_000)]  # fmt: skip
    wide = profile.limits.model_copy(update={"input_ceiling": 36_000})
    report = fit.fit_report(harness, frozen, profile.harness, grid[1][0], ratios=RATIOS,
                            limits=wide, s_provider=None)  # fmt: skip
    keys = ("fits_first", "fits_two_repairs", "certain_overflow", "likely_fits")
    for arm in ("A", "B"):
        assert {k: rows[1]["arms"][arm][k] for k in keys} == {
            k: report["arms"][arm][k] for k in keys}  # fmt: skip


def test_fit_report_reads_three_estimates_against_the_ceiling(
    harness: SimpleNamespace, frozen: Path, profile: Profile
) -> None:
    report = fit.fit_report(harness, frozen, profile.harness, Knobs(), ratios=RATIOS,
                            limits=profile.limits, s_provider=None)  # fmt: skip
    rows = report["rows"]
    assert {r["arm"] for r in rows} == {"A", "B"} and len(rows) == 6
    q01 = next(r for r in rows if r["arm"] == "A" and r["question_id"] == "R01")
    chars = first_prompt_chars(harness, frozen, profile, "A", "R01")

    def tokens(ratio: float) -> int:
        return runner_utils.estimate_tokens_from_chars(chars, 2, ratio)

    assert (q01["estimated_tokens"], q01["estimate_median"], q01["estimate_lenient"]) == (
        tokens(2.0), tokens(2.2), tokens(2.3))  # fmt: skip
    limit = 16_000 - 2_066
    assert q01["fits_first"] == (tokens(2.0) <= 16_000)
    assert q01["fits_two_repairs"] == (tokens(2.0) <= limit)
    assert q01["certain_overflow"] == (tokens(2.3) > limit)
    assert q01["likely_fits"] == (tokens(2.2) <= limit)
    assert report["arms"]["A"]["questions"] == 3


def test_an_overflowing_context_never_counts_as_fitting(
    harness: SimpleNamespace, frozen: Path, profile: Profile
) -> None:
    report = fit.fit_report(harness, frozen, profile.harness, Knobs(budget_tokens=20),
                            ratios=RATIOS, limits=profile.limits, s_provider=None)  # fmt: skip
    a_rows = [r for r in report["rows"] if r["arm"] == "A"]
    assert all(r["context_overflow"] and r["certain_overflow"] for r in a_rows)
    assert not any(r["fits_first"] or r["fits_two_repairs"] or r["likely_fits"] for r in a_rows)


def test_fit_report_includes_s_when_a_provider_is_given(
    harness: SimpleNamespace, frozen: Path, profile: Profile
) -> None:
    provider = lambda q: SimpleNamespace(text="T" * 100, identity="i", findings=())  # noqa: E731
    report = fit.fit_report(harness, frozen, profile.harness, Knobs(), ratios=RATIOS,
                            limits=profile.limits, s_provider=provider)  # fmt: skip
    assert report["arms"]["S"]["questions"] == 3


def test_run_manifest_pins_knobs_and_limits_and_refuses_a_different_resume(
    frozen: Path, tmp_path: Path
) -> None:
    args = ["--profile", str(PROFILE_PATH), "--frozen", str(frozen), "--run-dir", str(tmp_path),
            "--dry-run", "--settings", str(write_settings(tmp_path))]  # fmt: skip
    runner.main([*args, "--budget-tokens", "5000", "--input-ceiling", "36000"])
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["knobs"] == {"budget_tokens": 5000, "values_top_k": 5,
                                 "description_cap_chars": 160}  # fmt: skip
    assert (manifest["limits"]["input_ceiling"], manifest["limits"]["repair_reserve"]) == (
        36_000, 2_066)  # fmt: skip
    assert manifest["closure_id"] == frozen.name
    with pytest.raises(ValueError, match="manifest"):
        runner.main([*args, "--budget-tokens", "5000", "--input-ceiling", "16000"])
    with pytest.raises(ValueError, match="manifest"):
        runner.main([*args, "--budget-tokens", "3000", "--input-ceiling", "36000"])


def test_profile_knobs_and_limits_are_the_cli_default(frozen: Path, tmp_path: Path) -> None:
    runner.main(["--profile", str(PROFILE_PATH), "--frozen", str(frozen), "--run-dir",
                 str(tmp_path), "--dry-run",
                 "--settings", str(write_settings(tmp_path))])  # fmt: skip
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["knobs"] == {"budget_tokens": 3000, "values_top_k": 5,
                                 "description_cap_chars": 160}  # fmt: skip
    assert (manifest["limits"]["input_ceiling"], manifest["limits"]["repair_reserve"]) == (
        16_000, 2_066)  # fmt: skip

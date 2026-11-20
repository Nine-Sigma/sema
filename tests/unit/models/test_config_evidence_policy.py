"""Evidence policy and budgets on BuildConfig (design 5.3, 5.9)."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from sema.models.config import BuildConfig, ProfilingConfig

pytestmark = pytest.mark.unit


def _rich_profiling() -> ProfilingConfig:
    return ProfilingConfig(sample_rows=9, top_k_values=33, enable_distinct_counts=True)


def test_build_config_budget_defaults() -> None:
    config = BuildConfig()

    assert config.evidence_policy == "full"
    assert config.max_distinct_values_read == 200
    assert config.max_unmatched_reported == 20
    assert config.judge_workers == 4


def test_profiling_enable_distinct_counts_defaults_true() -> None:
    assert ProfilingConfig().enable_distinct_counts is True


def test_evidence_policy_rejects_unknown_value() -> None:
    with pytest.raises(ValidationError) as excinfo:
        BuildConfig(evidence_policy="everything")

    assert [e["type"] for e in excinfo.value.errors()] == ["literal_error"]


@pytest.mark.parametrize("policy", ["metadata_only", "bounded_values", "full"])
def test_evidence_policy_accepts_named_values(policy: str) -> None:
    assert BuildConfig(evidence_policy=policy).evidence_policy == policy


def test_metadata_only_forces_value_free_profiling_and_no_fk() -> None:
    config = BuildConfig(
        evidence_policy="metadata_only",
        enable_fk_detection=True,
        profiling=_rich_profiling(),
    )

    assert config.profiling.sample_rows == 0
    assert config.profiling.top_k_values == 0
    assert config.profiling.enable_distinct_counts is False
    assert config.enable_fk_detection is False


def test_metadata_only_forces_defaults_too() -> None:
    config = BuildConfig(evidence_policy="metadata_only")

    assert config.profiling.sample_rows == 0
    assert config.profiling.top_k_values == 0
    assert config.profiling.enable_distinct_counts is False
    assert config.enable_fk_detection is False


def test_metadata_only_leaves_unrelated_profiling_fields() -> None:
    profiling = ProfilingConfig(categorical_threshold=77, max_sample_values=3)
    config = BuildConfig(evidence_policy="metadata_only", profiling=profiling)

    assert config.profiling.categorical_threshold == 77
    assert config.profiling.max_sample_values == 3


@pytest.mark.parametrize(
    "overrides",
    [
        {"enable_fk_detection": True, "profiling": {"sample_rows": 9,
                                                    "top_k_values": 33,
                                                    "enable_distinct_counts": True}},
        {"enable_fk_detection": False, "profiling": {"sample_rows": 0,
                                                     "top_k_values": 0,
                                                     "enable_distinct_counts": False}},
    ],
)
def test_full_leaves_every_caller_value(overrides: dict[str, Any]) -> None:
    config = BuildConfig(evidence_policy="full", **overrides)

    profiling = overrides["profiling"]
    assert config.enable_fk_detection is overrides["enable_fk_detection"]
    assert config.profiling.sample_rows == profiling["sample_rows"]
    assert config.profiling.top_k_values == profiling["top_k_values"]
    assert config.profiling.enable_distinct_counts is profiling["enable_distinct_counts"]


def test_budgets_accept_caller_values() -> None:
    config = BuildConfig(
        max_distinct_values_read=50, max_unmatched_reported=5, judge_workers=2,
    )

    assert (config.max_distinct_values_read, config.max_unmatched_reported,
            config.judge_workers) == (50, 5, 2)

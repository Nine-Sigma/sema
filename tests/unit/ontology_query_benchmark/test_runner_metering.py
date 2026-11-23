"""Done-when 4 (N5): the meter counts what is sent.

A model call that is sent and then raises is charged by the rule the runner applies to a reply
that reports no usage: the request's conservative prompt estimate (the second number of each
`request_estimates` pair) and the output ceiling, at the profile's prices. So a run stops at
`max_cost_usd` on raised calls alone. Arm S's embedding calls are counted in each receipt's
`embedding_calls` and charged at `settings.prices.embedding_usd_per_1m`; a profile with no
such price loads and charges them 0. A call that returns usage is charged from its usage, as
today. Every run here is on the neutral set with the toy harness: no network, no model.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import yaml

from showcase.ontology_query_benchmark.profile import load_sealed_profile
from tests.unit.ontology_query_benchmark.sealed_set_env import (
    SEALED_SET_FIXTURE,
    ScriptedModel,
    materialize,
    question_texts,
    read_jsonl,
    sealed_set_module,
)

pytestmark = pytest.mark.unit


class RaisedAfterSend:
    """A model client whose request reaches the provider and then fails client-side."""

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, messages: list[dict[str, Any]]) -> Any:
        self.calls += 1
        raise TimeoutError("read timed out after the request was sent")


class CountingEmbedder:
    def __init__(self) -> None:
        self.calls = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls += len(texts)
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


def profile_with(root: Path, frozen: Path, **settings: Any) -> Path:
    path = materialize(root, frozen)
    raw = yaml.safe_load(path.read_text())
    prices = {**raw["settings"]["prices"], **settings.pop("prices", {})}
    raw["settings"] = {**raw["settings"], **settings, "prices": prices}
    path.write_text(yaml.safe_dump(raw))
    return path


def run(path: Path, arms: tuple[str, ...], model: Any, s_provider: Any = None
        ) -> tuple[dict[str, Any], list[dict[str, Any]]]:  # fmt: skip
    summary = sealed_set_module().run_sealed_set(path, arms, 1, model=model,
                                                 s_provider=s_provider,
                                                 report=path.parent / "report.md")  # fmt: skip
    sealed = path.parent / "neutral" / "sealed"
    receipts = [r for p in sorted(sealed.rglob("receipts.jsonl")) for r in read_jsonl(p)]
    return summary, receipts


def generation_cost(receipt: dict[str, Any], path: Path) -> float:
    prices = load_sealed_profile(path).settings.prices
    return (receipt["prompt_tokens"] * prices.input_usd_per_1m
            + receipt["completion_tokens"] * prices.output_usd_per_1m) / 1e6  # fmt: skip


def test_a_call_that_raises_after_it_is_sent_is_charged_its_estimate(
    frozen: Path, tmp_path: Path,
) -> None:
    path = profile_with(tmp_path / "sealed_set", frozen)
    model = RaisedAfterSend()

    summary, receipts = run(path, ("A",), model)

    profile = load_sealed_profile(path)
    prices, ceiling = profile.settings.prices, profile.limits.output_ceiling
    expected = {r["slot_id"]: sum(estimate * prices.input_usd_per_1m
                                  + ceiling * prices.output_usd_per_1m
                                  for _, estimate in r["request_estimates"]) / 1e6
                for r in receipts}  # fmt: skip
    assert model.calls == sum(len(r["request_estimates"]) for r in receipts) > 0
    assert {r["slot_id"]: r["cost_usd"] for r in receipts} == pytest.approx(expected, abs=1e-6)
    assert summary["spent_usd"] == pytest.approx(sum(expected.values()), abs=1e-5)


def test_the_run_stops_at_max_cost_usd_on_raised_calls_alone(
    frozen: Path, tmp_path: Path,
) -> None:
    path = profile_with(tmp_path / "sealed_set", frozen, max_cost_usd=1e-6)
    model = RaisedAfterSend()

    summary, receipts = run(path, ("A",), model)

    assert summary["stopped"] == "cost_ceiling"
    assert model.calls == 1
    assert Counter(r["outcome"] for r in receipts)["unrun"] == len(receipts) - 1


def _arm_s_run(frozen: Path, harness: SimpleNamespace, root: Path, **prices: float
               ) -> tuple[dict[str, Any], list[dict[str, Any]], CountingEmbedder, Path]:  # fmt: skip
    from showcase.ontology_query_benchmark.core_s_provider import CoreSContextProvider

    path = profile_with(root, frozen, prices=prices)
    embedder = CountingEmbedder()
    provider = CoreSContextProvider(MagicMock(), embedder=embedder)
    model = ScriptedModel(harness, question_texts(SEALED_SET_FIXTURE / "neutral"), arm="A")
    summary, receipts = run(path, ("S",), model, provider)
    return summary, receipts, embedder, path


def test_a_profile_with_no_embedding_price_loads_counts_arm_s_embeddings_and_charges_0(
    frozen: Path, harness: SimpleNamespace, tmp_path: Path,
) -> None:
    summary, receipts, embedder, path = _arm_s_run(frozen, harness, tmp_path / "sealed_set")

    assert embedder.calls > 0
    assert sum(r.get("embedding_calls", 0) for r in receipts) == embedder.calls
    tokens_only = sum(generation_cost(r, path) for r in receipts)
    assert summary["spent_usd"] == pytest.approx(tokens_only, abs=1e-5)


def test_arm_s_embedding_calls_are_charged_at_the_profiles_embedding_price(
    frozen: Path, harness: SimpleNamespace, tmp_path: Path,
) -> None:
    extra: dict[float, float] = {}
    for price in (1000.0, 2000.0):
        summary, receipts, embedder, path = _arm_s_run(
            frozen, harness, tmp_path / str(int(price)) / "sealed_set", embedding_usd_per_1m=price)
        assert sum(r.get("embedding_calls", 0) for r in receipts) == embedder.calls > 0
        extra[price] = summary["spent_usd"] - sum(generation_cost(r, path) for r in receipts)

    assert extra[1000.0] > 0
    assert extra[2000.0] == pytest.approx(2 * extra[1000.0], rel=1e-3)


def test_decoy_a_call_that_returns_usage_is_charged_from_its_usage(
    frozen: Path, harness: SimpleNamespace, tmp_path: Path,
) -> None:
    path = profile_with(tmp_path / "sealed_set", frozen)
    model = ScriptedModel(harness, question_texts(SEALED_SET_FIXTURE / "neutral"), arm="A")

    summary, receipts = run(path, ("A",), model)

    assert receipts and all(r["usage_reported"] for r in receipts)
    assert {r["slot_id"]: r["cost_usd"] for r in receipts} == pytest.approx(
        {r["slot_id"]: generation_cost(r, path) for r in receipts}, abs=1e-6)
    assert summary["spent_usd"] == pytest.approx(
        sum(generation_cost(r, path) for r in receipts), abs=1e-5)

"""Done-when 4 (N5): one context budget for both arms.

Arm S's outcome is `overflow` when its context's token estimate (the harness's estimate of the
S provider's text) is above the run's `budget_tokens`, with no model call, as arm A's context
is held to the same budget. A context at the budget exactly runs.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from tests.unit.ontology_query_benchmark.sealed_set_env import (
    SEALED_SET_FIXTURE,
    ScriptedModel,
    materialize,
    question_texts,
    read_jsonl,
    sealed_set_module,
    stub_s_provider,
)

pytestmark = pytest.mark.unit


def stub_tokens(harness: SimpleNamespace) -> int:
    return int(harness.context.estimate_tokens(stub_s_provider("", lambda _: False).text))


def run_arm_s(frozen: Path, harness: SimpleNamespace, root: Path, budget: int
              ) -> tuple[list[dict[str, Any]], ScriptedModel]:  # fmt: skip
    path = materialize(root, frozen)
    raw = yaml.safe_load(path.read_text())
    raw["knobs"] = {"budget_tokens": budget}
    path.write_text(yaml.safe_dump(raw))
    model = ScriptedModel(harness, question_texts(SEALED_SET_FIXTURE / "neutral"))
    sealed_set_module().run_sealed_set(path, ("S",), 1, model=model, s_provider=stub_s_provider,
                                       report=root / "report.md")  # fmt: skip
    sealed = root / "neutral" / "sealed"
    return [r for p in sorted(sealed.rglob("receipts.jsonl")) for r in read_jsonl(p)], model


def test_arm_s_overflows_above_the_budget_with_no_model_call(
    frozen: Path, harness: SimpleNamespace, tmp_path: Path,
) -> None:
    receipts, model = run_arm_s(frozen, harness, tmp_path / "sealed_set", stub_tokens(harness) - 1)

    assert receipts
    assert {r["outcome"] for r in receipts} == {"overflow"}
    assert model.calls == []
    assert {r["generation_calls"] for r in receipts} == {0}


def test_decoy_arm_s_at_the_budget_exactly_runs(
    frozen: Path, harness: SimpleNamespace, tmp_path: Path,
) -> None:
    receipts, model = run_arm_s(frozen, harness, tmp_path / "sealed_set", stub_tokens(harness))

    assert receipts
    assert "overflow" not in {r["outcome"] for r in receipts}
    assert len(model.calls) >= len(receipts)

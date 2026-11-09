import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from showcase.ontology_query_benchmark import runner, runner_live_utils
from showcase.ontology_query_benchmark.routing import RoutingError
from tests.unit.ontology_query_benchmark.conftest import PROFILE_PATH
from tests.unit.ontology_query_benchmark.test_runner import (
    GOLD,
    R01_SQL,
    FakeExecutor,
    FakeModel,
    s_provider,
)

pytestmark = pytest.mark.unit

S_INPUTS = (
    "--neo4j-uri", "bolt://localhost:1", "--s-manifest", "build.json",
    "--s-projection", "projection.json", "--s-projection-receipt", "receipt.json",
)
PRICES = {"input_usd_per_1m": 2.0, "output_usd_per_1m": 10.0}


def _argv(frozen: Path, tmp_path: Path, arm: str) -> list[str]:
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"prices": PRICES, "max_cost_usd": 1.0, "max_wall_s": 60.0}))
    return [
        "--profile", str(PROFILE_PATH), "--frozen", str(frozen), "--run-dir", str(tmp_path / "run"),
        "--arms", arm, "--ids", "R01", "--reps", "1", "--settings", str(settings),
    ]


def _record_clients(events: list[Any]) -> Any:
    def live_clients(harness: SimpleNamespace, *_: Any) -> tuple[Any, dict[str, Any]]:
        events.append("clients")
        return FakeModel(harness, [R01_SQL]), {"S": FakeExecutor(harness, ((GOLD,),))}

    return live_clients


def test_arm_s_without_its_inputs_is_refused_by_name(
    frozen: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[Any] = []
    monkeypatch.setattr(runner, "live_clients", _record_clients(events))
    with pytest.raises(ValueError, match="arm S needs --neo4j-uri"):
        runner.main(_argv(frozen, tmp_path, "S"))
    assert events == []


def test_a_routing_violation_stops_the_run_before_any_client_is_built(
    frozen: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[Any] = []
    monkeypatch.setattr(runner, "live_clients", _record_clients(events))
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.example.com")
    with pytest.raises(RoutingError):
        runner.main(_argv(frozen, tmp_path, "A"))
    assert events == []


def test_arm_s_runs_inside_the_provider_opened_from_explicit_inputs(
    frozen: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[Any] = []

    @contextmanager
    def open_live_provider(
        profile: Any, neo4j_uri: str, manifest: Path, projection: Path, receipt: Path,
        *, concept_spec: Path | None,
    ) -> Iterator[Any]:
        events.append((neo4j_uri, manifest.name, projection.name, receipt.name, concept_spec))
        yield s_provider()
        events.append("closed")

    monkeypatch.setattr(runner_live_utils, "open_live_provider", open_live_provider)
    monkeypatch.setattr(runner, "require_clean_process", lambda *_: events.append("clean"))
    monkeypatch.setattr(runner, "live_clients", _record_clients(events))
    assert runner.main([*_argv(frozen, tmp_path, "S"), *S_INPUTS]) == 0
    opened = ("bolt://localhost:1", "build.json", "projection.json", "receipt.json", None)
    assert events == ["clean", opened, "clients", "closed"]
    receipts = (tmp_path / "run/receipts.jsonl").read_text().splitlines()
    assert [(r["outcome"], r["context_identity"]) for r in map(json.loads, receipts)] == [
        ("correct", "s-identity-1")
    ]

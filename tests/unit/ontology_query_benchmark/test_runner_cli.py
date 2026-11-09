import itertools
import json
from collections.abc import Callable, Iterator
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
PINS = {"main.s.person": 7, "main.s.visit": 3}


def _argv(frozen: Path, tmp_path: Path, arm: str) -> list[str]:
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"prices": PRICES, "max_cost_usd": 1.0, "max_wall_s": 60.0}))
    return [
        "--profile", str(PROFILE_PATH), "--frozen", str(frozen), "--run-dir", str(tmp_path / "run"),
        "--arms", arm, "--ids", "R01", "--reps", "1", "--settings", str(settings),
    ]


class HistoryExecutor(FakeExecutor):
    """Answers DESCRIBE HISTORY with the version ``versions`` gives; None is a denied read."""

    def __init__(self, harness: SimpleNamespace, versions: Callable[[str], int | None]) -> None:
        super().__init__(harness, ((GOLD,),))
        self.versions = versions

    def execute(self, sql: str, row_limit: int) -> Any:
        if not sql.startswith("DESCRIBE HISTORY "):
            return super().execute(sql, row_limit)
        version, execution = self.versions(sql.split()[2]), self.harness.models.Execution
        if version is None:
            return execution(False, (), (), 0.01, "permission", "denied")
        return execution(True, ("version", "timestamp"), ((version, "t"),), 0.01)


def _record_clients(
    events: list[Any], versions: Callable[[str], int | None] = PINS.get
) -> Any:
    def live_clients(harness: SimpleNamespace, arms: list[str], *_: Any) -> tuple[Any, Any]:
        events.append("clients")
        model = FakeModel(harness, [R01_SQL])
        events.append(model)
        return model, {arm: HistoryExecutor(harness, versions) for arm in arms}

    return live_clients


def _receipts(tmp_path: Path) -> list[dict[str, Any]]:
    path = tmp_path / "run/receipts.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.is_file() else []


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
        *, concept_spec: Path | None, limits: Any = None,
    ) -> Iterator[Any]:
        events.append((neo4j_uri, manifest.name, projection.name, receipt.name, concept_spec))
        yield s_provider()
        events.append("closed")

    monkeypatch.setattr(runner_live_utils, "open_live_provider", open_live_provider)
    monkeypatch.setattr(runner, "require_clean_process", lambda *_: events.append("clean"))
    monkeypatch.setattr(runner, "live_clients", _record_clients(events))
    assert runner.main([*_argv(frozen, tmp_path, "S"), *S_INPUTS]) == 0
    opened = ("bolt://localhost:1", "build.json", "projection.json", "receipt.json", None)
    assert [e for e in events if not isinstance(e, FakeModel)] == [
        "clean", opened, "clients", "closed"
    ]
    receipts = (tmp_path / "run/receipts.jsonl").read_text().splitlines()
    assert [(r["outcome"], r["context_identity"]) for r in map(json.loads, receipts)] == [
        ("correct", "s-identity-1")
    ]


def test_an_overriding_ceiling_sizes_the_s_provider_from_the_run_limits(
    frozen: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened_with: list[Any] = []

    @contextmanager
    def open_live_provider(*_: Any, concept_spec: Path | None, limits: Any = None) -> Iterator[Any]:
        opened_with.append(limits)
        yield s_provider()

    monkeypatch.setattr(runner_live_utils, "open_live_provider", open_live_provider)
    monkeypatch.setattr(runner, "require_clean_process", lambda *_: None)
    monkeypatch.setattr(runner, "live_clients", _record_clients([]))
    argv = [*_argv(frozen, tmp_path, "S"), *S_INPUTS, "--input-ceiling", "36000"]
    assert runner.main(argv) == 0
    assert [limits.input_ceiling for limits in opened_with] == [36000]


@pytest.mark.parametrize(
    ("versions", "refusal"),
    [
        ({"main.s.person": 8, "main.s.visit": 3}.get, r"main\.s\.person\D+8\D+7\b"),
        ({"main.s.person": None, "main.s.visit": 3}.get, r"main\.s\.person.*denied"),
    ],
)
def test_a_source_off_its_pin_refuses_the_run_before_any_model_call(
    frozen: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    versions: Callable[[str], int | None], refusal: str,
) -> None:
    events: list[Any] = []
    monkeypatch.setattr(runner, "require_clean_process", lambda *_: None)
    monkeypatch.setattr(runner, "live_clients", _record_clients(events, versions))
    with pytest.raises(RuntimeError, match=refusal):
        runner.main(_argv(frozen, tmp_path, "A"))
    assert [model.calls for model in events if isinstance(model, FakeModel)] == [[]]
    assert _receipts(tmp_path) == []


def test_a_source_change_mid_run_stops_and_writes_the_rest_unrun(
    frozen: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reads = itertools.count()

    def versions(relation: str) -> int:
        return PINS[relation] + (next(reads) >= len(PINS))

    events: list[Any] = []
    monkeypatch.setattr(runner, "require_clean_process", lambda *_: None)
    monkeypatch.setattr(runner, "live_clients", _record_clients(events, versions))
    assert runner.main([*_argv(frozen, tmp_path, "A"), "--reps", "3"]) == 0
    receipts = _receipts(tmp_path)
    slots = (tmp_path / "run/slots.jsonl").read_text().splitlines()
    assert len(receipts) == len(slots) == 3
    assert [(r["outcome"], r.get("reason")) for r in receipts] == [
        ("correct", None), ("unrun", "source_drift"), ("unrun", "source_drift")
    ]
    assert [len(model.calls) for model in events if isinstance(model, FakeModel)] == [1]


def test_the_manifest_records_the_source_versions_the_run_read(
    frozen: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "require_clean_process", lambda *_: None)
    monkeypatch.setattr(runner, "live_clients", _record_clients([]))
    assert runner.main(_argv(frozen, tmp_path, "A")) == 0
    manifest = json.loads((tmp_path / "run/manifest.json").read_text())
    assert manifest["source_versions"] == PINS
    assert [r["outcome"] for r in _receipts(tmp_path)] == ["correct"]

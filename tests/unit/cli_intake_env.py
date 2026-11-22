"""Shared setup for the `sema target discover|approve|runs` CLI tests.

The commands run through the top-level ``sema`` group, so the tests hold wherever the
commands are registered. The model is the routed mock from ``graph_env``, handed to the
CLI by replacing ``sema.cli_intake._intake_client``; the intake config comes from the
``INTAKE_MODEL_*`` environment, as a user's shell would set it.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from itertools import takewhile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import yaml
from click.testing import CliRunner, Result

from sema.binding.spec import ConceptSourceSpec, load_concept_source_spec
from sema.intake.calls.ledger import Ledger
from sema.intake.run.metrics import RunMetrics
from sema.intake.calls.registry import Registry
from sema.log import logger
from tests.unit.intake.decisions_env import open_entries, presented, write_decisions
from tests.unit.intake.graph_env import (
    ACRIS_DESCRIPTOR,
    LAYOUT_PROPOSAL,
    MAX_CALLS,
    REGISTRY_FILE,
    GraphRun,
    asks_for_values,
    fills_down,
    graph,
    mixed_source,
    routed_client,
)
from tests.unit.intake.model_calls_env import INPUT_PER_1K, MODEL, OUTPUT_PER_1K
from tests.unit.intake.structure_env import A2_WORKBOOK, copy_code_tables

STRUCTURE, ROLES = "structure", "roles"
CLIENT_FACTORY = "sema.cli_intake._intake_client"
DOTENV_LOADER = "dotenv.load_dotenv"
APPROVER = "reviewer-cli"
MAX_COST = 1.0
INTAKE_ENV = {
    "INTAKE_MODEL_PROVIDER": "openrouter",
    "INTAKE_MODEL_MODEL": MODEL,
    "INTAKE_MODEL_MAX_CALLS": str(MAX_CALLS),
    "INTAKE_MODEL_MAX_COST": str(MAX_COST),
    "INTAKE_MODEL_PRICES": json.dumps(
        {MODEL: {"input_per_1k": INPUT_PER_1K, "output_per_1k": OUTPUT_PER_1K}}
    ),
    "INTAKE_TRACING_BACKEND": "none",
}
CLEARED_ENV = ("INTAKE_MODEL_RECORDINGS_DIR",)

RUN_KEYS = (
    "run_id", "source", "stage", "status", "reason", "approver",
    "model_calls", "tokens", "cost_estimate", "started_at", "updated_at",
)
OLD_RUN, MID_RUN, NEW_RUN = "run-old", "run-mid", "run-new"
SOURCE_A, SOURCE_B = "source-a", "source-b"
SEEDED = (
    (OLD_RUN, SOURCE_A, "roles", "running", None),
    (MID_RUN, SOURCE_B, "review", "waiting_review", "budget_exceeded"),
    (NEW_RUN, SOURCE_A, "import", "approved", None),
)
A2_SPEC = A2_WORKBOOK.with_name("codes_a2.spec.yaml")
A2_SCHEMA = "land_a2"
FIXED_RUN_ID = "cli-fixed-run"
DISCOVER_IDS = ["defaults", "every-option"]
DISCOVER_OPTIONS = [
    ((), {"hint": None, "schema": None, "tier": "metadata_only", "enrich": False}),
    (
        (
            "--hint", A2_SPEC, "--schema", A2_SCHEMA, "--tier", "bounded_values",
            "--enrich", "--run-id", FIXED_RUN_ID,
        ),
        {
            "hint": A2_SPEC, "schema": A2_SCHEMA, "tier": "bounded_values",
            "enrich": True, "run_id": FIXED_RUN_ID,
        },
    ),
]
SEEDED_METRICS = {
    "relations": 3, "ignored": {"empty": 1}, "bound": 2, "bound_fraction": 2 / 3,
    "open_structure": 1, "open_roles": 0, "proposals_made": 0, "proposals_bound": 0,
    "model_calls": 4, "tokens": 6000, "cost": 0.024, "value_asks": 0,
}


@dataclass
class Cli:
    """One ledger root, the mocked model the CLI receives, and the configs it built."""

    root: Path
    client: MagicMock
    configs: list[Any] = field(default_factory=list)

    def run(self, *args: str | Path) -> Result:
        command, *rest = (str(arg) for arg in args)
        argv = ["target", command, *rest, "--root", str(self.root)]
        from sema.cli import cli

        return CliRunner().invoke(cli, argv)

    def ok(self, *args: str | Path) -> Any:
        result = self.run(*args)
        assert result.exit_code == 0, result.output
        return json.loads(result.stdout)

    def graph_run(self, run_id: str) -> GraphRun:
        return GraphRun(self.root, run_id, self.client)

    def discover(self, source: Path, descriptor: Path, *options: str | Path) -> GraphRun:
        printed = self.ok("discover", source, "--descriptor", descriptor, *options)
        return self.graph_run(printed["run_id"])

    def approve(self, run: GraphRun, *options: str | Path) -> dict[str, Any]:
        return self.ok("approve", run.run_id, "--approver", APPROVER, *options)

    def reviewed(self, source: Path, descriptor: Path, *options: str | Path) -> GraphRun:
        """Discover a source whose Roles asks for values once, approve them with
        ``--values``, and stop at Review."""
        asked = self.ok("discover", source, "--descriptor", descriptor, *options)
        assert asked["stopped_at"] == "value_ask", asked
        run = self.graph_run(asked["run_id"])
        printed = self.approve(run, "--values")
        assert (printed["stopped_at"], printed["run_metrics"]["value_asks"]) == ("review", 1)
        return run


def intake_cli(root: Path, monkeypatch: pytest.MonkeyPatch, *, model: bool = True) -> Cli:
    """``model=False`` for commands that read only the registry and ledger."""
    for name, value in INTAKE_ENV.items():
        monkeypatch.setenv(name, value)
    for name in CLEARED_ENV:
        monkeypatch.delenv(name, raising=False)
    # The sema group loads .env on every call; a developer's .env must not refill the env.
    monkeypatch.setattr(DOTENV_LOADER, lambda *args, **kwargs: False)
    cli = Cli(root, routed_client())
    if not model:
        return cli

    def _client(config: Any) -> MagicMock:
        cli.configs.append(config)
        return cli.client

    monkeypatch.setattr(CLIENT_FACTORY, _client)
    return cli


def _ticks() -> Iterator[datetime]:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    return (start + timedelta(minutes=step) for step in range(1000))


def seed_runs(root: Path) -> Registry:
    """Three runs registered a minute apart, oldest first, one stopped by the cap."""
    ticks = _ticks()
    registry = Registry(root / REGISTRY_FILE, clock=lambda: next(ticks))
    for run_id, source, stage, status, reason in SEEDED:
        registry.register(run_id, source=source, stage="land")
        registry.advance(run_id, stage, status, reason)
    Ledger(root).run(MID_RUN).artifact("run_metrics", RunMetrics(**SEEDED_METRICS))
    return registry


def listed_ids(printed: Sequence[dict[str, Any]]) -> list[str]:
    return [row["run_id"] for row in printed]


def snapshot(run: GraphRun) -> tuple[Any, int, list[str]]:
    """The registry row, the ledger length, and every file under the run's import dir."""
    imported = run.ledger_dir / "import"
    files = sorted(
        f"{path.relative_to(imported)}:{hashlib.sha256(path.read_bytes()).hexdigest()}"
        for path in imported.rglob("*")
        if path.is_file()
    )
    return run.row(), run.mark(), files


def open_listed(run: GraphRun, stage: str) -> list[Any]:
    """Open entries of one stage as Review presented them in the run's ledger."""
    return open_entries(presented(Ledger(run.root).run(run.run_id)), stage)


def open_count(run: GraphRun) -> int:
    return len(open_listed(run, STRUCTURE)) + len(open_listed(run, ROLES))


def decide(cli: Cli, run: GraphRun, rows: list[tuple[str, str, str]], name: str) -> Any:
    """One decisions round through the CLI; it must reach the provider zero times."""
    path = write_decisions(cli.root.parent / name, rows)
    cli.client = routed_client()
    printed = cli.approve(run, "--decisions", path)
    assert cli.client.invoke.call_count == 0
    return printed


def asked_for_values(cli: Cli) -> tuple[GraphRun, dict[str, Any]]:
    """A run whose layout proposal asks for values, stopped at the value interrupt."""
    cli.client = routed_client({LAYOUT_PROPOSAL: [asks_for_values, fills_down]})
    source = mixed_source(cli.root.parent)
    printed = cli.ok("discover", source, "--descriptor", ACRIS_DESCRIPTOR)
    return cli.graph_run(printed["run_id"]), printed


def descriptor_dir(root: Path, count: int) -> Path:
    """A source directory holding ``count`` run descriptors beside the code tables."""
    directory = copy_code_tables(root / f"descriptors_{count}")
    for index in range(count):
        shutil.copy(ACRIS_DESCRIPTOR, directory / f"run_{index}.run.yaml")
    return directory


def spy_discover(
    monkeypatch: pytest.MonkeyPatch, drop: tuple[str, ...] = ()
) -> list[tuple[Path, dict[str, Any]]]:
    """Record each ``graph.discover`` call the CLI makes, then run the real one without ``drop``."""
    module = graph()
    real, seen = module.discover, []

    def _spy(source: Path, **kwargs: Any) -> Any:
        seen.append((Path(source), kwargs))
        kept = {name: value for name, value in kwargs.items() if name not in drop}
        return real(source, **kept)

    monkeypatch.setattr(module, "discover", _spy)
    return seen


def zero_cap_replay(monkeypatch: pytest.MonkeyPatch, recordings: Path, max_calls: int) -> None:
    """The replay variables: a call cap and an existing recordings directory."""
    recordings.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("INTAKE_MODEL_MAX_CALLS", str(max_calls))
    monkeypatch.setenv("INTAKE_MODEL_RECORDINGS_DIR", str(recordings))


@contextmanager
def captured_logs(level: str = "INFO") -> Iterator[list[str]]:
    """Every ``sema.log.logger`` message at ``level`` or above while the block runs."""
    messages: list[str] = []
    sink_id = logger.add(lambda message: messages.append(message.record["message"]), level=level)
    try:
        yield messages
    finally:
        logger.remove(sink_id)


def ok_or_raise(result: Result) -> Any:
    """A command's printed JSON; an exception the command raised is re-raised as is."""
    if result.exception is not None and not isinstance(result.exception, SystemExit):
        raise result.exception
    assert result.exit_code == 0, result.output
    return json.loads(result.stdout)


SPEC_FILE = "spec.yaml"
SPEC_PATH_KEY = "spec_path"
DEFAULT_SPECS_DIR = "specs"


def descriptor_id(descriptor: Path) -> str:
    return str(yaml.safe_load(descriptor.read_text())["id"])


def default_specs_dir(cli: Cli) -> Path:
    """The default spec output folder: ``specs`` beside the ledger root ``.sema/intake``."""
    return cli.root.parent / DEFAULT_SPECS_DIR


def spec_file(out: Path, descriptor: Path) -> Path:
    return out / descriptor_id(descriptor) / SPEC_FILE


def files_under(directory: Path) -> list[Path]:
    return sorted(path for path in directory.rglob("*") if path.is_file())


def header_of(path: Path) -> str:
    """The comment lines the written spec opens with."""
    lines = path.read_text().splitlines()
    header = list(takewhile(lambda line: line.startswith("#"), lines))
    assert header, f"{path} opens with no header comment"
    return "\n".join(header)


def printed_path(printed: dict[str, Any]) -> Path:
    return Path(printed[SPEC_PATH_KEY]).resolve()


def same_spec(written: Path, artifact: Any) -> bool:
    """The written file loads, with the project's loader, to the ledger's ``spec`` artifact."""
    loaded = load_concept_source_spec(written)
    held = ConceptSourceSpec.model_validate(artifact, context={"base_dir": written.parent})
    return bool(loaded.model_dump(mode="json") == held.model_dump(mode="json"))

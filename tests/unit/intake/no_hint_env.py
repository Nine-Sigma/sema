"""Replays of the committed intake recordings for the no-hint acceptance (task 14.5).

Each source runs through ``scripts.record_intake_run.verify_replay``: no hint, no
decisions, enrichment off, ``max_calls=0``, and an offline client, so a recorded call that
misses is refused and fails the run. Value asks are approved the way the recording was.
Held-out sources are passed by path only; nothing here opens a file under one.
"""

from __future__ import annotations

import csv
import difflib
import importlib
import io
import json
import shutil
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock

import pytest
import yaml

from sema.binding.importer import import_concept_source
from sema.binding.spec import ConceptSourceSpec
from sema.binding.spec_sections import TermStorageSpec
from sema.ingest.duckdb_staging import Staging
from sema.intake import graph as intake_graph
from sema.intake.graph_stages import REGISTRY_FILE, STAGING_FILE
from sema.intake.ledger import Ledger
from sema.intake.metrics import RunMetrics
from sema.intake.model_calls import BUDGET_REASON
from sema.intake.registry import Registry, RunRecord
from sema.intake.scorecard import Scorecard, score
from sema.intake.relations import RelationProvider
from sema.models.config import IntakeModelConfig, ModelPrice
from tests.support.athena_bundle import export_schema
from tests.unit.intake.model_calls_env import intake_config

APPROVER = "dean"
VALUE_APPROVAL = "value_approval"
AUDITED_STAGES = frozenset({"structure", "roles"})
DECISIONS_FILE = "decisions.yaml"
RECORDED_CALL_CAP = 200
RECORDED_COST_CAP = 2.0
PRICE = ModelPrice(input_per_1k=0.0001, output_per_1k=0.0005)


def script() -> ModuleType:
    return importlib.import_module("scripts.record_intake_run")


def utils() -> ModuleType:
    return importlib.import_module("scripts.record_intake_run_utils")


def recording(name: str) -> dict[str, Any]:
    manifest: dict[str, Any] = json.loads(
        (utils().RECORDINGS_DIR / name / utils().MANIFEST_FILE).read_text()
    )
    return manifest


def source_of(name: str) -> Any:
    (source,) = [s for s in script().RECORDED_SOURCES if s.name == name]
    return source


@dataclass
class Replayed:
    name: str
    root: Path
    outcome: intake_graph.RunOutcome
    config: IntakeModelConfig
    client: MagicMock
    approved: list[str] = field(default_factory=list)

    @property
    def ledger_dir(self) -> Path:
        return Ledger(self.root).run(self.outcome.run_id).directory

    def row(self) -> RunRecord:
        return Registry(self.root / REGISTRY_FILE).get(self.outcome.run_id)

    def replay_config(self) -> IntakeModelConfig:
        update = {"recordings_dir": utils().RECORDINGS_DIR / self.name, "max_calls": 0}
        return self.config.model_copy(update=update)


def _offline_client() -> MagicMock:
    client = MagicMock()
    client.invoke.side_effect = AssertionError("a recorded call missed and reached the client")
    return client


@contextmanager
def _value_asks_seen(approved: list[str]) -> Iterator[None]:
    """Collect the relations every value ask of the run requests, as it is answered."""

    def seen(outcome: intake_graph.RunOutcome) -> intake_graph.RunOutcome:
        if outcome.node == "value_ask":
            approved.extend(r["relation"] for r in (outcome.payload or {})["requests"])
        return outcome

    discover, approve_values = intake_graph.discover, intake_graph.approve_values
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(intake_graph, "discover", lambda *a, **k: seen(discover(*a, **k)))
        patch.setattr(
            intake_graph, "approve_values", lambda *a, **k: seen(approve_values(*a, **k))
        )
        yield


def replay(name: str, root: Path) -> Replayed:
    model = recording(name)["model"]
    config = intake_config(model=model, max_calls=RECORDED_CALL_CAP, prices={model: PRICE})
    client, approved = _offline_client(), []
    with _value_asks_seen(approved):
        outcome = script().verify_replay(
            source_of(name),
            recordings=utils().RECORDINGS_DIR,
            workdir=root,
            config=config,
            client=client,
            approver=recording(name).get("approver"),
        )
    return Replayed(name, root, outcome, config, client, approved)


def decide(run: Replayed, decisions: Path) -> intake_graph.RunOutcome:
    return intake_graph.approve(
        run.outcome.run_id,
        root=run.root,
        config=run.replay_config(),
        approver=APPROVER,
        client=run.client,
        decisions=decisions,
    )


def spec_of(outcome: intake_graph.RunOutcome) -> ConceptSourceSpec | None:
    raw = (outcome.payload or {}).get("spec")
    return ConceptSourceSpec.model_validate(raw) if raw is not None else None


def metrics_of(outcome: intake_graph.RunOutcome) -> RunMetrics:
    return RunMetrics.model_validate((outcome.payload or {})["run_metrics"])


def spec_diff(assembled: ConceptSourceSpec | None, written: ConceptSourceSpec) -> str:
    def dump(spec: ConceptSourceSpec | None) -> str:
        raw = spec.model_dump(mode="json", by_alias=True, exclude_none=True) if spec else {}
        return yaml.safe_dump(raw, sort_keys=True)

    return text_diff(dump(written), dump(assembled))


def text_diff(expected: str, actual: str) -> str:
    lines = difflib.unified_diff(
        expected.splitlines(), actual.splitlines(), "written", "assembled", lineterm=""
    )
    return "\n".join(lines)


def capped(spec: ConceptSourceSpec, cap: int | None) -> ConceptSourceSpec:
    if cap is None:
        return spec
    storage = spec.term_storage.model_dump() | {"inline_cap": cap}
    return spec.model_copy(update={"term_storage": TermStorageSpec.model_validate(storage)})


@contextmanager
def staging_copy(run: Replayed, directory: Path) -> Iterator[Staging]:
    """The run's landed staging file, copied so imports never write into the run."""
    directory.mkdir(parents=True, exist_ok=True)
    copy = directory / STAGING_FILE
    shutil.copy2(run.ledger_dir / STAGING_FILE, copy)
    with Staging(str(copy), schemas=()) as staging:
        yield staging


def imported(spec: ConceptSourceSpec, staging: Staging) -> tuple[dict[str, Any], str, str]:
    """The import's manifest mapping with entities in qualified-name order, that mapping
    as text, and the term staging export text."""
    provider = RelationProvider(warehouse=None, staging=staging)
    result = import_concept_source(spec, provider, dataset_map={}, staging=staging)
    entities = sorted(result.manifest["entities"], key=lambda e: e["qualified_name"])
    manifest = result.manifest | {"entities": entities}
    text = yaml.safe_dump(manifest, sort_keys=False)
    return manifest, text, export_schema(staging, spec.external_namespace)


def jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _reads(statement: str, relation: str) -> bool:
    schema, _, name = relation.partition(".")
    return f'"{schema}"."{name}"' in statement


def uncovered_reads(run: Replayed) -> list[tuple[str, str]]:
    """Value-bearing structure and roles statements neither over a schema relation nor
    issued after a logged value approval of a relation they read."""
    approvals = [
        e["timestamp"] for e in jsonl(run.ledger_dir / "events.jsonl")
        if e["kind"] == VALUE_APPROVAL
    ]
    first = min(approvals, default=None)
    uncovered = []
    for line in jsonl(run.ledger_dir / "sql.jsonl"):
        if not line["value_bearing"] or line["stage"] not in AUDITED_STAGES:
            continue
        if line["schema_relation"]:
            continue
        approved = any(_reads(line["statement"], r) for r in run.approved)
        if not (approved and first is not None and first <= line["timestamp"]):
            uncovered.append((line["stage"], line["statement"]))
    return uncovered


def csv_blocks(text: str) -> dict[str, list[list[str]]]:
    """An ``export_schema`` text as {table: [typed header, *rows]}."""
    blocks: dict[str, list[list[str]]] = {}
    current: list[list[str]] = []
    for row in csv.reader(io.StringIO(text)):
        if row and row[0].startswith("## "):
            current = blocks.setdefault(row[0].removeprefix("## "), [])
        else:
            current.append(row)
    return blocks


def scorecard_of(
    run: Replayed, spec: ConceptSourceSpec, oracle: ConceptSourceSpec, cap: int | None,
    directory: Path,
) -> Scorecard:
    with staging_copy(run, directory) as staging:
        manifest, _, staged = imported(capped(spec, cap), staging)
        expected_manifest, _, expected_staged = imported(capped(oracle, cap), staging)
    return score(expected_manifest, manifest, expected_staged, staged)


def column(block: list[list[str]], name: str) -> list[str]:
    header, *rows = block
    index = [cell.partition(":")[0] for cell in header].index(name)
    return [row[index] for row in rows]


def marked_param(name: str, reasons: Mapping[str, str], **xfail: Any) -> Any:
    reason = reasons.get(name)
    marks = () if reason is None else pytest.mark.xfail(strict=True, reason=reason, **xfail)
    return pytest.param(name, marks=marks)


def assert_zero_provider_calls(run: Replayed) -> None:
    row = run.row()
    calls, reason, invoked = row.model_calls, row.reason, run.client.invoke.call_count

    assert (calls, invoked) == (0, 0)
    assert reason != BUDGET_REASON

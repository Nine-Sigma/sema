"""Shared setup for the ingestion graph tests: a routed mocked model, run readers, sources.

``sema.intake.graph`` is imported lazily, so a missing module fails each test on its own.
The mocked client answers each call by its output model's name: batch models abstain on
every entry unless scripted, layout proposals abstain unless scripted, the engine's stage B
names each prompt column (``label_for``, ``synonym_for``) and stage C decodes nothing, and
any other model fails the test. ``ProcessExit`` stands in for the process dying inside a node.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock

from pydantic import BaseModel

from sema.ingest.duckdb_staging import Staging
from sema.intake.output_models import ROLE_MAP_PREFIX
from sema.intake.registry import Registry, RunRecord
from sema.models.stages import StageBBatchResult, StageBColumnResult, StageCBatchResult
from tests.unit.intake.enrichment_env import (
    CONFIDENCE,
    SEMANTIC_TYPE,
    label_for,
    prompt_columns,
    synonym_for,
)
from tests.unit.intake.layout_proposal_env import CODEBOOK_CSV, abstain, answer, write_codebook
from tests.unit.intake.model_calls_env import intake_config, stats
from tests.unit.intake.structure_env import A2_WORKBOOK, ACRIS, copy_code_tables

A2_DESCRIPTOR = A2_WORKBOOK.parent / "codes_a2.run.yaml"
ACRIS_DESCRIPTOR = ACRIS / "acris.run.yaml"
REGISTRY_FILE = "registry.sqlite"
CHECKPOINT_FILE = "checkpoints.sqlite"
IMPORT_DIR = "import"
MAX_CALLS = 200
APPROVER = "reviewer-a"
VALUE_ASK, REVIEW = "value_ask", "review"

STRUCTURE_BATCH = "StructureBatch_v1"
ROLE_MAP = ROLE_MAP_PREFIX
REFERENCE_CHOICE = "ReferenceChoice_v1"
LAYOUT_PROPOSAL = "LayoutProposal_v1"
CHOICE_PREFIXES = ("entry_", "role_", "reference_")
REASONING = ("confidences", "reasons")
FLAT_LIST = "concept_list("
DENORMALIZED = "concept_with_values_denormalized"
FILL_DOWN = "fill_down"
VARIABLE = "variable"
ASKED_COLUMNS = (VARIABLE, "label")
CODEBOOK = "codebook"

STAGE_NODES = (
    "land", "structure", "resolve_structure", "propose_layouts", "roles", "resolve_roles",
    "enrich", "dry_run", "decide", "import",
)
REVIEW_KEYS = ("ambiguities", "proposals", "spec", "dry_run", "run_metrics", "ledger")

Answer = Callable[[type[BaseModel]], BaseModel]


class ProcessExit(BaseException):  # noqa: N818 - it stands in for SystemExit, not an error
    """Raised by the mocked client where the process would die; nothing may catch it."""


def graph() -> ModuleType:
    return importlib.import_module("sema.intake.graph")


def enum(model: type[BaseModel], field: str) -> tuple[str, ...]:
    """The members a field offers; a role-map field is free text and offers none."""
    schema = model.model_json_schema()["properties"][field]
    return tuple(schema.get("items", schema).get("enum", ()))


def batch_answer(model: type[BaseModel], pick: Callable[[tuple[str, ...]], str]) -> BaseModel:
    fields = [name for name in model.model_fields if name.startswith(CHOICE_PREFIXES)]
    values: dict[str, Any] = {name: pick(enum(model, name)) for name in fields}
    values |= {"confidences": [0.6] * len(fields), "reasons": ["read from names"] * len(fields)}
    return model(**values)


def abstain_all(model: type[BaseModel]) -> BaseModel:
    return batch_answer(model, lambda _members: "abstain")


def choose_flat_lists(model: type[BaseModel]) -> BaseModel:
    """Every structure entry settled on its flat concept list member."""
    return choosing(lambda member: member.startswith(FLAT_LIST))(model)


def choosing(wanted: Callable[[str], bool]) -> Answer:
    """An answer settling every structure entry on its first wanted member, if it has one."""

    def _pick(members: tuple[str, ...]) -> str:
        return next((member for member in members if wanted(member)), "abstain")

    return lambda model: batch_answer(model, _pick)


def asks_for_values(model: type[BaseModel]) -> BaseModel:
    return answer(model, kind="needs_values", needs_values=ASKED_COLUMNS)


def fills_down(model: type[BaseModel]) -> BaseModel:
    (relation,) = [r for r in enum(model, "relations") if r.rpartition(".")[2] == CODEBOOK]
    return answer(
        model,
        template=DENORMALIZED,
        relations=[relation],
        view=FILL_DOWN,
        view_columns=[VARIABLE],
        slots={"group_by": VARIABLE},
    )


DEFAULTS: dict[str, Answer] = {
    STRUCTURE_BATCH: abstain_all,
    ROLE_MAP: abstain_all,
    REFERENCE_CHOICE: abstain_all,
    LAYOUT_PROPOSAL: lambda model: abstain()(model),
}


def engine_answer(prompt: str, output_model: type[BaseModel]) -> BaseModel | None:
    """Enrichment's stage B and C answers, read from the prompt as ``enrichment_env`` does."""
    if issubclass(output_model, StageBBatchResult):
        columns = [
            StageBColumnResult(
                column=name,
                canonical_property_label=label_for(name),
                semantic_type=SEMANTIC_TYPE,
                confidence=CONFIDENCE,
                synonyms=[synonym_for(name)],
            )
            for name in prompt_columns(prompt)
        ]
        return output_model(columns=columns)
    if issubclass(output_model, StageCBatchResult):
        return output_model(columns=[])
    return None


def routed_client(
    script: Mapping[str, Sequence[Answer]] | None = None,
    *,
    defaults: Mapping[str, Answer] | None = None,
    exit_on: str | None = None,
) -> MagicMock:
    """A mocked ``LLMClient``; ``exit_on`` names the model whose first call kills the process.
    Every role-map model, whatever its name, is routed as ``ROLE_MAP``."""
    client = MagicMock()
    client.last_stats = stats()
    client.last_path = None
    pending = {name: list(answers) for name, answers in (script or {}).items()}
    fallback = {**DEFAULTS, **(defaults or {})}

    def _invoke(prompt: str, output_model: type[BaseModel], **_: Any) -> BaseModel:
        name = output_model.__name__
        name = ROLE_MAP if name.startswith(ROLE_MAP_PREFIX) else name
        if name == exit_on:
            raise ProcessExit(name)
        client.last_stats = stats()
        client.last_path = "structured_output"
        engine = engine_answer(prompt, output_model)
        if engine is not None:
            return engine
        assert name in fallback, f"unexpected model call for {name}"
        queue = pending.get(name, [])
        return (queue.pop(0) if queue else fallback[name])(output_model)

    client.invoke.side_effect = _invoke
    return client


def model_names(client: MagicMock) -> list[str]:
    return [call.args[1].__name__ for call in client.invoke.call_args_list]


def code_tables(root: Path) -> Path:
    return copy_code_tables(root / "inputs" / "code_tables")


def mixed_source(root: Path) -> Path:
    """The code tables beside a codebook only a fill-down view reads."""
    return write_codebook(copy_code_tables(root / "inputs" / "mixed"), CODEBOOK_CSV)


def settings(root: Path) -> dict[str, Any]:
    return {"root": root, "config": intake_config(max_calls=MAX_CALLS)}


@dataclass
class GraphRun:
    root: Path
    run_id: str
    client: MagicMock
    outcome: Any = None

    @property
    def ledger_dir(self) -> Path:
        return self.root / self.run_id

    def row(self) -> RunRecord:
        return Registry(self.root / REGISTRY_FILE).get(self.run_id)

    def events(self, kind: str | None = None) -> list[dict[str, Any]]:
        path = self.ledger_dir / "events.jsonl"
        lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
        found = [json.loads(line) for line in lines if line.strip()]
        return [event for event in found if kind is None or event["kind"] == kind]

    def mark(self) -> int:
        return len(self.events())

    def steps(self, since: int = 0) -> list[tuple[str, str]]:
        """Node entries and verdicts in ledger order, as ``(kind, stage)``."""
        later = self.events()[since:]
        return [(e["kind"], e["stage"]) for e in later if e["kind"] in ("node", "verdict")]

    def nodes(self, since: int = 0) -> list[str]:
        return [stage for kind, stage in self.steps(since) if kind == "node"]

    def verdicts(self, stage: str) -> list[dict[str, Any]]:
        return [e["payload"] for e in self.events("verdict") if e["stage"] == stage]

    def replayed(self, since: int = 0) -> list[str]:
        later = self.events()[since:]
        return [e["payload"]["schema_name"] for e in later if e["kind"] == "replay"]

    def artifact(self, name: str) -> Any:
        return json.loads((self.ledger_dir / "artifacts" / f"{name}.json").read_text())

    def has_artifact(self, name: str) -> bool:
        return (self.ledger_dir / "artifacts" / f"{name}.json").is_file()

    def counters(self) -> tuple[int, int, float]:
        row = self.row()
        return row.model_calls, row.tokens, row.cost


def discover(
    root: Path, source: Path, descriptor: Path, client: MagicMock, **options: Any
) -> GraphRun:
    outcome = graph().discover(
        source, descriptor=descriptor, client=client, **settings(root), **options
    )
    return GraphRun(root, outcome.run_id, client, outcome)


def through_values(run: GraphRun, **options: Any) -> Any:
    """Approves the one value ask a ``metadata_only`` run stops at, then continues.

    A source with a settled status column asks for its values before Roles resolves
    (ingestion-graph: interrupts are nodes); ``options`` override ``settings`` (a cap).
    """
    assert run.outcome.node == VALUE_ASK, run.outcome.node
    run.outcome = graph().approve_values(
        run.run_id, client=run.client, **{**settings(run.root), **options}
    )
    assert run.outcome.node != VALUE_ASK, "the run asked for values twice"
    if run.outcome.node == REVIEW:
        assert run.artifact("run_metrics")["value_asks"] == 1
    return run.outcome


def approve(run: GraphRun, client: MagicMock | None = None, **options: Any) -> Any:
    run.client = client or run.client
    run.outcome = graph().approve(
        run.run_id, approver=APPROVER, client=run.client, **settings(run.root), **options
    )
    return run.outcome


def subsequence(found: Sequence[Any], wanted: Sequence[Any]) -> bool:
    remaining = iter(found)
    return all(any(item == step for step in remaining) for item in wanted)


def staged_namespaces(path: Path, namespace: str) -> tuple[list[str], int]:
    """The staging file's schemas named after ``namespace``, and its concept rows there."""
    with Staging(str(path), schemas=()) as staging:
        named = [s for s in staging.list_all_schemas() if s.startswith(namespace)]
        tables = staging.execute(
            "SELECT table_name FROM information_schema.tables"
            " WHERE table_schema = ? AND table_name = 'concept'",
            [namespace],
        ).fetchall()
        if not tables:
            return named, 0
        row = staging.execute(f'SELECT count(*) FROM "{namespace}".concept').fetchone()
    return named, int(row[0]) if row else 0

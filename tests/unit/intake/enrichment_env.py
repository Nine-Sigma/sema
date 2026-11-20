"""Shared setup for the enrichment tests: a run through Roles, a mocked model, readers.

Each run lands a committed fixture, runs Structure and Roles with its hand-written spec as
the hint (so Roles assembles a spec with no model), and wires a real ``ModelCalls`` over a
mocked client that answers the engine's stage B and C output models from the prompt.
``sema.intake.report.enrichment`` is imported inside the helpers so a missing module fails each
test instead of the whole collection.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import yaml

from sema.binding.spec import load_concept_source_spec
from sema.engine import semantic
from sema.engine.semantic import SemanticEngine
from sema.intake.roles.eligibility_spec_utils import RolesResult
from sema.intake.roles.eligibility_utils import RunDescriptor
from sema.llm_client import InvocationStats
from sema.models.stages import StageBBatchResult, StageBColumnResult, StageCBatchResult
from tests.unit.intake.model_calls_env import intake_config, stats
from tests.unit.intake.pipeline_env import StageRun, landed_run, run_stages

FIXTURES = Path(__file__).parents[2] / "fixtures" / "intake"
A2 = FIXTURES / "tabular"
ACRIS = FIXTURES / "acris"
ENRICH_STAGE = "enrich"
ARTIFACT = "enrichment"
SERVED_BY = "structured_output"
STAGE_B_VERSION_LINE = "# prompt: enrich_stage_b v1"
SEMANTIC_TYPE = "categorical"
CONFIDENCE = 0.7
MAX_CALLS = 200
COLUMNS_HEADER = "Columns to classify ("
COLUMN_INDENT = "  "


def label_for(column: str) -> str:
    return f"Model label for {column}"


def synonym_for(column: str) -> str:
    return f"{column} alias"


def prompt_columns(prompt: str) -> list[str]:
    """Column names of a rendered stage B prompt, read from its column lines."""
    lines = prompt.splitlines()
    starts = [i for i, line in enumerate(lines) if line.startswith(COLUMNS_HEADER)]
    if not starts:
        return []
    names: list[str] = []
    for line in lines[starts[0] + 1 :]:
        if not line.strip():
            break
        if line.startswith(COLUMN_INDENT) and not line.startswith(COLUMN_INDENT * 2):
            names.append(line[len(COLUMN_INDENT) :].split(" (", 1)[0])
    return names


def _column(name: str) -> StageBColumnResult:
    return StageBColumnResult(
        column=name,
        canonical_property_label=label_for(name),
        semantic_type=SEMANTIC_TYPE,
        confidence=CONFIDENCE,
        synonyms=[synonym_for(name)],
    )


def fake_engine_client() -> MagicMock:
    """A mocked `LLMClient` that answers stage B from the prompt's columns and refuses A."""
    client = MagicMock()
    client.last_stats = InvocationStats()
    client.last_path = None

    def _invoke(prompt: str, output_model: type[Any], **_kwargs: Any) -> Any:
        client.last_stats = stats()
        client.last_path = SERVED_BY
        if issubclass(output_model, StageBBatchResult):
            return output_model(columns=[_column(n) for n in prompt_columns(prompt)])
        if issubclass(output_model, StageCBatchResult):
            return output_model(columns=[])
        raise AssertionError(f"enrichment asked the model for {output_model.__name__}")

    client.invoke.side_effect = _invoke
    return client


@dataclass
class EnrichRun:
    stages: StageRun
    descriptor: RunDescriptor
    calls: Any
    client: MagicMock
    prompt_kwargs: list[dict[str, Any]] = field(default_factory=list)
    roles_before: RolesResult | None = None

    def enrich(self, **kwargs: Any) -> Any:
        from sema.intake.report.enrichment import enrich

        landed = self.stages.landed
        tools = landed.tools(ENRICH_STAGE, self.stages.structure)
        return enrich(
            self.stages.spec, self.stages.roles, tools, self.calls, landed.ledger, **kwargs
        )

    def enrich_watched(self) -> Any:
        """Enrich with stage A refused and every stage B prompt's arguments captured."""
        self.roles_before = copy.deepcopy(self.stages.roles)
        real = semantic.build_stage_b_prompt

        def _capture(*args: Any, **kwargs: Any) -> str:
            self.prompt_kwargs.append(kwargs)
            return real(*args, **kwargs)

        refuse = AssertionError("stage A must not run during enrichment")
        with (
            patch.object(SemanticEngine, "run_stage_a", side_effect=refuse),
            patch.object(semantic, "build_stage_b_prompt", side_effect=_capture),
        ):
            return self.enrich(enabled=True)

    def prompts(self) -> list[str]:
        return [str(c.args[0]) for c in self.client.invoke.call_args_list]

    def asked_models(self) -> list[str]:
        return [c.args[1].__name__ for c in self.client.invoke.call_args_list]

    def call_records(self) -> dict[str, dict[str, Any]]:
        calls = self.stages.landed.ledger.directory / "calls"
        paths = sorted(calls.glob("*.json")) if calls.is_dir() else []
        return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in paths}

    def events(self, kind: str) -> list[dict[str, Any]]:
        path = self.stages.landed.ledger.directory / "events.jsonl"
        lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
        return [e for e in map(json.loads, lines) if e["kind"] == kind]

    def artifact_path(self) -> Path:
        return self.stages.landed.ledger.directory / "artifacts" / f"{ARTIFACT}.json"


@contextmanager
def enrich_run(source: Path, root: Path, schema: str, stem: str) -> Iterator[EnrichRun]:
    from sema.intake.calls.model_calls import ModelCalls

    directory = source if source.is_dir() else source.parent
    with landed_run(source, root, schema) as landed:
        run = yaml.safe_load((directory / f"{stem}.run.yaml").read_text())
        descriptor = RunDescriptor.model_validate(run)
        hint = load_concept_source_spec(directory / f"{stem}.spec.yaml")
        stages = run_stages(landed, descriptor, hint=hint)
        assert stages.spec is not None, "Roles assembled no spec from the hint"
        client = fake_engine_client()
        # Mocked answers are scripted one per ask; only the agreement tests turn the pair on.
        config = intake_config(max_calls=MAX_CALLS, agreement=False)
        calls = ModelCalls(client, landed.ledger, landed.registry, landed.run_id, config)
        yield EnrichRun(stages, descriptor, calls, client)


def a2_run(root: Path) -> Any:
    return enrich_run(A2 / "codes_a2.xlsx", root, "land_a2", "codes_a2")


def acris_run(root: Path) -> Any:
    return enrich_run(ACRIS, root, "land_acris", "acris")


def domain_blocks() -> list[str]:
    """Every healthcare and financial prompt layer the engine can render, plus its examples."""
    from sema.engine.domain_prompts import (
        build_domain_bias_header,
        build_vocab_family_hints,
        get_semantic_type_inventory,
    )
    from sema.engine.few_shot import format_examples
    from sema.models.domain import DomainContext

    blocks: list[str] = []
    for domain in ("healthcare", "financial"):
        context = DomainContext(declared_domain=domain, domain_source="user")
        blocks += [
            build_domain_bias_header(context),
            build_vocab_family_hints(context),
            get_semantic_type_inventory(context),
        ]
        blocks += [format_examples(domain, stage) for stage in ("B", "C")]
    blocks += [format_examples(None, stage) for stage in ("B", "C")]
    return [block for block in blocks if block]


def domain_types() -> set[str]:
    """Healthcare and financial semantic types that the generic inventory does not hold."""
    from sema.engine.domain_prompts import get_semantic_type_inventory
    from sema.models.domain import DomainContext

    generic = set(get_semantic_type_inventory(None).split(", "))
    found: set[str] = set()
    for domain in ("healthcare", "financial"):
        context = DomainContext(declared_domain=domain, domain_source="user")
        found |= set(get_semantic_type_inventory(context).split(", ")) - generic
    return found


def manifest_property(manifest: dict[str, Any], entity: str, name: str) -> dict[str, Any]:
    (found,) = [e for e in manifest["entities"] if e["qualified_name"] == entity]
    (prop,) = [p for p in found["properties"] if p["name"] == name]
    return dict(prop)

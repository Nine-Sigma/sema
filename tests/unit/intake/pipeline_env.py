"""Every stage run end to end into one ledger and one registry row, with no model.

Used by the A10 end-to-end acceptance (task 13.1) and the held-out generality gate
(task 13.2). Each stage gets its own ``ProfileTools`` under ``metadata_only``, writes its
output and verdict to the run's ledger, and advances the registry row. The dry run runs
only when Roles assembles a spec; otherwise the ledger records why it did not.
"""

from __future__ import annotations

import json
import re
import uuid
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from sema.binding.importer import ConceptImport, import_concept_source
from sema.binding.spec import ConceptSourceSpec
from sema.ingest.duckdb_staging import Staging
from sema.intake.roles.eligibility import assemble_spec, run_roles
from sema.intake.roles.eligibility_spec_utils import RolesResult
from sema.intake.roles.eligibility_utils import RunDescriptor
from sema.intake.land.landing import LandResult
from sema.intake.calls.ledger import Ledger, RunLedger
from sema.intake.land.loaders.tabular import land_tabular
from sema.intake.land.profile import EvidencePolicy, ProfileTools
from sema.intake.calls.registry import Registry, RunRecord
from sema.intake.land.relations import RelationProvider
from sema.intake.report.report import DryRunReport, dry_run
from sema.intake.structure.shapes.base import profile_relations
from sema.intake.structure.structure import StructureResult, run_structure
from sema.intake.verify.base import Ambiguity, Verdict
from sema.intake.verify.dry_run import verify_dry_run
from sema.intake.verify.land import verify_land
from sema.intake.verify.roles import verify_roles
from sema.intake.verify.structure import verify_structure

PROVIDER = "staging"
TIER = "metadata_only"
EXPLORATORY_STAGES = frozenset({"structure", "roles"})
FINAL_STAGE = "dry_run"
REVIEW_STATUS = "waiting_review"
QUOTED_RELATION = re.compile(r'"([^"]+)"\."([^"]+)"')


class AmbiguityList(BaseModel):
    model_config = ConfigDict(frozen=True)

    stage: str
    count: int
    ambiguities: tuple[Ambiguity, ...]


@dataclass
class Landed:
    staging: Staging
    provider: RelationProvider
    land: LandResult
    ledger: RunLedger
    registry: Registry
    run_id: str

    def tools(self, stage: str, structure: StructureResult | None = None) -> ProfileTools:
        policy = EvidencePolicy(tier=TIER)
        return ProfileTools(self.provider, policy, self.ledger, stage=stage, structure=structure)

    def record(self) -> RunRecord:
        (found,) = [r for r in self.registry.list() if r.run_id == self.run_id]
        return found

    def sql_lines(self) -> list[dict[str, Any]]:
        path = self.ledger.directory / "sql.jsonl"
        if not path.is_file():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    def value_bearing(self, stages: frozenset[str]) -> list[dict[str, Any]]:
        return [
            line for line in self.sql_lines() if line["value_bearing"] and line["stage"] in stages
        ]

    def model_calls(self) -> list[Path]:
        calls = self.ledger.directory / "calls"
        return sorted(calls.iterdir()) if calls.is_dir() else []

    def namespace(self) -> set[str]:
        return {f"{r.schema}.{r.name}" for r in self.land.relations}

    def qualified(self, name: str) -> str:
        return f"{self.land.relation(name).schema}.{name}"


@dataclass
class StageRun:
    landed: Landed
    land_verdict: Verdict
    structure: StructureResult
    structure_verdict: Verdict
    roles: RolesResult
    roles_verdict: Verdict
    spec: ConceptSourceSpec | None
    imported: ConceptImport | None = None
    dry_run_verdict: Verdict | None = None
    report: DryRunReport | None = None
    verdicts: dict[str, Verdict] = field(default_factory=dict)


@contextmanager
def landed_run(directory: Path, root: Path, schema: str) -> Iterator[Landed]:
    staging = Staging(str(root / "staging.duckdb"), schemas=())
    try:
        land = land_tabular(directory, staging, schema=schema)
        run_id = str(uuid.uuid4())
        registry = Registry(root / "registry.sqlite")
        registry.register(run_id, source=directory.name, stage="land")
        ledger = Ledger(root / "ledger").run(run_id)
        ledger.artifact("land_result", land)
        provider = RelationProvider(warehouse=None, staging=staging)
        yield Landed(staging, provider, land, ledger, registry, run_id)
    finally:
        staging.close()


def run_stages(
    landed: Landed, descriptor: RunDescriptor, hint: ConceptSourceSpec | None = None
) -> StageRun:
    ledger, registry, run_id = landed.ledger, landed.registry, landed.run_id
    land_verdict = verify_land(landed.land, landed.tools("land"))
    ledger.artifact("land_verdict", land_verdict)
    registry.advance(run_id, "structure", "running")
    structure_tools = landed.tools("structure")
    structure = run_structure(landed.land, structure_tools, provider=PROVIDER, hint=hint)
    structure_verdict = verify_structure(structure, landed.land, structure_tools)
    ledger.artifact("structure", structure)
    ledger.artifact("structure_verdict", structure_verdict)
    registry.advance(run_id, "roles", "running")
    roles_tools = landed.tools("roles", structure)
    profiles = profile_relations(landed.land, roles_tools, PROVIDER)
    roles = run_roles(structure, profiles, roles_tools, hint=hint)
    spec = assemble_spec(descriptor, roles)
    roles_verdict = verify_roles(roles, roles_tools, spec=spec)
    ledger.artifact("roles_verdict", roles_verdict)
    write_ambiguity_lists(ledger, structure.ambiguities, roles.ambiguities)
    run = StageRun(
        landed, land_verdict, structure, structure_verdict, roles, roles_verdict, spec
    )
    if spec is None:
        ledger.event(FINAL_STAGE, "skipped", {"reason": "no spec assembled"})
        return run
    ledger.artifact("spec", spec)
    _dry_run(run, spec)
    registry.advance(run_id, FINAL_STAGE, REVIEW_STATUS)
    return run


def _dry_run(run: StageRun, spec: ConceptSourceSpec) -> None:
    landed = run.landed
    run.imported = import_concept_source(spec, landed.provider, dataset_map={})
    tools = landed.tools(FINAL_STAGE)
    run.dry_run_verdict = verify_dry_run(
        spec, run.imported, structure=run.roles.structure, tools=tools
    )
    landed.ledger.artifact("dry_run_verdict", run.dry_run_verdict)
    run.report = dry_run(spec, landed.provider, landed.staging, landed.ledger)


def write_ambiguity_lists(
    ledger: RunLedger,
    structure: tuple[Ambiguity, ...],
    roles: tuple[Ambiguity, ...],
) -> None:
    for stage, listed in (("structure", structure), ("roles", roles)):
        entry = AmbiguityList(stage=stage, count=len(listed), ambiguities=tuple(listed))
        ledger.artifact(f"{stage}_ambiguities", entry)


def read_ambiguity_list(ledger: RunLedger, stage: str) -> AmbiguityList:
    path = ledger.directory / "artifacts" / f"{stage}_ambiguities.json"
    return AmbiguityList.model_validate_json(path.read_text())


def statement_relations(statement: str) -> set[str]:
    return {f"{schema}.{name}" for schema, name in QUOTED_RELATION.findall(statement)}


def partition_failures(landed: Landed, structure: StructureResult) -> list[str]:
    """Every landed relation exactly once across bound, ignored, and ambiguous."""
    listed = [name for entry in structure.bound for name in entry.binding.relations]
    listed += [name for ignored in structure.ignored for name in ignored.relations]
    listed += [ambiguity.subject for ambiguity in structure.ambiguities]
    counts = Counter(listed)
    repeated = [f"listed twice: {name}" for name, n in counts.items() if n > 1]
    missing = [f"not listed: {name}" for name in sorted(landed.namespace() - set(counts))]
    unknown = [f"not landed: {name}" for name in sorted(set(counts) - landed.namespace())]
    return repeated + missing + unknown


def describe(verdict: Verdict) -> list[tuple[str, str, str | None, dict[str, Any]]]:
    return [(f.check, f.relation, f.column, f.evidence) for f in verdict.failures]

"""The A2 stages run into one ledger directory, for the run-ledger 'Artifacts after a dry run'
scenario: land, structure, roles, their verdicts, and the assembled spec, each written as an
artifact; the dry run itself writes the report."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import yaml

from sema.binding.importer import import_concept_source
from sema.binding.spec import ConceptSourceSpec
from sema.intake.eligibility import assemble_spec, run_roles
from sema.intake.eligibility_utils import RunDescriptor
from sema.intake.landing import LandResult
from sema.intake.ledger import Ledger, RunLedger
from sema.intake.profile import EvidencePolicy, ProfileTools
from sema.intake.relations import RelationProvider
from sema.intake.shapes.base import profile_relations
from sema.intake.structure import StructureResult, run_structure
from sema.intake.verify.base import Verdict
from sema.intake.verify.dry_run import verify_dry_run
from sema.intake.verify.land import verify_land
from sema.intake.verify.roles import verify_roles
from sema.intake.verify.structure import verify_structure
from tests.unit.intake.structure_env import A2_SPEC

A2_RUN = A2_SPEC.parent / "codes_a2.run.yaml"
VERDICTS = ("land_verdict", "structure_verdict", "roles_verdict", "dry_run_verdict")
PROVIDER = "staging"


@dataclass
class Stages:
    provider: RelationProvider
    land: LandResult
    ledger: RunLedger
    structure: StructureResult
    spec: ConceptSourceSpec

    @property
    def artifacts(self) -> Path:
        return self.ledger.directory / "artifacts"

    def tools(self, stage: str) -> ProfileTools:
        policy = EvidencePolicy(tier="metadata_only")
        return ProfileTools(
            self.provider, policy, self.ledger, stage=stage, structure=self.structure
        )

    def verify_dry_run(self) -> None:
        imported = import_concept_source(self.spec, self.provider, dataset_map={})
        tools = self.tools("dry_run")
        verdict = verify_dry_run(self.spec, imported, structure=self.structure, tools=tools)
        self.ledger.artifact("dry_run_verdict", verdict)

    def assert_stage_artifacts_parse(self) -> None:
        land = json.loads((self.artifacts / "land_result.json").read_text())
        assert len(land["relations"]) == len(self.land.relations)
        structure = (self.artifacts / "structure.json").read_text()
        assert StructureResult.model_validate_json(structure).bound
        for name in VERDICTS:
            Verdict.model_validate_json((self.artifacts / f"{name}.json").read_text())
        spec = json.loads((self.artifacts / "spec.json").read_text())
        assert ConceptSourceSpec.model_validate(spec).id == self.spec.id


def a2_stages_into(provider: RelationProvider, land: LandResult, run_dir: Path) -> Stages:
    ledger = Ledger(run_dir.parent).run(run_dir.name)
    ledger.artifact("land_result", land)
    land_tools = ProfileTools(
        provider, EvidencePolicy(tier="metadata_only"), ledger, stage="land"
    )
    ledger.artifact("land_verdict", verify_land(land, land_tools))
    structure_tools = ProfileTools(
        provider, EvidencePolicy(tier="metadata_only"), ledger, stage="structure"
    )
    structure = run_structure(land, structure_tools, provider=PROVIDER, hint=None)
    ledger.artifact("structure", structure)
    ledger.artifact("structure_verdict", verify_structure(structure, land, structure_tools))
    roles_tools = ProfileTools(
        provider, EvidencePolicy(tier="full"), ledger, stage="roles", structure=structure
    )
    roles = run_roles(structure, profile_relations(land, roles_tools, PROVIDER), roles_tools)
    descriptor = RunDescriptor.model_validate(yaml.safe_load(A2_RUN.read_text()))
    spec = assemble_spec(descriptor, roles)
    assert spec is not None, [(a.subject, a.members) for a in roles.ambiguities]
    ledger.artifact("spec", spec)
    ledger.artifact("roles_verdict", verify_roles(roles, roles_tools, spec=spec))
    return Stages(provider, land, ledger, structure, spec)

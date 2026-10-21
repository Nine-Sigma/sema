"""The ingestion graph's node bodies: each recomputes its inputs, runs its stage, and writes
its node event, verdict, and artifacts. The graph module wires them and owns the interrupts.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from sema.binding.importer import import_concept_source, write_import
from sema.binding.spec import load_concept_source_spec
from sema.intake.eligibility import assemble_spec
from sema.intake.enrichment import apply_enrichment
from sema.intake.graph_stages import (
    DECIDE,
    DRY_RUN,
    LAND,
    PROPOSE,
    RESOLVE_ROLES,
    RESOLVE_STRUCTURE,
    ROLES,
    STRUCTURE,
    Current,
    Stages,
)
from sema.intake.graph_utils import (
    APPROVE,
    ENRICH,
    FAIL,
    IMPORT,
    PASS,
    REJECT,
    REVIEW,
    VALUE_ASK,
    AmbiguityList,
    DecisionsArtifact,
    Hooks,
    IngestState,
    approvals_of,
    approved_pairs,
    dry_run_payload,
    inputs_of,
    model_bound,
    proposal_summary,
    reviewed_of,
    unapproved,
    value_payload,
    verdict_update,
)
from sema.intake.metrics import RunMetrics, run_metrics
from sema.intake.eligibility_utils import Embedder
from sema.intake.model_calls import BUDGET_REASON, IntakeClient
from sema.intake.verify.base import Ambiguity, Verdict
from sema.intake.verify.land import verify_land
from sema.intake.verify.structure import verify_structure
from sema.log import logger
from sema.models.config import IntakeModelConfig

NODE_KIND, VERDICT_KIND, RESUME_KIND = "node", "verdict", "resume"
RUNNING, WAITING, APPROVED, REJECTED = "running", "waiting_review", "approved", "rejected"
IMPORT_DIR, MANIFEST_FILE, EVIDENCE_FILE = "import", "manifest.yaml", "authored_evidence.json"
SPEC_ARTIFACT, STRUCTURE_ARTIFACT, METRICS_ARTIFACT = "spec", "structure", "run_metrics"
DECISIONS_ARTIFACT, AMBIGUITY_SUFFIX = "decisions", "_ambiguities"

Update = dict[str, Any]


class Session:
    """One process's view of one run: the stages, built once from the run's state."""

    def __init__(
        self,
        root: Path,
        config: IntakeModelConfig,
        client: IntakeClient | None,
        hooks: Hooks,
        embedder: Embedder | None = None,
    ) -> None:
        self.root, self.config, self.client, self.hooks = root, config, client, hooks
        self.embedder = embedder
        self.answering = False
        self._stages: Stages | None = None

    def take_answer(self) -> bool:
        """True once after a reviewer's answer resumes Review, so it is not recomputed."""
        answering, self.answering = self.answering, False
        return answering

    def stages(self, state: IngestState) -> Stages:
        if self._stages is None:
            self._stages = Stages(
                inputs_of(state), self.root, self.config, self.client, self.embedder
            )
        return self._stages

    def close(self) -> None:
        if self._stages is not None:
            self._stages.close()


class Nodes:
    def __init__(self, session: Session) -> None:
        self.session = session

    def land(self, state: IngestState) -> Update:
        stages, update = self._enter(state, LAND)
        verdict = verify_land(stages.land(), stages.tools(LAND))
        return update | self._verdict(state, LAND, LAND, verdict)

    def structure(self, state: IngestState) -> Update:
        stages, update = self._enter(state, STRUCTURE)
        _write(stages, update, STRUCTURE_ARTIFACT, stages.structure())
        return update

    def resolve_structure(self, state: IngestState) -> Update:
        stages, update = self._enter(state, RESOLVE_STRUCTURE)
        resolved = stages.resolved()
        _write(stages, update, STRUCTURE_ARTIFACT, resolved)
        verdict = verify_structure(resolved, stages.land(), stages.tools(RESOLVE_STRUCTURE))
        return update | self._verdict(state, RESOLVE_STRUCTURE, STRUCTURE, verdict)

    def propose_layouts(self, state: IngestState) -> Update:
        stages, update = self._enter(state, PROPOSE)
        approvals = approvals_of(state)
        proposal = stages.proposals(approvals)
        _write(stages, update, STRUCTURE_ARTIFACT, proposal.structure)
        if unapproved(proposal.value_requests, approvals):
            return update | {"asking": True, "value_asks": state.get("value_asks", 0) + 1}
        verdict = verify_structure(proposal.structure, stages.land(), stages.tools(PROPOSE))
        return update | {"asking": False} | self._verdict(state, PROPOSE, STRUCTURE, verdict)

    def roles(self, state: IngestState) -> Update:
        stages, update = self._enter(state, ROLES)
        stages.roles(approvals_of(state))
        return update

    def resolve_roles(self, state: IngestState) -> Update:
        stages, update = self._enter(state, RESOLVE_ROLES)
        approvals = approvals_of(state)
        roles = stages.resolved_roles(approvals)
        spec = assemble_spec(stages.descriptor(), roles)
        if spec is not None:
            _write(stages, update, SPEC_ARTIFACT, spec)
        tools = stages.tools(RESOLVE_ROLES, stages.proposals(approvals).structure)
        verdict = self.session.hooks.verify_roles(roles, tools, spec=spec)
        update["has_spec"] = spec is not None
        return update | self._verdict(state, RESOLVE_ROLES, ROLES, verdict)

    def enrich(self, state: IngestState) -> Update:
        stages, update = self._enter(state, ENRICH)
        current = stages.current(reviewed_of(state))
        if self._enrichment(stages, current, ENRICH) is not None:
            update["artifacts"].append(ENRICH)
        return update

    def dry_run(self, state: IngestState) -> Update:
        stages, update = self._enter(state, DRY_RUN)
        ran = stages.dry_run(reviewed_of(state))
        if ran is None:
            return update | {"has_spec": False, "verdict": PASS}
        return update | self._verdict(state, DRY_RUN, DRY_RUN, ran[0])

    def value_ask(self, state: IngestState) -> dict[str, Any]:
        """The payload the value interrupt presents: every column not yet approved."""
        stages, _ = self._enter(state, VALUE_ASK)
        approvals = approvals_of(state)
        requests = unapproved(stages.proposals(approvals).value_requests, approvals)
        stages.registry.advance(state["run_id"], VALUE_ASK, WAITING)
        return value_payload(requests)

    def after_values(self, state: IngestState, payload: dict[str, Any]) -> Update:
        approvals = [*state.get("approvals", []), *approved_pairs(payload)]
        return {"approvals": approvals, "asking": False, "reached": [VALUE_ASK]}

    def review(self, state: IngestState) -> dict[str, Any]:
        """Writes what Review presents and returns its payload: open structure entries, then
        settled ones with their source, then role entries."""
        stages, update = self._enter(state, REVIEW)
        current = stages.current(reviewed_of(state))
        listed = self._ambiguity_lists(stages, current, update)
        metrics = self._metrics(stages, state, current, update)
        ran = stages.dry_run(reviewed_of(state)) if DRY_RUN in state.get("reached", []) else None
        stages.registry.advance(state["run_id"], REVIEW, WAITING, reason=state.get("reason"))
        spec = current.spec.model_dump(mode="json", by_alias=True) if current.spec else None
        written = sorted({*state.get("artifacts", []), *update["artifacts"]})
        return {
            "ambiguities": [entry.model_dump(mode="json") for entry in listed],
            "proposals": proposal_summary(current.proposal),
            "spec": spec,
            "dry_run": dry_run_payload(ran),
            "run_metrics": metrics.model_dump(mode="json"),
            "ledger": {"directory": str(stages.ledger.directory), "artifacts": written},
        }

    def after_review(self, state: IngestState, answer: dict[str, Any]) -> Update:
        stages = self.session.stages(state)
        action, approver = answer["action"], answer.get("approver")
        update: Update = {
            "action": action, "approver": approver, "reason": None, "stopped": False,
        }
        update["artifacts"] = []
        if action == REJECT:
            stages.registry.advance(state["run_id"], REVIEW, REJECTED)
            stages.registry.set_approver(state["run_id"], approver)
        if action in (APPROVE, REJECT):
            return update
        if answer.get("spec"):
            _write(stages, update, SPEC_ARTIFACT, load_concept_source_spec(Path(answer["spec"])))
            return update | {"spec_path": answer["spec"]}
        decisions = [*state.get("decisions", []), *answer["decisions"]]
        return update | {"decisions": decisions, "spec_path": None}

    def decide(self, state: IngestState) -> Update:
        stages, update = self._enter(state, DECIDE)
        reviewed = reviewed_of(state)
        if reviewed.decisions:
            written = DecisionsArtifact(decisions=reviewed.decisions)
            _write(stages, update, DECISIONS_ARTIFACT, written)
        current = stages.current(reviewed)
        _write(stages, update, STRUCTURE_ARTIFACT, current.structure)
        if current.spec is not None and reviewed.spec_path is None:
            _write(stages, update, SPEC_ARTIFACT, current.spec)
        self._ambiguity_lists(stages, current, update)
        self._metrics(stages, state, current, update)
        update |= self._decided_verdicts(state, stages, current)
        return update | {"has_spec": current.spec is not None}

    def import_(self, state: IngestState) -> Update:
        stages, update = self._enter(state, IMPORT)
        current = stages.current(reviewed_of(state))
        if current.spec is None:
            raise ValueError(f"run {state['run_id']} has no spec to import")
        imported = import_concept_source(current.spec, stages.provider, dataset_map={})
        if state.get("enrich_on"):
            found = self._enrichment(stages, current, IMPORT)
            manifest = apply_enrichment(imported.manifest, found)
            imported = dataclasses.replace(imported, manifest=manifest)
        directory = stages.ledger.directory / IMPORT_DIR
        directory.mkdir(parents=True, exist_ok=True)
        write_import(imported, directory / MANIFEST_FILE, directory / EVIDENCE_FILE)
        stages.registry.advance(state["run_id"], IMPORT, APPROVED)
        stages.registry.set_approver(state["run_id"], state.get("approver"))
        logger.info(f"Run {state['run_id']} imported to {directory}")
        return update

    def presented(self, state: IngestState) -> list[Ambiguity]:
        """The ambiguity list Review presented, recomputed to validate decisions against."""
        current = self.session.stages(state).current(reviewed_of(state))
        roles = current.roles.ambiguities if current.roles is not None else ()
        return [*current.structure.ambiguities, *roles]

    def has_spec(self, state: IngestState) -> bool:
        return self.session.stages(state).current(reviewed_of(state)).spec is not None

    def budgeted(self, state: IngestState, update: Update) -> Update:
        """A call the budget refused in this process stops the run at Review."""
        if not self.session.stages(state).calls.refused:
            return update
        return update | {"stopped": True, "reason": BUDGET_REASON}

    def resumed(self, state: IngestState, node: str) -> None:
        self.session.stages(state).ledger.event(node, RESUME_KIND, {})

    def _enrichment(self, stages: Stages, current: Current, stage: str) -> Any:
        if current.spec is None or current.roles is None:
            return None
        tools = stages.tools(stage, current.structure)
        return self.session.hooks.enrich(
            current.spec, current.roles, tools, stages.calls, stages.ledger, enabled=True
        )

    def _decided_verdicts(self, state: IngestState, stages: Stages, current: Current) -> Update:
        """Every decided binding is still verified; any failure returns the run to Review."""
        tools = stages.tools(DECIDE, current.structure)
        found = [(STRUCTURE, verify_structure(current.structure, current.land, tools))]
        if current.roles is not None:
            verdict = self.session.hooks.verify_roles(current.roles, tools, spec=current.spec)
            found.append((ROLES, verdict))
        update: Update = {}
        for stage, verdict in found:
            update |= self._verdict({**state, **update}, f"{DECIDE}.{stage}", stage, verdict)
        passed = all(verdict.status == PASS for _, verdict in found)
        return update | {"verdict": PASS if passed else FAIL}

    def _enter(self, state: IngestState, node: str) -> tuple[Stages, Update]:
        stages = self.session.stages(state)
        stages.ledger.event(node, NODE_KIND, {})
        stages.registry.advance(state["run_id"], node, RUNNING)
        return stages, {"stage": node, "reached": [node], "artifacts": []}

    def _verdict(self, state: Any, key: str, stage: str, verdict: Verdict) -> Update:
        status, attempt, update = verdict_update(state, key, verdict)
        failures = [failure.model_dump(mode="json") for failure in verdict.failures]
        payload = {"status": status, "attempt": attempt, "failures": failures}
        self.session.stages(state).ledger.event(stage, VERDICT_KIND, payload)
        return update

    def _ambiguity_lists(
        self, stages: Stages, current: Current, update: Update
    ) -> list[Ambiguity]:
        roles = current.roles.ambiguities if current.roles is not None else ()
        for stage, listed in ((STRUCTURE, current.structure.ambiguities), (ROLES, roles)):
            entry = AmbiguityList(stage=stage, count=len(listed), ambiguities=tuple(listed))
            _write(stages, update, stage + AMBIGUITY_SUFFIX, entry)
        structure = current.structure
        return [*structure.ambiguities, *structure.settled, *roles]

    def _metrics(
        self, stages: Stages, state: IngestState, current: Current, update: Update
    ) -> RunMetrics:
        proposal = current.proposal
        metrics = run_metrics(
            current.land, current.structure, current.roles,
            stages.registry.get(state["run_id"]),
            proposals_made=len(proposal.attempts) if proposal else 0,
            proposals_bound=len(model_bound(proposal)) if proposal else 0,
            value_asks=state.get("value_asks", 0),
        )
        _write(stages, update, METRICS_ARTIFACT, metrics)
        return metrics


def _write(stages: Stages, update: Update, name: str, obj: BaseModel) -> None:
    stages.ledger.artifact(name, obj)
    update["artifacts"].append(name)

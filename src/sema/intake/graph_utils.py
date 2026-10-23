"""Ingestion graph state, routing, and Review payload pieces; none imports the graph runtime.

The state holds scalars and short lists of names only: the run inputs, the node the run
is at, consecutive failures per verdict, the reviewer's decisions and value approvals,
and the nodes entered and artifacts written. Routing reads the state, never the ledger.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, TypedDict, cast, get_args

from pydantic import BaseModel, ConfigDict, SerializeAsAny

from sema.intake.decisions import Decision
from sema.intake.enrichment import Enrichment
from sema.intake.graph_stages import (
    DECIDE,
    DRY_RUN,
    LAND,
    PROPOSE,
    RESOLVE_ROLES,
    RESOLVE_STRUCTURE,
    ROLES,
    STRUCTURE,
    Approvals,
    Reviewed,
    RunInputs,
)
from sema.intake.loaders.base import relation_name
from sema.intake.profile import EvidenceTier
from sema.intake.profile_utils import NeedsValues
from sema.intake.proposal import ProposalResult
from sema.intake.report import DryRunReport
from sema.intake.structure import BindingSource
from sema.intake.verify.base import Ambiguity, Verdict

MAX_ATTEMPTS = 3
PASS, FAIL, ESCALATE = "pass", "fail", "escalate"
REVIEW, VALUE_ASK, IMPORT = "review", "value_ask", "import"
ENRICH = "enrich"
END_ROUTE = "end"
APPROVE, REJECT = "approve", "reject"
SCHEMA_PREFIX = "land_"
MODEL_BINDING: BindingSource = "model"


def merged(left: list[str], right: list[str]) -> list[str]:
    return list(dict.fromkeys([*left, *right]))


class IngestState(TypedDict, total=False):
    run_id: str
    source_path: str | None
    catalog: str | None
    descriptor: str
    hint: str | None
    schema: str
    tier: str
    enrich_on: bool
    stage: str
    attempts: dict[str, int]
    verdict: str
    reason: str | None
    stopped: bool
    has_spec: bool
    asking: str | None
    value_asks: int
    approvals: list[list[str]]
    decisions: list[dict[str, str]]
    spec_path: str | None
    action: str | None
    approver: str | None
    reached: Annotated[list[str], merged]
    artifacts: Annotated[list[str], merged]


@dataclass(frozen=True)
class Hooks:
    """Stage functions the graph module names, so they can be replaced where it names them."""

    verify_roles: Callable[..., Verdict]
    enrich: Callable[..., Enrichment | None]


class AmbiguityList(BaseModel):
    model_config = ConfigDict(frozen=True)

    stage: str
    count: int
    # A cause entry's ``cause`` key reaches the ledger, where the reader prints it.
    ambiguities: tuple[SerializeAsAny[Ambiguity], ...]


class DecisionsArtifact(BaseModel):
    model_config = ConfigDict(frozen=True)

    decisions: tuple[Decision, ...]


def initial_state(
    run_id: str,
    source: Path | None,
    descriptor: Path,
    *,
    catalog: str | None,
    schema: str | None,
    tier: EvidenceTier,
    hint: Path | None,
    enrich: bool,
) -> IngestState:
    """A path run, or a warehouse run naming ``catalog`` and ``schema``."""
    return IngestState(
        run_id=run_id,
        source_path=str(source.resolve()) if source is not None else None,
        catalog=catalog,
        descriptor=str(descriptor.resolve()),
        hint=str(hint.resolve()) if hint is not None else None,
        schema=_landing_schema(source, catalog, schema),
        tier=tier,
        enrich_on=enrich,
        stage=LAND,
        attempts={},
        verdict=PASS,
        reason=None,
        stopped=False,
        has_spec=False,
        asking=None,
        value_asks=0,
        approvals=[],
        decisions=[],
        spec_path=None,
        action=None,
        approver=None,
        reached=[],
        artifacts=[],
    )


def schema_for(source: Path) -> str:
    return SCHEMA_PREFIX + relation_name(source.stem).lower()


def source_label(source: Path | None, catalog: str | None, schema: str | None) -> str:
    """The registry's source: the path's name, or ``<catalog>.<schema>``."""
    return source.name if source is not None else f"{catalog}.{schema}"


def _landing_schema(source: Path | None, catalog: str | None, schema: str | None) -> str:
    if (source is None) == (catalog is None):
        raise ValueError("a run lands either a source path or a warehouse catalog and schema")
    if source is not None:
        return schema or schema_for(source)
    if not schema:
        raise ValueError(f"a warehouse run on catalog {catalog} needs its schema")
    return schema


def inputs_of(state: IngestState) -> RunInputs:
    hint, source = state.get("hint"), state.get("source_path")
    return RunInputs(
        run_id=state["run_id"],
        source=Path(source) if source else None,
        catalog=state.get("catalog"),
        descriptor=Path(state["descriptor"]),
        schema=state["schema"],
        tier=_tier(state["tier"]),
        hint=Path(hint) if hint else None,
        enrich=bool(state.get("enrich_on")),
    )


def _tier(value: str) -> EvidenceTier:
    tiers = get_args(EvidenceTier)
    if value not in tiers:
        raise ValueError(f"unknown evidence tier {value!r}; expected one of {', '.join(tiers)}")
    return cast(EvidenceTier, value)


def reviewed_of(state: IngestState) -> Reviewed:
    spec_path = state.get("spec_path")
    return Reviewed(
        approvals=approvals_of(state),
        decisions=tuple(Decision.model_validate(row) for row in state.get("decisions", [])),
        spec_path=Path(spec_path) if spec_path else None,
        reached=frozenset(state.get("reached", [])),
    )


def approvals_of(state: IngestState) -> Approvals:
    return tuple((relation, column) for relation, column in state.get("approvals", []))


def unapproved(requests: Sequence[NeedsValues], approvals: Approvals) -> list[NeedsValues]:
    allowed = set(approvals)
    return [
        request for request in requests
        if any((request.relation, column) not in allowed for column in request.columns)
    ]


def value_payload(requests: Sequence[NeedsValues]) -> dict[str, Any]:
    return {"requests": [{"relation": r.relation, "columns": list(r.columns)} for r in requests]}


def approved_pairs(payload: dict[str, Any]) -> list[list[str]]:
    return [
        [request["relation"], column]
        for request in payload["requests"]
        for column in request["columns"]
    ]


def verdict_update(
    state: IngestState, key: str, verdict: Verdict
) -> tuple[str, int, dict[str, Any]]:
    """The routed status, the attempt number, and the state update for one verdict.

    Attempts count consecutive failures, so a node that passes and later re-runs starts again.
    """
    attempts = dict(state.get("attempts", {}))
    attempt = attempts.get(key, 0) + 1
    status = str(verdict.status)
    if status != PASS and attempt >= MAX_ATTEMPTS:
        status = ESCALATE
    attempts[key] = 0 if status == PASS else attempt
    update: dict[str, Any] = {"attempts": attempts, "verdict": status}
    if status == ESCALATE:
        update["reason"] = ESCALATE
    return status, attempt, update


def after_verdict(state: IngestState, node: str, passed: str) -> str:
    """A run the budget stopped goes to Review whatever its verdict."""
    if state.get("stopped"):
        return REVIEW
    status = state.get("verdict", PASS)
    if status == PASS:
        return passed
    return REVIEW if status == ESCALATE else node


def route_land(state: IngestState) -> str:
    return after_verdict(state, LAND, STRUCTURE)


def route_resolve_structure(state: IngestState) -> str:
    return after_verdict(state, RESOLVE_STRUCTURE, PROPOSE)


def _asked(state: IngestState, node: str) -> bool:
    return state.get("asking") == node and not state.get("stopped")


def route_propose(state: IngestState) -> str:
    return VALUE_ASK if _asked(state, PROPOSE) else after_verdict(state, PROPOSE, ROLES)


def route_roles(state: IngestState) -> str:
    return VALUE_ASK if _asked(state, ROLES) else RESOLVE_ROLES


def route_value_ask(state: IngestState) -> str:
    """An approval re-enters the node that asked."""
    return state.get("asking") or PROPOSE


def route_resolve_roles(state: IngestState) -> str:
    if not state.get("has_spec"):
        return after_verdict(state, RESOLVE_ROLES, REVIEW)
    following = ENRICH if state.get("enrich_on") else DRY_RUN
    return after_verdict(state, RESOLVE_ROLES, following)


def route_dry_run(state: IngestState) -> str:
    return after_verdict(state, DRY_RUN, REVIEW)


def route_review(state: IngestState) -> str:
    action = state.get("action")
    if action == APPROVE:
        return IMPORT
    return END_ROUTE if action == REJECT else DECIDE


def route_decide(state: IngestState) -> str:
    """A settled spec always dry-runs before Review, budget stop or not."""
    passed = state.get("verdict") == PASS and state.get("has_spec")
    return DRY_RUN if passed else REVIEW


def proposal_summary(proposal: ProposalResult | None) -> list[dict[str, Any]]:
    if proposal is None:
        return []
    bound = model_bound(proposal)
    return [
        {
            "relation": relation,
            "attempts": len(attempts),
            "bound": relation in bound,
            "escalated": relation in proposal.escalated,
            "findings": [f.rationale for f in proposal.findings if f.relation == relation],
        }
        for relation, attempts in proposal.attempts.items()
    ]


def model_bound(proposal: ProposalResult) -> set[str]:
    bound = {
        name
        for entry in proposal.structure.bound
        if entry.source == MODEL_BINDING
        for name in entry.binding.relations
    }
    return {relation for relation in proposal.attempts if relation in bound}


def dry_run_payload(ran: tuple[Verdict, DryRunReport] | None) -> dict[str, Any] | None:
    if ran is None:
        return None
    verdict, report = ran
    return {"verdict": verdict.model_dump(mode="json"), "report": report.model_dump(mode="json")}

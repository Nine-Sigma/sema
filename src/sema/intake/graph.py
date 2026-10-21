"""Run an ingestion as a resumable LangGraph graph (design Decision 5).

Nodes run Land, Structure, its resolver, layout proposals, Roles, its resolver, Enrich when
enabled, and the dry run, with each verifier after its node; a failing verdict re-enters
the node at most twice, then Review is reached with reason ``escalate``; a call the budget
refuses sends the run to Review with reason ``budget_exceeded``. Review and value asks are
interrupts. The SQLite checkpointer only tells which node to re-enter: every node
recomputes its inputs from Land, with model calls replayed from the run ledger.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Hashable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, interrupt

from sema.binding.spec import load_concept_source_spec
from sema.intake.decisions import load_decisions, validate_decisions
from sema.intake.eligibility_utils import Embedder
from sema.intake.enrichment import enrich
from sema.intake.graph_nodes import Nodes, Session
from sema.intake.graph_stages import (
    DECIDE,
    DRY_RUN,
    LAND,
    PROPOSE,
    REGISTRY_FILE,
    RESOLVE_ROLES,
    RESOLVE_STRUCTURE,
    ROLES,
    STRUCTURE,
)
from sema.intake.graph_utils import (
    APPROVE,
    END_ROUTE,
    ENRICH,
    IMPORT,
    REJECT,
    REVIEW,
    VALUE_ASK,
    Hooks,
    IngestState,
    initial_state,
    route_decide,
    route_dry_run,
    route_land,
    route_propose,
    route_resolve_roles,
    route_resolve_structure,
    route_review,
)
from sema.intake.model_calls import IntakeClient
from sema.intake.profile import EvidenceTier
from sema.intake.registry import Registry
from sema.intake.tracing import Tracer, run_tracing
from sema.intake.verify.roles import verify_roles
from sema.models.config import IntakeModelConfig, TracingConfig

__all__ = ["RunOutcome", "approve", "approve_values", "discover", "reject", "resume"]

CHECKPOINT_FILE = "checkpoints.sqlite"
RECURSION_LIMIT = 100
ROUTES = (
    LAND, STRUCTURE, RESOLVE_STRUCTURE, PROPOSE, VALUE_ASK, ROLES, RESOLVE_ROLES, ENRICH,
    DRY_RUN, REVIEW, DECIDE, IMPORT,
)

Graph = CompiledStateGraph[IngestState, None, IngestState, IngestState]
NodeBody = Callable[[IngestState], dict[str, Any]]
INTERRUPTS = frozenset({REVIEW, VALUE_ASK})


@dataclass(frozen=True)
class RunOutcome:
    """``node`` is the interrupt the run stopped at, or None once the run ended."""

    run_id: str
    node: str | None
    payload: dict[str, Any] | None


@dataclass(frozen=True)
class _Opened:
    graph: Graph
    config: RunnableConfig
    nodes: Nodes
    run_id: str

    def state(self) -> IngestState:
        return cast(IngestState, self.graph.get_state(self.config).values)

    def invoke(self, command: IngestState | Command[Any] | None) -> RunOutcome:
        self.graph.invoke(command, self.config, durability="sync")
        return self.outcome()

    def answer(self, answer: dict[str, Any]) -> RunOutcome:
        """Resume Review with the reviewer's answer; Review is not recomputed to take it."""
        self.nodes.session.answering = True
        return self.invoke(Command(resume=answer))

    def outcome(self) -> RunOutcome:
        for task in self.graph.get_state(self.config).tasks:
            if task.interrupts:
                return RunOutcome(self.run_id, task.name, task.interrupts[0].value)
        return RunOutcome(self.run_id, None, None)

    def waiting(self, node: str) -> IngestState:
        """The run's state, once it is waiting at ``node``; anything else is refused."""
        stopped = self.outcome().node
        if stopped != node:
            row = Registry(self.nodes.session.root / REGISTRY_FILE).get(self.run_id)
            raise ValueError(
                f"run {self.run_id} is not waiting at {node}: it is {row.status} at {row.stage}"
            )
        return self.state()


def discover(
    source: Path,
    *,
    root: Path,
    descriptor: Path,
    config: IntakeModelConfig,
    client: IntakeClient | None = None,
    hint: Path | None = None,
    enrich: bool = False,
    tier: EvidenceTier = "metadata_only",
    run_id: str | None = None,
    schema: str | None = None,
    tracing: TracingConfig | None = None,
    embedder: Embedder | None = None,
) -> RunOutcome:
    """Start a run from a source and run it to its first interrupt.

    ``embedder`` ranks role columns; pass the same one to every later call on the run.
    """
    run_id = run_id or str(uuid.uuid4())
    state = initial_state(
        run_id, source, descriptor, schema=schema, tier=tier, hint=hint, enrich=enrich
    )
    with _opened(run_id, "discover", root, config, client, tracing, embedder) as run:
        Registry(root / REGISTRY_FILE).register(run_id, source=source.name, stage=LAND)
        return run.invoke(state)


def approve(
    run_id: str,
    *,
    root: Path,
    config: IntakeModelConfig,
    approver: str,
    client: IntakeClient | None = None,
    spec: Path | None = None,
    decisions: Path | None = None,
    tracing: TracingConfig | None = None,
    embedder: Embedder | None = None,
) -> RunOutcome:
    """Import, re-enter with an edited spec, or settle open entries with a decisions file."""
    with _opened(run_id, "approve", root, config, client, tracing, embedder) as run:
        state = run.waiting(REVIEW)
        answer: dict[str, Any] = {"action": APPROVE, "approver": approver}
        if decisions is not None:
            checked = validate_decisions(load_decisions(decisions), run.nodes.presented(state))
            rows = [decision.model_dump(mode="json") for decision in checked]
            answer |= {"action": DECIDE, "decisions": rows}
        elif spec is not None:
            load_concept_source_spec(spec)
            answer |= {"action": DECIDE, "spec": str(spec.resolve())}
        elif not run.nodes.has_spec(state):
            raise ValueError(f"run {run_id} has no spec to import; settle its open entries")
        return run.answer(answer)


def reject(
    run_id: str,
    *,
    root: Path,
    config: IntakeModelConfig,
    approver: str,
    client: IntakeClient | None = None,
    tracing: TracingConfig | None = None,
    embedder: Embedder | None = None,
) -> RunOutcome:
    with _opened(run_id, "reject", root, config, client, tracing, embedder) as run:
        run.waiting(REVIEW)
        return run.answer({"action": REJECT, "approver": approver})


def approve_values(
    run_id: str,
    *,
    root: Path,
    config: IntakeModelConfig,
    client: IntakeClient | None = None,
    tracing: TracingConfig | None = None,
    embedder: Embedder | None = None,
) -> RunOutcome:
    """Approve every column the value interrupt requested; the requesting node re-runs."""
    with _opened(run_id, "approve_values", root, config, client, tracing, embedder) as run:
        run.waiting(VALUE_ASK)
        return run.invoke(Command(resume={"action": APPROVE}))


def resume(
    run_id: str,
    *,
    root: Path,
    config: IntakeModelConfig,
    client: IntakeClient | None = None,
    tracing: TracingConfig | None = None,
    embedder: Embedder | None = None,
) -> RunOutcome:
    """Continue a run after a process exit at the node it stopped in."""
    with _opened(run_id, "resume", root, config, client, tracing, embedder) as run:
        pending = run.graph.get_state(run.config).next
        if not pending:
            return run.outcome()
        run.nodes.resumed(run.state(), pending[0])
        return run.invoke(None)


@contextmanager
def _opened(
    run_id: str,
    entry: str,
    root: Path,
    config: IntakeModelConfig,
    client: IntakeClient | None,
    tracing: TracingConfig | None,
    embedder: Embedder | None,
) -> Iterator[_Opened]:
    """Tracing opens first, so a missing backend package fails the run before Land."""
    with run_tracing(tracing or TracingConfig(), run_id, entry=entry) as tracer:
        root.mkdir(parents=True, exist_ok=True)
        hooks = Hooks(verify_roles=verify_roles, enrich=enrich)
        session = Session(root, config, client, hooks, embedder)
        nodes = Nodes(session)
        runnable: RunnableConfig = {
            "configurable": {"thread_id": run_id}, "recursion_limit": RECURSION_LIMIT,
        }
        try:
            with SqliteSaver.from_conn_string(str(root / CHECKPOINT_FILE)) as saver:
                graph = _graph(nodes, tracer).compile(checkpointer=saver)
                yield _Opened(graph, runnable, nodes, run_id)
        finally:
            session.close()


def _graph(
    nodes: Nodes, tracer: Tracer
) -> StateGraph[IngestState, None, IngestState, IngestState]:
    graph: StateGraph[IngestState, None, IngestState, IngestState] = StateGraph(IngestState)
    bodies: dict[str, NodeBody] = {
        LAND: nodes.land,
        STRUCTURE: nodes.structure,
        RESOLVE_STRUCTURE: nodes.resolve_structure,
        PROPOSE: nodes.propose_layouts,
        VALUE_ASK: _value_ask(nodes),
        ROLES: nodes.roles,
        RESOLVE_ROLES: nodes.resolve_roles,
        ENRICH: nodes.enrich,
        DRY_RUN: nodes.dry_run,
        REVIEW: _review(nodes),
        DECIDE: nodes.decide,
        IMPORT: nodes.import_,
    }
    for name, body in bodies.items():
        graph.add_node(name, _traced(tracer, name, body, None if name in INTERRUPTS else nodes))
    graph.add_edge(START, LAND)
    graph.add_edge(STRUCTURE, RESOLVE_STRUCTURE)
    graph.add_edge(VALUE_ASK, PROPOSE)
    graph.add_edge(ROLES, RESOLVE_ROLES)
    graph.add_edge(ENRICH, DRY_RUN)
    graph.add_edge(IMPORT, END)
    routes: dict[Hashable, str] = {name: name for name in (*ROUTES, END_ROUTE)}
    routes[END_ROUTE] = END
    graph.add_conditional_edges(LAND, route_land, routes)
    graph.add_conditional_edges(RESOLVE_STRUCTURE, route_resolve_structure, routes)
    graph.add_conditional_edges(PROPOSE, route_propose, routes)
    graph.add_conditional_edges(RESOLVE_ROLES, route_resolve_roles, routes)
    graph.add_conditional_edges(DRY_RUN, route_dry_run, routes)
    graph.add_conditional_edges(REVIEW, route_review, routes)
    graph.add_conditional_edges(DECIDE, route_decide, routes)
    return graph


def _traced(tracer: Tracer, stage: str, body: NodeBody, budgeted: Nodes | None) -> Any:
    """``budgeted`` stops the run at Review after a node in which the budget refused a call."""

    def node(state: IngestState) -> dict[str, Any]:
        with tracer.node(stage):
            update = body(state)
        return budgeted.budgeted(state, update) if budgeted is not None else update

    return node


def _review(nodes: Nodes) -> NodeBody:
    def review(state: IngestState) -> dict[str, Any]:
        payload = None if nodes.session.take_answer() else nodes.review(state)
        answer: dict[str, Any] = interrupt(payload)
        return nodes.after_review(state, answer)

    return review


def _value_ask(nodes: Nodes) -> NodeBody:
    def value_ask(state: IngestState) -> dict[str, Any]:
        payload = nodes.value_ask(state)
        interrupt(payload)
        return nodes.after_values(state, payload)

    return value_ask

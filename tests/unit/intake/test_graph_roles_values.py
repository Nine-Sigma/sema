"""Roles value requests reach the value interrupt (ingestion-graph: interrupts are nodes).

When Roles needs values a ``metadata_only`` policy withholds (here: a settled status
column, whose active and retired values only a read can classify), the run stops at the
value interrupt naming the relation and columns, never silently assembling without them.
Approving re-enters ``roles`` with the approval in its profiling policy, so the status role
assembles; the ask is counted in ``run_metrics`` and is not a Review. Every model call the
re-entry repeats is a ledger replay. Without such a need Roles never asks.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.unit.intake.catalog_env import (
    ACTIVE,
    RETIRED,
    STATUS,
    status_classes,
    status_role,
    values_relation,
    write_catalog,
)
from tests.unit.intake.graph_env import (
    LAYOUT_PROPOSAL,
    GraphRun,
    discover,
    graph,
    model_names,
    routed_client,
    settings,
)

pytestmark = pytest.mark.unit

VALUE_ASK, ROLES, REVIEW = "value_ask", "roles", "review"


def _asked(root: Path) -> GraphRun:
    source, descriptor = write_catalog(root / "inputs")
    return discover(root, source, descriptor, routed_client())


def test_roles_status_values_stop_the_run_at_the_value_interrupt(tmp_path: Path) -> None:
    run = _asked(tmp_path)

    assert run.outcome.node == VALUE_ASK
    (request,) = run.outcome.payload["requests"]
    assert values_relation(request["relation"])
    assert request["columns"] == [STATUS]
    row = run.row()
    assert (row.stage, row.status, row.reason) == (VALUE_ASK, "waiting_review", None)
    assert run.nodes()[-2:] == [ROLES, VALUE_ASK]
    assert not {"resolve_roles", "dry_run", REVIEW} & set(run.nodes())
    assert model_names(run.client) == [LAYOUT_PROPOSAL]


def test_value_approval_reenters_roles_and_the_status_role_assembles(tmp_path: Path) -> None:
    run = _asked(tmp_path)
    mark, fresh = run.mark(), routed_client()

    outcome = graph().approve_values(run.run_id, client=fresh, **settings(tmp_path))

    assert outcome.node == REVIEW
    later = [node for node in run.nodes(mark) if node != VALUE_ASK]
    assert later[0] == ROLES
    assert "propose_layouts" not in later
    role = status_role(outcome.payload["spec"])
    assert role is not None and role["column"] == STATUS
    assert status_classes(role) == ({*ACTIVE}, {*RETIRED})
    assert status_role(run.artifact("spec")) == role
    assert run.artifact("run_metrics")["value_asks"] == 1
    assert run.nodes(mark).count(REVIEW) == 1


def test_value_approval_replays_every_call_the_reentry_repeats(tmp_path: Path) -> None:
    run = _asked(tmp_path)
    mark, before, fresh = run.mark(), run.counters(), routed_client()

    graph().approve_values(run.run_id, client=fresh, **settings(tmp_path))

    assert LAYOUT_PROPOSAL not in model_names(fresh)
    assert LAYOUT_PROPOSAL in run.replayed(mark)
    assert run.row().model_calls == before[0] + fresh.invoke.call_count


def test_resume_at_the_roles_value_ask_asks_again_without_a_call(tmp_path: Path) -> None:
    run = _asked(tmp_path)
    asked, before, fresh = run.outcome.payload, run.counters(), routed_client()

    outcome = graph().resume(run.run_id, client=fresh, **settings(tmp_path))

    assert outcome.node == VALUE_ASK
    assert outcome.payload == asked
    assert fresh.invoke.call_count == 0
    assert run.counters() == before


@pytest.mark.parametrize(
    ("tier", "status"),
    [("full", True), ("metadata_only", False)],
    ids=["values-allowed", "no-status-column"],
)
def test_roles_without_a_withheld_value_need_never_ask(
    tmp_path: Path, tier: str, status: bool
) -> None:
    source, descriptor = write_catalog(tmp_path / "inputs", status=status)

    run = discover(tmp_path, source, descriptor, routed_client(), tier=tier)

    assert run.outcome.node == REVIEW
    assert VALUE_ASK not in run.nodes()
    assert run.artifact("run_metrics")["value_asks"] == 0
    spec = run.outcome.payload["spec"]
    if status:
        role = status_role(spec)
        assert role is not None and status_classes(role) == ({*ACTIVE}, {*RETIRED})
    else:
        assert status_role(spec) is None

"""The ingestion graph's re-entries: the value interrupt, decisions, and resume after exit.

The pinned ``sema.intake.graph`` API is in ``test_graph.py``. Every re-entry re-executes the
plain stage functions from Land over the persisted staging; model calls repeat as ledger
replays, so only a new prompt reaches the provider and the registry counts only those.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sema.intake.decisions import DecisionRejectedError
from sema.intake.ledger import Ledger, RunLedger
from sema.intake.verify.base import Ambiguity
from tests.unit.intake.decisions_env import flat_list, open_entries, presented, write_decisions
from tests.unit.intake.graph_env import (
    ACRIS_DESCRIPTOR,
    ASKED_COLUMNS,
    CODEBOOK,
    LAYOUT_PROPOSAL,
    ROLES_CHOICE,
    STAGE_NODES,
    STRUCTURE_BATCH,
    GraphRun,
    ProcessExit,
    approve,
    asks_for_values,
    choose_flat_lists,
    code_tables,
    discover,
    fills_down,
    graph,
    mixed_source,
    model_names,
    routed_client,
    settings,
)

pytestmark = pytest.mark.unit

STRUCTURE, HUMAN = "structure", "human"
EXIT_RUN_ID = "run-exits-inside-roles"


def _asked(root: Path) -> GraphRun:
    client = routed_client({LAYOUT_PROPOSAL: [asks_for_values, fills_down]})
    return discover(root, mixed_source(root), ACRIS_DESCRIPTOR, client)


def _open_structure(run: GraphRun) -> list[Ambiguity]:
    """Open structure entries as Review presented them in the ledger's ambiguity lists."""
    ledger: RunLedger = Ledger(run.root).run(run.run_id)
    return open_entries(presented(ledger), STRUCTURE)


def _bound_sources(run: GraphRun) -> list[str]:
    return [entry["source"] for entry in run.artifact("structure")["bound"]]


def test_needs_values_routes_to_the_value_interrupt(tmp_path: Path) -> None:
    run = _asked(tmp_path)

    assert run.outcome.node == "value_ask"
    (request,) = run.outcome.payload["requests"]
    assert request["relation"].rpartition(".")[2] == CODEBOOK
    assert tuple(request["columns"]) == ASKED_COLUMNS
    row = run.row()
    assert (row.stage, row.status, row.reason) == ("value_ask", "waiting_review", None)
    assert run.nodes()[-2:] == ["propose_layouts", "value_ask"]
    assert "roles" not in run.nodes()
    assert model_names(run.client) == [STRUCTURE_BATCH, LAYOUT_PROPOSAL]


def test_value_approval_replays_every_repeated_call(tmp_path: Path) -> None:
    run = _asked(tmp_path)
    mark, asked_before = run.mark(), run.client.invoke.call_count

    outcome = graph().approve_values(run.run_id, client=run.client, **settings(tmp_path))

    assert outcome.node == "review"
    new_calls = model_names(run.client)[asked_before:]
    assert STRUCTURE_BATCH not in new_calls
    assert new_calls.count(LAYOUT_PROPOSAL) == 1
    assert STRUCTURE_BATCH in run.replayed(mark)
    assert [node for node in run.nodes(mark) if node != "value_ask"][0] == "propose_layouts"
    assert run.row().model_calls == run.client.invoke.call_count
    assert "model" in _bound_sources(run)
    assert run.artifact("run_metrics")["value_asks"] == 1


def test_decisions_reenter_through_decide_with_zero_calls(tmp_path: Path) -> None:
    run = discover(tmp_path, code_tables(tmp_path), ACRIS_DESCRIPTOR, routed_client())
    listed = _open_structure(run)
    assert len(listed) == 5
    assert run.artifact("run_metrics")["open_structure"] == 5
    rows = [(STRUCTURE, entry.subject, flat_list(entry)) for entry in listed]
    path = write_decisions(tmp_path / "decisions.yaml", rows)
    before, mark, calls = run.counters(), run.mark(), run.client.invoke.call_count

    outcome = approve(run, decisions=path)

    assert outcome.node == "review"
    assert run.client.invoke.call_count == calls
    assert run.counters() == before
    assert "decide" in run.nodes(mark)
    assert not {"resolve_structure", "resolve_roles", "import"} & set(run.nodes(mark))
    assert run.nodes(mark)[-1] == "review"
    assert len(run.artifact("decisions")["decisions"]) == 5
    assert _open_structure(run) == []
    assert _bound_sources(run).count(HUMAN) == 5
    metrics = run.artifact("run_metrics")
    assert metrics["open_structure"] == 0
    assert metrics["model_calls"] == run.row().model_calls
    row = run.row()
    assert (row.stage, row.status) == ("review", "waiting_review")


def test_decision_outside_the_members_is_rejected_before_anything_reruns(
    tmp_path: Path,
) -> None:
    run = discover(tmp_path, code_tables(tmp_path), ACRIS_DESCRIPTOR, routed_client())
    entry = _open_structure(run)[0]
    path = write_decisions(tmp_path / "bad.yaml", [(STRUCTURE, entry.subject, "not_a_member")])
    mark, calls = run.mark(), run.client.invoke.call_count

    with pytest.raises(DecisionRejectedError) as rejected:
        approve(run, decisions=path)

    assert rejected.value.members == entry.members
    assert not set(STAGE_NODES) & set(run.nodes(mark))
    assert not run.has_artifact("decisions")
    assert run.client.invoke.call_count == calls
    row = run.row()
    assert (row.stage, row.status) == ("review", "waiting_review")


def test_resume_after_exit_reexecutes_from_land_and_continues(tmp_path: Path) -> None:
    source = code_tables(tmp_path)
    dying = routed_client({STRUCTURE_BATCH: [choose_flat_lists]}, exit_on=ROLES_CHOICE)
    with pytest.raises(ProcessExit):
        graph().discover(
            source, descriptor=ACRIS_DESCRIPTOR, client=dying, run_id=EXIT_RUN_ID,
            **settings(tmp_path),
        )
    run = GraphRun(tmp_path, EXIT_RUN_ID, routed_client())
    assert model_names(dying)[0] == STRUCTURE_BATCH
    assert run.nodes()[-1] == "resolve_roles"
    before, mark = run.counters(), run.mark()
    assert before[0] == 1

    run.outcome = graph().resume(EXIT_RUN_ID, client=run.client, **settings(tmp_path))

    assert run.outcome.node == "review"
    (resumed,) = run.events("resume")
    assert resumed["stage"] == "resolve_roles"
    later = run.nodes(mark)
    assert later[0] == "resolve_roles"
    assert not {"land", "structure", "resolve_structure", "propose_layouts", "roles"} & set(
        later
    )
    assert STRUCTURE_BATCH not in model_names(run.client)
    assert STRUCTURE_BATCH in run.replayed(mark)
    assert run.row().model_calls - before[0] == run.client.invoke.call_count
    artifacts = sorted(p.name for p in (run.ledger_dir / "artifacts").glob("structure*.json"))
    assert artifacts.count("structure.json") == 1


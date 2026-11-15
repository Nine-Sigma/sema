"""``no_template`` is not a decision choice (ingestion-graph, task 14.4b).

Pinned contract (ingestion-graph "no_template is not a decision choice", scenario
"no_template is not a choice"):

- ``validate_decisions`` rejects a structure decision whose choice is the ``no_template``
  sentinel with ``DecisionRejectedError`` naming the entry and pointing to ``IGNORE``; the
  members it reports never include the sentinel.
- Review presents a ``no_template`` entry with its reason and offers ``IGNORE``, never
  ``no_template``, in the payload and in the ``structure_ambiguities`` artifact.
- ``graph.approve(..., decisions=...)`` and ``sema target approve --decisions`` reject the
  sentinel before anything re-runs; ``IGNORE`` on the same entry closes it with zero calls.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from sema.intake.run.decisions import IGNORE, DecisionRejectedError, validate_decisions
from sema.intake.verify.base import Ambiguity
from tests.unit.cli_intake_env import Cli, decide, intake_cli, snapshot
from tests.unit.intake.decisions_env import decision, write_decisions
from tests.unit.intake.graph_env import A2_DESCRIPTOR, STAGE_NODES, GraphRun, approve
from tests.unit.intake.graph_env import routed_client
from tests.unit.intake.graph_sources_env import discover_with

pytestmark = pytest.mark.unit

STRUCTURE, HUMAN, NO_TEMPLATE = "structure", "human", "no_template"
WIDE_CAP = 50
# One repeating text column: no ignore reason and no template reads it.
ONE_COLUMN_CSV = "note\nalpha\nalpha\nbeta\nbeta\ngamma\n"
SUBJECT = "land_inputs.notes"
UNFIT = "no template over any view reads this relation"


def _source(root: Path) -> Path:
    directory = root / "inputs"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "notes.csv").write_text(ONE_COLUMN_CSV, encoding="utf-8")
    return directory


def _reviewed(root: Path) -> GraphRun:
    found = discover_with(
        root, _source(root), A2_DESCRIPTOR, routed_client(), config={"max_calls": WIDE_CAP}
    )
    assert found.outcome.node == "review", found.outcome.node
    return found


def _listed(entries: list[dict[str, Any]]) -> dict[str, Any]:
    (entry,) = [e for e in entries if e["stage"] == STRUCTURE and e["subject"] == SUBJECT]
    return entry


# --- unit: validation ------------------------------------------------------------------


def test_validation_rejects_no_template_naming_the_entry_and_pointing_to_ignore() -> None:
    listed = Ambiguity(stage=STRUCTURE, subject=SUBJECT, members=(NO_TEMPLATE,), reason=UNFIT)

    with pytest.raises(DecisionRejectedError) as raised:
        validate_decisions([decision(STRUCTURE, SUBJECT, NO_TEMPLATE)], [listed])

    message = str(raised.value)
    assert SUBJECT in message and IGNORE in message, message
    assert NO_TEMPLATE not in raised.value.members


def test_validation_still_accepts_ignore_on_a_no_template_entry() -> None:
    listed = Ambiguity(stage=STRUCTURE, subject=SUBJECT, members=(NO_TEMPLATE,), reason=UNFIT)
    chosen = [decision(STRUCTURE, SUBJECT, IGNORE)]

    assert validate_decisions(chosen, [listed]) == tuple(chosen)


# --- graph: Review presents ignore, Decide rejects the sentinel ------------------------


def test_review_offers_ignore_not_no_template_and_keeps_the_reason(tmp_path: Path) -> None:
    found = _reviewed(tmp_path)

    presented = _listed(found.outcome.payload["ambiguities"])
    written = _listed(found.artifact("structure_ambiguities")["ambiguities"])

    for entry in (presented, written):
        assert NO_TEMPLATE not in entry["members"], entry
        assert IGNORE in entry["members"], entry
        assert entry["reason"], entry
        assert (entry["choice"], entry["source"]) == (None, None)


def test_graph_no_template_decision_is_rejected_and_nothing_reruns(tmp_path: Path) -> None:
    found = _reviewed(tmp_path)
    calls, mark = found.client.invoke.call_count, found.mark()
    path = write_decisions(tmp_path / "sentinel.yaml", [(STRUCTURE, SUBJECT, NO_TEMPLATE)])

    with pytest.raises(DecisionRejectedError) as raised:
        approve(found, decisions=path)

    assert SUBJECT in str(raised.value) and IGNORE in str(raised.value), str(raised.value)
    assert not set(STAGE_NODES) & set(found.nodes(mark))
    assert found.client.invoke.call_count == calls
    assert not found.has_artifact("decisions")


def test_graph_ignore_on_a_no_template_entry_closes_it_with_zero_calls(tmp_path: Path) -> None:
    found = _reviewed(tmp_path)
    calls = found.client.invoke.call_count
    path = write_decisions(tmp_path / "ignore.yaml", [(STRUCTURE, SUBJECT, IGNORE)])

    approve(found, decisions=path)

    assert found.client.invoke.call_count == calls
    entry = _listed(found.outcome.payload["ambiguities"])
    assert (entry["choice"], entry["source"]) == (None, HUMAN)


# --- CLI: approve --decisions validates first ------------------------------------------


@pytest.fixture
def cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Cli:
    return intake_cli(tmp_path / "runs", monkeypatch)


def test_cli_no_template_decision_fails_pointing_to_ignore_and_changes_nothing(
    cli: Cli,
) -> None:
    found = cli.discover(_source(cli.root.parent), A2_DESCRIPTOR)
    path = write_decisions(cli.root.parent / "sentinel.yaml", [(STRUCTURE, SUBJECT, NO_TEMPLATE)])
    before = snapshot(found)

    result = cli.run("approve", found.run_id, "--approver", "reviewer", "--decisions", path)

    assert result.exit_code != 0
    assert SUBJECT in result.output and IGNORE in result.output, result.output
    assert snapshot(found) == before
    assert not found.has_artifact("decisions")


def test_cli_ignore_on_a_no_template_entry_settles_through_decide(cli: Cli) -> None:
    found = cli.discover(_source(cli.root.parent), A2_DESCRIPTOR)

    printed = decide(cli, found, [(STRUCTURE, SUBJECT, IGNORE)], "ignore.yaml")

    assert printed["stopped_at"] == "review"
    assert printed["open_entries"] == 0

"""`sema target discover`, `approve`, and `runs` over the ingestion graph, with a mocked model.

Pinned CLI surface (task 13.2), every command under ``sema target``, in ``sema.cli_intake``:

- ``discover SOURCE [--descriptor YAML] [--hint SPEC] [--schema NAME]
  [--tier metadata_only|bounded_values|full] [--enrich] [--run-id ID] [--root DIR]`` calls
  ``sema.intake.graph.discover(source, root=, descriptor=, config=, client=, hint=,
  enrich=, tier=, run_id=, schema=)``. Defaults: tier ``metadata_only``, enrichment off, no
  hint, schema and run id left to the graph. ``--root`` is the ledger root (default
  ``.sema/intake``); the tests always pass it. Without ``--descriptor`` the single
  ``*.run.yaml`` in the source directory is used; none or several fail non-zero naming the
  directory and the files found, before any run is registered.
- ``approve RUN_ID --approver NAME [--spec YAML] [--decisions YAML] [--reject] [--values]
  [--root DIR]`` calls ``graph.approve`` (with ``spec=`` / ``decisions=``), with
  ``--reject`` ``graph.reject``, with ``--values`` ``graph.approve_values``. Every refusal
  happens before the graph is called and writes nothing: a run whose registry status is not
  ``waiting_review`` fails naming the run id and its status; a run at ``value_ask`` (registry
  stage) refuses a plain approve naming ``value_ask`` and ``--values``; ``--values`` on a
  run not at ``value_ask`` fails naming its status. ``--reject`` with ``--spec``,
  ``--decisions`` or ``--values`` is a usage error (exit 2) naming both options. An
  unregistered run id fails naming it. A ``DecisionRejectedError`` fails non-zero with its
  message (the entry's members) and nothing re-runs.
- discover and approve print one JSON object: ``run_id``; ``stopped_at`` (the interrupt the
  run stopped at, ``review`` or ``value_ask``, or null once it ended); ``status`` (the
  registry status after the call); and, when stopped at Review, ``open_entries`` (open
  structure and roles entries Review presented: no choice and no source) and
  ``run_metrics`` (the run's ``run_metrics`` artifact, which carries ``bound_fraction``);
  when stopped at ``value_ask``, ``requests`` (the interrupt payload's requests).
- ``runs [--status S] [--stage S] [--source S] [--metrics RUN_ID] [--root DIR]`` prints a
  JSON list, newest first, one object per run with keys ``RUN_KEYS``; cost is printed as
  ``cost_estimate`` because it comes from the configured price table. ``--metrics`` prints
  that run's ``run_metrics`` artifact as JSON, and fails naming the run when it has none.
- The model: each discover or approve builds ``IntakeModelConfig()`` from the environment
  and passes ``_intake_client(config)`` to the graph as ``client``. The CLI imports
  ``sema.intake.graph`` inside the command, so ``sema --help`` loads no LangGraph.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from sema.intake.registry import Registry
from sema.models.config import IntakeModelConfig
from tests.unit.cli_intake_env import (
    APPROVER,
    MAX_COST,
    MID_RUN,
    NEW_RUN,
    OLD_RUN,
    RUN_KEYS,
    SEEDED,
    SEEDED_METRICS,
    SOURCE_A,
    DISCOVER_IDS,
    DISCOVER_OPTIONS,
    Cli,
    asked_for_values,
    decide,
    descriptor_dir,
    intake_cli,
    listed_ids,
    open_count,
    open_listed,
    seed_runs,
    snapshot,
    spy_discover,
)
from tests.unit.intake.decisions_env import (
    flat_list,
    hand_written_roles,
    reviewer_choice,
    write_decisions,
)
from tests.unit.intake.graph_env import (
    A2_DESCRIPTOR,
    ACRIS_DESCRIPTOR,
    ASKED_COLUMNS,
    CODEBOOK,
    MAX_CALLS,
    REGISTRY_FILE,
    code_tables,
    routed_client,
)
from tests.unit.intake.structure_env import A2_WORKBOOK, ACRIS

pytestmark = pytest.mark.unit

STRUCTURE, ROLES, HUMAN = "structure", "roles", "human"
ACRIS_SCHEMA = "land_acris"


@pytest.fixture
def cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Cli:
    return intake_cli(tmp_path / "runs", monkeypatch)


@pytest.fixture
def registry_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Cli:
    return intake_cli(tmp_path / "runs", monkeypatch, model=False)


def test_discover_acris_registers_a_run_stopped_at_review_and_prints_the_summary(
    cli: Cli,
) -> None:
    printed = cli.ok("discover", ACRIS)

    run = cli.graph_run(printed["run_id"])
    row = run.row()
    assert (row.stage, row.status) == ("review", "waiting_review")
    assert (printed["stopped_at"], printed["status"]) == ("review", "waiting_review")
    metrics = run.artifact("run_metrics")
    assert printed["run_metrics"] == metrics
    assert 0.0 <= printed["run_metrics"]["bound_fraction"] <= 1.0
    assert printed["open_entries"] == open_count(run) > 0
    assert printed["open_entries"] == metrics["open_structure"] + metrics["open_roles"]
    assert row.model_calls == cli.client.invoke.call_count >= 1
    (config,) = cli.configs
    assert isinstance(config, IntakeModelConfig)
    assert (config.max_calls, config.max_cost) == (MAX_CALLS, MAX_COST)


@pytest.mark.parametrize("count", [0, 2], ids=["none", "many"])
def test_discover_without_a_descriptor_needs_exactly_one_in_the_source(
    cli: Cli, count: int
) -> None:
    source = descriptor_dir(cli.root.parent, count)

    result = cli.run("discover", source)

    assert result.exit_code != 0
    assert source.name in result.output
    assert all(path.name in result.output for path in source.glob("*.run.yaml"))
    assert Registry(cli.root / REGISTRY_FILE).list() == ()
    assert cli.client.invoke.call_count == 0


@pytest.mark.parametrize(("options", "expected"), DISCOVER_OPTIONS, ids=DISCOVER_IDS)
def test_discover_threads_its_options_to_the_graph(
    cli: Cli,
    monkeypatch: pytest.MonkeyPatch,
    options: tuple[str | Path, ...],
    expected: dict[str, Any],
) -> None:
    seen = spy_discover(monkeypatch)

    cli.discover(A2_WORKBOOK, A2_DESCRIPTOR, *options)

    ((source, kwargs),) = seen
    assert source == A2_WORKBOOK
    assert Path(kwargs["root"]) == cli.root
    assert Path(kwargs["descriptor"]) == A2_DESCRIPTOR
    assert kwargs["client"] is cli.client
    assert isinstance(kwargs["config"], IntakeModelConfig)
    hint = kwargs.get("hint")
    found = {**kwargs, "hint": Path(hint) if hint is not None else None}
    assert {name: found.get(name) for name in expected} == expected


def test_approve_imports_and_records_the_approver(cli: Cli) -> None:
    run = cli.discover(A2_WORKBOOK, A2_DESCRIPTOR)

    printed = cli.approve(run)

    assert (printed["run_id"], printed["stopped_at"]) == (run.run_id, None)
    assert printed["status"] == "approved"
    row = run.row()
    assert (row.status, row.approver, row.stage) == ("approved", APPROVER, "import")
    assert (run.ledger_dir / "import" / "manifest.yaml").is_file()


def test_approve_with_an_edited_spec_presents_review_again_with_zero_calls(
    cli: Cli,
) -> None:
    run = cli.discover(A2_WORKBOOK, A2_DESCRIPTOR)
    edited = run.artifact("spec")
    edited["value_sets"]["roles"]["replaced_by"] = "successor"
    path = cli.root.parent / "edited.spec.yaml"
    path.write_text(yaml.safe_dump({"concept_source": edited}, sort_keys=False))
    before = run.counters()
    cli.client = routed_client()

    printed = cli.approve(run, "--spec", path)

    assert (printed["stopped_at"], printed["status"]) == ("review", "waiting_review")
    assert cli.client.invoke.call_count == 0
    assert run.counters() == before
    assert run.artifact("spec")["value_sets"]["roles"]["replaced_by"] == "successor"
    assert not (run.ledger_dir / "import").exists()


def test_decisions_settle_open_entries_and_a_later_approve_imports(cli: Cli) -> None:
    run = cli.discover(ACRIS, ACRIS_DESCRIPTOR, "--schema", ACRIS_SCHEMA)
    structure = open_listed(run, STRUCTURE)
    assert structure
    before = run.counters()

    rows = [(STRUCTURE, entry.subject, flat_list(entry)) for entry in structure]
    first = decide(cli, run, rows, "structure.decisions.yaml")

    assert (first["stopped_at"], first["status"]) == ("review", "waiting_review")
    assert first["open_entries"] == open_count(run)
    assert open_listed(run, STRUCTURE) == []
    bound = [entry["source"] for entry in run.artifact("structure")["bound"]]
    assert bound.count(HUMAN) == len(structure)
    stated = hand_written_roles()
    chosen = [(entry.subject, reviewer_choice(entry, stated)) for entry in open_listed(run, ROLES)]
    rows = [(ROLES, subject, choice) for subject, choice in chosen if choice is not None]
    assert rows
    second = decide(cli, run, rows, "roles.decisions.yaml")
    assert second["stopped_at"] == "review"
    assert run.counters() == before

    printed = cli.approve(run)

    assert (printed["stopped_at"], printed["status"]) == (None, "approved")
    assert (run.ledger_dir / "import" / "manifest.yaml").is_file()


def test_a_decision_outside_the_members_fails_naming_them_and_changes_nothing(
    cli: Cli,
) -> None:
    run = cli.discover(code_tables(cli.root.parent), ACRIS_DESCRIPTOR)
    entry = open_listed(run, STRUCTURE)[0]
    path = write_decisions(
        cli.root.parent / "bad.yaml", [(STRUCTURE, entry.subject, "not_a_member")]
    )
    before = snapshot(run)

    result = cli.run("approve", run.run_id, "--approver", APPROVER, "--decisions", path)

    assert result.exit_code != 0
    assert "not_a_member" in result.output
    assert flat_list(entry) in result.output
    assert snapshot(run) == before
    assert not run.has_artifact("decisions")


def test_reject_records_the_approver_and_imports_nothing(cli: Cli) -> None:
    run = cli.discover(A2_WORKBOOK, A2_DESCRIPTOR)

    printed = cli.approve(run, "--reject")

    assert (printed["stopped_at"], printed["status"]) == (None, "rejected")
    row = run.row()
    assert (row.status, row.approver) == ("rejected", APPROVER)
    assert not (run.ledger_dir / "import").exists()


@pytest.mark.parametrize("options", [(), ("--reject",)], ids=["approve", "reject"])
def test_a_run_not_at_review_is_refused_naming_its_status(
    cli: Cli, options: tuple[str, ...]
) -> None:
    run = cli.discover(A2_WORKBOOK, A2_DESCRIPTOR)
    cli.approve(run)
    before = snapshot(run)
    cli.client = routed_client()

    result = cli.run("approve", run.run_id, "--approver", "second-reviewer", *options)

    assert result.exit_code != 0
    assert "approved" in result.output
    assert run.run_id in result.output
    assert snapshot(run) == before
    assert cli.client.invoke.call_count == 0


def test_values_resumes_a_run_at_value_ask_and_prints_the_next_outcome(cli: Cli) -> None:
    run, asked = asked_for_values(cli)
    assert (asked["stopped_at"], asked["status"]) == ("value_ask", "waiting_review")
    (request,) = asked["requests"]
    assert request["relation"].rpartition(".")[2] == CODEBOOK
    assert tuple(request["columns"]) == ASKED_COLUMNS

    printed = cli.approve(run, "--values")

    assert (printed["stopped_at"], printed["status"]) == ("review", "waiting_review")
    assert printed["run_metrics"]["value_asks"] == 1
    assert run.row().stage == "review"


def test_a_plain_approve_at_value_ask_is_refused_pointing_at_values(cli: Cli) -> None:
    run, _ = asked_for_values(cli)
    before, calls = snapshot(run), cli.client.invoke.call_count

    result = cli.run("approve", run.run_id, "--approver", APPROVER)

    assert result.exit_code != 0
    assert "value_ask" in result.output
    assert "--values" in result.output
    assert snapshot(run) == before
    assert cli.client.invoke.call_count == calls


@pytest.mark.parametrize("approved", [False, True], ids=["at-review", "approved"])
def test_values_on_a_run_not_at_value_ask_is_refused_naming_its_status(
    cli: Cli, approved: bool
) -> None:
    run = cli.discover(A2_WORKBOOK, A2_DESCRIPTOR)
    if approved:
        cli.approve(run)
    before = snapshot(run)
    cli.client = routed_client()

    result = cli.run("approve", run.run_id, "--approver", APPROVER, "--values")

    assert result.exit_code != 0
    assert run.row().status in result.output
    assert run.run_id in result.output
    assert snapshot(run) == before
    assert cli.client.invoke.call_count == 0


@pytest.mark.parametrize("conflict", ["--spec", "--decisions", "--values"])
def test_reject_with_another_approve_option_is_a_usage_error(
    registry_cli: Cli, conflict: str
) -> None:
    registry = seed_runs(registry_cli.root)
    before = registry.get(MID_RUN)
    given = registry_cli.root.parent / "given.yaml"
    given.write_text("[]\n")
    value = () if conflict == "--values" else (given,)

    result = registry_cli.run(
        "approve", MID_RUN, "--approver", APPROVER, "--reject", conflict, *value
    )

    assert result.exit_code == 2
    assert "--reject" in result.output
    assert conflict in result.output
    assert registry.get(MID_RUN) == before


def test_approve_an_unregistered_run_fails_naming_it(registry_cli: Cli) -> None:
    seed_runs(registry_cli.root)

    result = registry_cli.run("approve", "run-never-registered", "--approver", APPROVER)

    assert result.exit_code != 0
    assert "run-never-registered" in result.output


def test_runs_lists_newest_first_with_reason_and_cost_estimate(registry_cli: Cli) -> None:
    registry = seed_runs(registry_cli.root)

    printed = registry_cli.ok("runs")

    assert listed_ids(printed) == [NEW_RUN, MID_RUN, OLD_RUN]
    assert all(tuple(row) == RUN_KEYS for row in printed)
    by_id = {row["run_id"]: row for row in printed}
    assert by_id[MID_RUN]["reason"] == "budget_exceeded"
    assert by_id[OLD_RUN]["reason"] is None
    stored = registry.get(MID_RUN)
    assert by_id[MID_RUN]["started_at"] == stored.started_at.isoformat()
    assert by_id[MID_RUN]["updated_at"] == stored.updated_at.isoformat()
    assert by_id[MID_RUN]["cost_estimate"] == stored.cost
    assert {row["run_id"]: row["status"] for row in printed} == {
        run_id: status for run_id, _, _, status, _ in SEEDED
    }


@pytest.mark.parametrize(
    ("options", "expected"),
    [
        (("--status", "waiting_review"), [MID_RUN]),
        (("--stage", "roles"), [OLD_RUN]),
        (("--source", SOURCE_A), [NEW_RUN, OLD_RUN]),
        (("--status", "approved", "--source", SOURCE_A), [NEW_RUN]),
    ],
    ids=["status", "stage", "source", "status-and-source"],
)
def test_runs_filters_by_status_stage_and_source(
    registry_cli: Cli, options: tuple[str, ...], expected: list[str]
) -> None:
    seed_runs(registry_cli.root)

    printed = registry_cli.ok("runs", *options)

    assert listed_ids(printed) == expected


def test_runs_with_metrics_prints_the_run_metrics_artifact(registry_cli: Cli) -> None:
    seed_runs(registry_cli.root)

    printed = registry_cli.ok("runs", "--metrics", MID_RUN)

    assert printed == SEEDED_METRICS


def test_runs_with_metrics_for_a_run_without_them_fails_naming_it(registry_cli: Cli) -> None:
    seed_runs(registry_cli.root)

    result = registry_cli.run("runs", "--metrics", OLD_RUN)

    assert result.exit_code != 0
    assert OLD_RUN in result.output

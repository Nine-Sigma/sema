"""Graph follow-up (after task 12.2): the budget stop, Land by medium, and the embedder.

Pinned API changes to ``sema.intake.graph`` beyond ``test_graph.py``:

- A call or cost cap reached inside any node no longer propagates ``BudgetExceededError``
  out of ``discover``. The run stops at the Review interrupt: ``RunOutcome.node`` is
  ``"review"`` with the full Review payload, the registry row is ``review`` /
  ``waiting_review`` / ``budget_exceeded`` with the calls made before the cap, and
  ``run_metrics`` is written before Review (intake-model-calls "Call cap reached").
  A run stopped there is an ordinary Review stop: ``resume`` returns the same Review,
  ``approve(..., decisions=...)`` and ``approve(..., spec=...)`` re-enter through
  ``decide``, and none of them reaches the provider again.
- Land picks the existing loader for the source's medium: a semantic-sql SQLite file
  lands through ``land_semsql``, an Athena bundle directory through
  ``land_athena_bundle``, a DDL file through ``land_ddl``, and workbooks, CSVs, and
  directories of them through ``land_tabular``. The ``land_result`` artifact names the
  medium. A source no loader reads fails at Land with a ``ValueError`` naming the source,
  makes no model call, leaves the registry row at ``land`` / ``failed`` / ``no_loader``,
  and writes nothing past Land.
- ``discover(..., embedder=None)``: a supplied embedder reaches Roles and its resolver,
  which rank eligible columns with it under the module's measured margin floor. With no
  embedder nothing changes. ``approve``, ``reject``, ``approve_values``, and ``resume`` take
  the same ``embedder=None`` keyword and pass it to the stages they rebuild, so a run
  recomputed at ``decide`` or Import ranks with the embedder it was discovered with.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from sema.intake.model_calls import BudgetExceededError
from sema.intake.ledger import Ledger
from tests.unit.intake.decisions_env import flat_list, open_entries, presented, write_decisions
from tests.unit.intake.eligibility_env import clear_embedder
from tests.unit.intake.graph_env import (
    A2_DESCRIPTOR,
    ACRIS_DESCRIPTOR,
    APPROVER,
    REVIEW_KEYS,
    STRUCTURE_BATCH,
    GraphRun,
    choose_flat_lists,
    code_tables,
    graph,
    model_names,
    routed_client,
)
from tests.unit.intake.graph_sources_env import (
    A3_DESCRIPTOR,
    A3_RAW_ONTOLOGY,
    DDL,
    KEYED_HEADER,
    athena_bundle,
    definition_embedder,
    definitions,
    discover_with,
    embedding_hints,
    ranked_source,
    role_entry,
)
from tests.unit.intake.model_calls_env import intake_config
from tests.unit.intake.structure_env import A2_WORKBOOK, A3_DB, ACRIS, ACRIS_SPEC

pytestmark = pytest.mark.unit

ACRIS_SCHEMA = "land_acris"
CALL_CAP = 3
COST_CAP = 0.01
WIDE_CALL_CAP = 200
BUDGET_ROW = ("review", "waiting_review", "budget_exceeded")
REVIEW_ROW = ("review", "waiting_review", None)
EMBEDDING_CHOICE = "zorbit_text"
NO_LOADER_RUN = "run-no-loader"


def _capped(root: Path, **config: float) -> GraphRun:
    return discover_with(
        root, ACRIS, ACRIS_DESCRIPTOR, routed_client(),
        config=config, hint=ACRIS_SPEC, schema=ACRIS_SCHEMA,
    )


# --- 1. The budget stops the run at Review --------------------------------------------


def test_call_cap_stops_the_run_at_review_with_budget_exceeded(tmp_path: Path) -> None:
    try:
        run = _capped(tmp_path, max_calls=CALL_CAP)
    except BudgetExceededError as error:
        pytest.fail(f"the budget stop propagated out of discover: {error}")

    assert run.outcome.node == "review"
    assert tuple(run.outcome.payload) == REVIEW_KEYS
    assert run.client.invoke.call_count == CALL_CAP
    row = run.row()
    assert (row.stage, row.status, row.reason) == BUDGET_ROW
    assert row.model_calls == CALL_CAP
    assert run.nodes()[-1] == "review"


def test_budget_stop_writes_metrics_equal_to_the_registry_row(tmp_path: Path) -> None:
    try:
        run = _capped(tmp_path, max_calls=CALL_CAP)
    except BudgetExceededError as error:
        pytest.fail(f"the budget stop propagated out of discover: {error}")

    metrics = run.artifact("run_metrics")
    row = run.row()
    assert (metrics["model_calls"], metrics["tokens"]) == (row.model_calls, row.tokens)
    assert metrics["cost"] == pytest.approx(row.cost)
    assert run.outcome.payload["run_metrics"]["model_calls"] == CALL_CAP


def test_cost_cap_stops_the_run_at_review_with_budget_exceeded(tmp_path: Path) -> None:
    try:
        run = _capped(tmp_path, max_calls=WIDE_CALL_CAP, max_cost=COST_CAP)
    except BudgetExceededError as error:
        pytest.fail(f"the budget stop propagated out of discover: {error}")

    row = run.row()
    assert run.outcome.node == "review"
    assert (row.stage, row.status, row.reason) == BUDGET_ROW
    assert run.client.invoke.call_count == row.model_calls
    assert row.cost >= COST_CAP


def test_a_run_stopped_by_the_budget_can_be_rejected_at_review(tmp_path: Path) -> None:
    try:
        run = _capped(tmp_path, max_calls=CALL_CAP)
    except BudgetExceededError as error:
        pytest.fail(f"the budget stop propagated out of discover: {error}")

    outcome = graph().reject(
        run.run_id, root=tmp_path, config=intake_config(max_calls=CALL_CAP),
        approver=APPROVER, client=run.client,
    )

    assert outcome.node is None
    assert run.row().status == "rejected"
    assert run.client.invoke.call_count == CALL_CAP


def _settle(run: GraphRun, cap: int, **options: object) -> object:
    return graph().approve(
        run.run_id, root=run.root, config=intake_config(max_calls=cap),
        approver=APPROVER, client=run.client, **options,
    )


def test_resume_after_a_budget_stop_returns_review_without_a_call(tmp_path: Path) -> None:
    try:
        run = _capped(tmp_path, max_calls=CALL_CAP)
    except BudgetExceededError as error:
        pytest.fail(f"the budget stop propagated out of discover: {error}")

    outcome = graph().resume(
        run.run_id, root=tmp_path, config=intake_config(max_calls=CALL_CAP), client=run.client
    )

    assert outcome.node == "review"
    assert run.client.invoke.call_count == CALL_CAP
    row = run.row()
    assert (row.stage, row.status, row.reason, row.model_calls) == (*BUDGET_ROW, CALL_CAP)


def test_decisions_settle_a_budget_stop_with_zero_calls(tmp_path: Path) -> None:
    client = routed_client()
    try:
        run = discover_with(
            tmp_path, code_tables(tmp_path), ACRIS_DESCRIPTOR, client, config={"max_calls": 0}
        )
    except BudgetExceededError as error:
        pytest.fail(f"the budget stop propagated out of discover: {error}")
    assert run.outcome.node == "review"
    listed = open_entries(presented(Ledger(tmp_path).run(run.run_id)), "structure")
    assert listed, "Review after a budget stop presents the open structure entries"
    rows = [("structure", entry.subject, flat_list(entry)) for entry in listed]
    path = write_decisions(tmp_path / "decisions.yaml", rows)

    outcome = _settle(run, 0, decisions=path)

    assert outcome.node == "review"
    assert client.invoke.call_count == 0
    assert run.row().model_calls == 0
    assert "decide" in run.nodes()


def test_edited_spec_settles_a_budget_stop_with_zero_calls(tmp_path: Path) -> None:
    try:
        run = _capped(tmp_path, max_calls=CALL_CAP)
    except BudgetExceededError as error:
        pytest.fail(f"the budget stop propagated out of discover: {error}")
    mark = run.mark()

    outcome = _settle(run, CALL_CAP, spec=ACRIS_SPEC)

    assert outcome.node == "review"
    assert outcome.payload["spec"] is not None
    assert run.client.invoke.call_count == CALL_CAP
    assert run.row().model_calls == CALL_CAP
    assert run.nodes(mark)[0] == "decide"
    assert run.nodes(mark)[-1] == "review"


# --- 2. Land picks the existing loader by medium --------------------------------------


def test_semsql_file_lands_through_its_loader_and_reaches_review(tmp_path: Path) -> None:
    client = routed_client()
    run = discover_with(tmp_path, A3_DB, A3_DESCRIPTOR, client)

    assert run.artifact("land_result")["medium"] == "semsql"
    assert run.outcome.node == "review"
    assert run.outcome.payload["spec"] is not None
    assert run.outcome.payload["dry_run"] is not None
    assert model_names(client) == []
    row = run.row()
    assert (row.stage, row.status, row.reason) == REVIEW_ROW


def test_athena_bundle_lands_through_its_loader_and_reaches_review(tmp_path: Path) -> None:
    source = athena_bundle(tmp_path)
    client = routed_client()
    run = discover_with(tmp_path, source.path, source.descriptor, client)

    assert run.artifact("land_result")["medium"] == "athena"
    assert run.outcome.node == "review"
    assert run.outcome.payload["spec"] is not None
    assert model_names(client) == []
    assert all(e["payload"]["status"] == "pass" for e in run.events("verdict"))


def test_ddl_file_lands_through_the_dictionary_loader(tmp_path: Path) -> None:
    source = athena_bundle(tmp_path)
    run = discover_with(tmp_path, DDL, source.descriptor, routed_client())

    assert run.artifact("land_result")["medium"] == "dictionary"
    assert run.outcome.node == "review"


def test_workbook_still_lands_as_tabular(tmp_path: Path) -> None:
    run = discover_with(tmp_path, A2_WORKBOOK, A2_DESCRIPTOR, routed_client())

    assert run.artifact("land_result")["medium"] == "tabular"
    assert run.outcome.node == "review"


def test_source_no_loader_reads_fails_at_land_naming_it(tmp_path: Path) -> None:
    client = routed_client()

    with pytest.raises(ValueError, match=A3_RAW_ONTOLOGY.name) as raised:
        discover_with(tmp_path, A3_RAW_ONTOLOGY, A3_DESCRIPTOR, client, run_id=NO_LOADER_RUN)

    assert not isinstance(raised.value, OSError)
    assert client.invoke.call_count == 0
    run = GraphRun(tmp_path, NO_LOADER_RUN, client)
    row = run.row()
    assert (row.stage, row.status, row.reason) == ("land", "failed", "no_loader")
    assert set(run.nodes()) <= {"land"}
    assert not run.has_artifact("structure")


# --- 3. A supplied embedder reaches Roles ---------------------------------------------


def _ranked(root: Path, **options: object) -> GraphRun:
    client = routed_client(defaults={STRUCTURE_BATCH: choose_flat_lists})
    return discover_with(root, ranked_source(root), A2_DESCRIPTOR, client, **options)


def test_supplied_embedder_records_its_hint_on_the_role_entry(tmp_path: Path) -> None:
    embedder = clear_embedder()

    run = _ranked(tmp_path, embedder=embedder)

    definition = role_entry(run, "definition")
    assert embedder.calls >= 1
    assert embedding_hints(definition)
    assert EMBEDDING_CHOICE in embedding_hints(definition)[0]
    assert (definition["choice"], definition["source"]) == (EMBEDDING_CHOICE, "embedding")


def test_no_embedder_leaves_roles_unchanged(tmp_path: Path) -> None:
    explicit = _ranked(tmp_path / "explicit", embedder=None)
    omitted = _ranked(tmp_path / "omitted")

    for run in (explicit, omitted):
        definition = role_entry(run, "definition")
        assert embedding_hints(definition) == []
        assert definition["choice"] is None
    assert explicit.artifact("roles_ambiguities") == omitted.artifact("roles_ambiguities")


def _decide(
    run: GraphRun, embedder: object, rows: list[tuple[str, str, str]], name: str
) -> object:
    path = write_decisions(run.root / name, rows)
    return _settle(run, WIDE_CALL_CAP, decisions=path, embedder=embedder)


def test_embedder_carries_through_decide_and_import_with_zero_calls(tmp_path: Path) -> None:
    embedder = definition_embedder()
    source = ranked_source(tmp_path, KEYED_HEADER)
    run = discover_with(tmp_path, source, A2_DESCRIPTOR, routed_client(), embedder=embedder)
    calls = run.client.invoke.call_count
    (entry,) = open_entries(presented(Ledger(tmp_path).run(run.run_id)), "structure")

    _decide(run, embedder, [("structure", entry.subject, flat_list(entry))], "structure.yaml")
    section = role_entry(run, "section")
    decided = _decide(run, embedder, [("roles", section["subject"], "concepts")], "roles.yaml")
    assert decided.payload["spec"] is not None
    definition = role_entry(run, "definition")
    imported = _settle(run, WIDE_CALL_CAP, embedder=embedder)

    assert imported.node is None
    assert run.row().status == "approved"
    assert run.client.invoke.call_count == calls
    assert (definition["choice"], definition["source"]) == (EMBEDDING_CHOICE, "embedding")
    assert embedding_hints(definition)
    manifest = yaml.safe_load((run.ledger_dir / "import" / "manifest.yaml").read_text())
    found = definitions(manifest)
    assert found and all(str(text).startswith("Zorbit text") for text in found)

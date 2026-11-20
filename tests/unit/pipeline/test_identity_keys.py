"""Done-when 2 (N4): each loaded relation's key and grade, with zero model calls.

The identity step runs over the identity fixture in DuckDB, through the real connector, with
the fixture's dictionary; only the graph is a stand-in (`StepGraph`). A key fact is read
back from the `has_key` assertions the step stores: the column set, its grade, its evidence
and its alternatives (each `{"columns", "verification"}`). The expected numbers come from the
fixture's rows.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pytest

from tests.support.connector_fixture_identity import (
    IDENTITY_SCHEMA,
    TABLES,
    distinct_count,
    null_row_count,
)
from tests.support.connector_fixture_identity_runs import (
    STEP_RUN_ID,
    StepRun,
    TimesOut,
    prepare_step,
    relation_of,
    run_step,
)

pytestmark = pytest.mark.unit

ACCEPTED = ("proven", "supported")
# Each relation's key and grade: a declared key, a spec list exactly one relation passes, a
# spec list unique on two relations, or none.
KEYS: dict[str, tuple[tuple[str, ...], str] | None] = {
    "dock_kind": (("dock_code",), "proven"), "rig_kind": (("rig_code",), "proven"),
    "berth": (("berth_no",), "proven"), "hull_rev": (("hull_no", "rev_no"), "proven"),
    "hull_survey": (("survey_no",), "proven"), "hull_cargo": (("cargo_no",), "proven"),
    "hull_quote": (("quote_no",), "proven"), "tide_log": (("log_no",), "proven"),
    "mooring": None, "buoy_check": (("check_no",), "proven"),
    "crane_rev": (("crane_rev_id",), "proven"), "gate_tally": None,
    "pump": (("unit_no",), "proven"), "valve": (("unit_no",), "proven"),
    "slip_a": (("slip_id",), "proven"), "slip_b": (("slip_ref",), "supported"),
    "slip_use": (("use_no",), "proven"), "berth_call": (("call_no",), "proven"),
    "deck_plan": (("plan_no",), "proven"),
}


def _identity_step(prepared: StepRun) -> None:
    run_identity = importlib.import_module("sema.pipeline.identity").run_identity
    run_identity(prepared.graph, prepared.connector, prepared.config, [IDENTITY_SCHEMA],
                 STEP_RUN_ID)


def run(directory: Path, times_out: TimesOut | None = None) -> StepRun:
    prepared = prepare_step(directory, times_out) if times_out else prepare_step(directory)
    return run_step(prepared, _identity_step)


@pytest.fixture(scope="module")
def keyed(tmp_path_factory: pytest.TempPathFactory) -> StepRun:
    return run(tmp_path_factory.mktemp("keys"))


def key_payloads(step: StepRun, table: str) -> list[dict[str, Any]]:
    return [a.payload for a in step.built().asserted("has_key")
            if relation_of(a.subject_ref) == table]


def accepted_key(step: StepRun, table: str) -> dict[str, Any] | None:
    found = [p for p in key_payloads(step, table) if p.get("verification") in ACCEPTED]
    assert len(found) <= 1, f"{table}: more than one accepted key {found}"
    return found[0] if found else None


def graded(step: StepRun, table: str) -> dict[tuple[str, ...], str]:
    """Every column set the step graded on the relation, the key and its alternatives."""
    found: dict[tuple[str, ...], str] = {}
    for payload in key_payloads(step, table):
        for fact in (payload, *payload.get("alternatives", [])):
            found[tuple(sorted(fact["columns"]))] = fact["verification"]
    return found


@pytest.mark.parametrize("table", sorted(KEYS))
def test_each_relation_gets_its_key_and_grade(keyed: StepRun, table: str) -> None:
    key = accepted_key(keyed, table)
    expected = KEYS[table]
    if expected is None:
        assert key is None
        return
    assert key is not None, f"{table}: no key"
    columns, grade = expected
    assert (tuple(sorted(key["columns"])), key["verification"]) == (tuple(sorted(columns)), grade)
    rows = len(TABLES[table].rows)
    evidence = key["evidence"]
    assert (evidence["rows"], evidence["distinct"], evidence["null_rows"]) == (
        rows, distinct_count(table, columns), null_row_count(table, columns))


def test_a_proven_key_names_its_signal_and_a_supported_key_none(keyed: StepRun) -> None:
    proven, supported = accepted_key(keyed, "berth"), accepted_key(keyed, "slip_b")
    assert proven is not None and supported is not None
    assert proven["evidence"]["signals"]
    assert supported["evidence"]["signals"] == []


def test_a_column_unique_by_coincidence_is_supported_and_never_displaces_the_key(
    keyed: StepRun,
) -> None:
    assert distinct_count("tide_log", ["reading_mm"]) == len(TABLES["tide_log"].rows)

    key = accepted_key(keyed, "tide_log")

    assert key is not None and key["columns"] == ["log_no"]
    assert graded(keyed, "tide_log")[("reading_mm",)] == "supported"


def test_a_spec_list_unique_on_two_relations_corroborates_neither(keyed: StepRun) -> None:
    slip_b = accepted_key(keyed, "slip_b")
    assert slip_b is not None
    assert (slip_b["columns"], slip_b["verification"]) == (["slip_ref"], "supported")
    assert graded(keyed, "slip_a")[("slip_ref",)] == "supported"


def test_a_declared_key_outranks_a_spec_list(keyed: StepRun) -> None:
    key = accepted_key(keyed, "slip_a")

    assert key is not None and (key["columns"], key["verification"]) == (["slip_id"], "proven")


def test_a_spec_list_exactly_one_relation_passes_is_proven(keyed: StepRun) -> None:
    key = accepted_key(keyed, "rig_kind")

    assert key is not None and (key["columns"], key["verification"]) == (["rig_code"], "proven")
    assert graded(keyed, "rig_kind")[("rig_title",)] == "supported"


def test_a_unique_set_with_a_null_part_is_open_and_writes_no_key(keyed: StepRun) -> None:
    columns = ("pier_code", "slot_no")
    assert distinct_count("mooring", columns) == len(TABLES["mooring"].rows)

    assert graded(keyed, "mooring") == {columns: "open"}
    assert accepted_key(keyed, "mooring") is None
    (payload,) = key_payloads(keyed, "mooring")
    assert payload["evidence"]["null_rows"] == null_row_count("mooring", columns) == 3


def test_a_superset_of_a_key_is_not_a_key(keyed: StepRun) -> None:
    key = accepted_key(keyed, "valve")

    assert key is not None and key["columns"] == ["unit_no"]
    assert graded(keyed, "valve")[("unit_no", "valve_tag")] == "rejected"


def test_a_nominated_set_that_repeats_is_rejected(keyed: StepRun) -> None:
    assert distinct_count("slip_use", ["slip_ref"]) < len(TABLES["slip_use"].rows)

    assert graded(keyed, "slip_use")[("slip_ref",)] == "rejected"
    key = accepted_key(keyed, "slip_use")
    assert key is not None and key["columns"] == ["use_no"]


def test_a_relation_with_no_unique_set_gets_no_key(keyed: StepRun) -> None:
    assert accepted_key(keyed, "gate_tally") is None
    assert accepted_key(keyed, "berth") is not None


def test_no_model_is_called(tmp_path: Path) -> None:
    from unittest.mock import patch

    with patch("sema.llm_client.LLMClient.invoke", side_effect=AssertionError("model called")):
        step = run(tmp_path)

    assert accepted_key(step, "berth") is not None


def test_a_statement_that_times_out_leaves_the_fact_open(tmp_path: Path) -> None:
    step = run(tmp_path, lambda sql: "tide_log" in sql)

    assert accepted_key(step, "tide_log") is None
    assert graded(step, "tide_log").get(("log_no",)) == "open"
    berth = accepted_key(step, "berth")
    assert berth is not None and berth["verification"] == "proven"

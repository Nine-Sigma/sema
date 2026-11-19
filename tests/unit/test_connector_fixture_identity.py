"""The identity fixture holds each case and tempts each decoy's predicate, on its data."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import duckdb
import pytest
from pydantic import BaseModel

from sema.binding.spec import load_concept_source_spec
from sema.connectors.duckdb import DuckDBConnector
from sema.engine.join_detector import COVERAGE_THRESHOLD
from sema.eval.discovery_scorecard_utils import unique_keys, unique_keys_query
from sema.models.config import DuckDBConfig
from sema.models.stages import StageAResult
from tests.support.connector_fixture_identity import (
    CODE_LIST_ROLES,
    CODE_LIST_TEXTS,
    IDENTITY_CATALOG,
    IDENTITY_SCHEMA,
    IDENTITY_TABLES,
    L2_ENTITIES,
    PROPOSALS,
    RIG_GROUP,
    RIG_LIST_TEXTS,
    RIG_TITLES,
    SLIP_REFERENCE,
    TABLES,
    column_values,
    conflict_counts,
    contained_counts,
    distinct_count,
    null_row_count,
    write_identity_database,
    write_identity_dictionary,
)
from tests.support.connector_fixture_identity_runs import (
    IdentityFakeClient,
    proposal_for,
    relation_named,
)

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def database(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_identity_database(tmp_path_factory.mktemp("identity"))


@pytest.fixture
def warehouse(database: Path) -> Iterator[duckdb.DuckDBPyConnection]:
    con = duckdb.connect(str(database), read_only=True)
    yield con
    con.close()


def _one(con: duckdb.DuckDBPyConnection, sql: str) -> int:
    (row,) = con.execute(sql).fetchall()
    return int(row[0])


def _fqn(table: str) -> str:
    return f"{IDENTITY_CATALOG}.{IDENTITY_SCHEMA}.{table}"


def test_the_file_holds_every_row_the_python_spells(warehouse: duckdb.DuckDBPyConnection) -> None:
    for table in IDENTITY_TABLES:
        stored = warehouse.execute(
            f"SELECT {', '.join(table.column_names)} FROM {_fqn(table.name)}"
        ).fetchall()
        assert sorted(stored, key=repr) == sorted(table.rows, key=repr), table.name


def test_the_declared_keys_are_the_ones_the_source_reads(database: Path) -> None:
    connector = DuckDBConnector(DuckDBConfig(path=str(database)))
    try:
        rows = connector.execute_query(unique_keys_query(IDENTITY_CATALOG, IDENTITY_SCHEMA))
    finally:
        connector.close()
    declared = {(t.name, "+".join(sorted(k))) for t in IDENTITY_TABLES
                for k in (t.primary_key, *t.unique) if k}
    assert unique_keys(rows) == declared
    assert ("valve", "unit_no+valve_tag") in declared and ("mooring", "pier_code+slot_no") in declared


@pytest.mark.parametrize("table", [t.name for t in IDENTITY_TABLES])
def test_the_counting_helpers_equal_the_warehouse(
    warehouse: duckdb.DuckDBPyConnection, table: str,
) -> None:
    columns = TABLES[table].column_names
    for width in (1, 2):
        for start in range(len(columns) - width + 1):
            picked = columns[start:start + width]
            tuples = f"SELECT DISTINCT {', '.join(picked)} FROM {_fqn(table)}"
            assert distinct_count(table, picked) == _one(
                warehouse, f"SELECT COUNT(*) FROM ({tuples})"), picked
            nulls = " OR ".join(f"{c} IS NULL" for c in picked)
            assert null_row_count(table, picked) == _one(
                warehouse, f"SELECT COUNT(*) FROM {_fqn(table)} WHERE {nulls}"), picked


def _unique_single_columns(table: str) -> set[str]:
    rows = len(TABLES[table].rows)
    return {c for c in TABLES[table].column_names if distinct_count(table, [c]) == rows}


def test_only_the_named_columns_are_unique_on_their_own() -> None:
    found = {t.name: _unique_single_columns(t.name) for t in IDENTITY_TABLES}
    assert found == {
        "dock_kind": {"dock_code", "dock_title"},
        "rig_kind": {"rig_code", "rig_title"},
        "berth": {"berth_no"}, "hull_rev": set(), "hull_survey": {"survey_no"},
        "hull_cargo": {"cargo_no"}, "hull_quote": {"quote_no"},
        "tide_log": {"log_no", "reading_mm"}, "mooring": set(), "buoy_check": {"check_no"},
        "crane_rev": {"crane_rev_id"}, "gate_tally": set(), "pump": {"unit_no"},
        "valve": {"unit_no"}, "slip_a": {"slip_id", "slip_ref"}, "slip_b": {"slip_ref"},
        "slip_use": {"use_no"}, "berth_call": {"call_no"}, "deck_plan": {"plan_no"},
    }


def test_the_key_cases_and_their_decoys() -> None:
    rows = {t.name: len(t.rows) for t in IDENTITY_TABLES}
    assert distinct_count("hull_rev", ["hull_no", "rev_no"]) == rows["hull_rev"]
    assert distinct_count("mooring", ["pier_code", "slot_no"]) == rows["mooring"]
    assert null_row_count("mooring", ["pier_code", "slot_no"]) == 3
    assert distinct_count("gate_tally", ["gate_code", "tally"]) < rows["gate_tally"]
    assert "reading_mm" not in TABLES["tide_log"].primary_key
    unique_on = [t.name for t in IDENTITY_TABLES if set(SLIP_REFERENCE) <= set(t.column_names)
                 and distinct_count(t.name, SLIP_REFERENCE) == rows[t.name]]
    assert unique_on == ["slip_a", "slip_b"]
    assert distinct_count("slip_use", SLIP_REFERENCE) < rows["slip_use"]


def test_the_entity_cases_and_their_decoys() -> None:
    rows = {t.name: len(t.rows) for t in IDENTITY_TABLES}
    for table, key, by in (("hull_rev", "hull_no", "rev_no"), ("crane_rev", "crane_no", "rev_no"),
                           ("berth_call", "vessel_no", "call_day")):
        assert distinct_count(table, [key]) < rows[table]
        assert distinct_count(table, [key, by]) == rows[table], table
    assert distinct_count("buoy_check", ["buoy_no", "check_day"]) < rows["buoy_check"]
    assert null_row_count("crane_rev", ["crane_no"]) == 2
    assert null_row_count("berth_call", ["vessel_no"]) == 1
    assert distinct_count("deck_plan", ["deck_no"]) == distinct_count(
        "deck_plan", ["deck_no", "deck_class"])
    groups, conflicting = conflict_counts("hull_rev", ["hull_no"],
                                          ["paint_band", "keel_class", "rig_type"])
    assert (groups, conflicting) == (4, {"paint_band": 2, "keel_class": 0, "rig_type": 0})
    assert conflict_counts("buoy_check", ["buoy_no"], ["lamp_state", "buoy_zone"])[1] == {
        "lamp_state": 1, "buoy_zone": 0}


def test_the_shared_entity_cases_and_their_decoys() -> None:
    for child in ("hull_survey", "hull_cargo"):
        held, inside = contained_counts(child, ["hull_no"], "hull_rev", ["hull_no"])
        assert held == inside > 0, child
    held, inside = contained_counts("hull_quote", ["hull_no"], "hull_rev", ["hull_no"])
    assert 0 < inside / held < COVERAGE_THRESHOLD
    held, inside = contained_counts("valve", ["unit_no"], "pump", ["unit_no"])
    assert held == inside
    held, inside = contained_counts("pump", ["unit_no"], "valve", ["unit_no"])
    assert inside / held < COVERAGE_THRESHOLD
    assert TABLES["pump"].primary_key == TABLES["valve"].primary_key == ("unit_no",)


def test_the_label_cases_and_their_decoy() -> None:
    assert set(column_values("rig_kind", "kind_group")) == {RIG_GROUP}
    assert len(set(column_values("rig_kind", "rig_note"))) > 1
    assert set(RIG_TITLES) == set(RIG_LIST_TEXTS)
    assert not set(RIG_TITLES.values()) & set(RIG_LIST_TEXTS.values())
    assert CODE_LIST_ROLES["rig_kinds"][1] == "rig_title"
    assert set(column_values("hull_rev", "rig_type")) <= set(RIG_TITLES)
    assert all("rig_type" not in t.column_names for t in IDENTITY_TABLES if t.name != "hull_rev")
    texts = [c for c, kind in TABLES["dock_kind"].columns[1:] if kind == "VARCHAR"]
    assert all(len(set(column_values("dock_kind", c))) > 1 for c in texts) and len(texts) == 2
    hatch_code = CODE_LIST_ROLES["hatch_kinds"][0]
    assert all(hatch_code not in t.column_names for t in IDENTITY_TABLES)
    assert set(column_values("hull_cargo", "hatch_kind")) <= set(CODE_LIST_TEXTS["hatch_kinds"])


def test_the_dictionary_states_roles_and_one_foreign_key_and_no_identity(tmp_path: Path) -> None:
    spec = load_concept_source_spec(write_identity_dictionary(tmp_path).spec)
    roles = {c.vocabulary_id: (c.roles.code, c.roles.display) for c in spec.code_lists or []}
    assert roles == CODE_LIST_ROLES
    entries = spec.entity_model.entities if spec.entity_model else None
    assert entries
    keys = [tuple(k.reference_columns) for e in entries for k in e.foreign_keys]
    assert keys == [SLIP_REFERENCE]
    assert all(e.identity is None for e in entries)


class _Proposed(BaseModel):
    entity: str
    entity_key: list[str]
    latest_by: str | None = None
    row_key: list[str] | None = None


def test_the_fake_model_proposes_for_the_relation_a_call_names() -> None:
    assert set(PROPOSALS) == set(TABLES)
    for name, proposal in PROPOSALS.items():
        assert set(proposal.entity_key) <= set(TABLES[name].column_names), name
    assert relation_named("the columns of berth_call") == "berth_call"
    assert relation_named("the columns of berth: berth_no, dock_code (see dock_kind)") == "berth"
    assert relation_named("fixident.yard.crane_rev: crane_rev_id, crane_no") == "crane_rev"
    assert relation_named("", "duckdb://x/fixident/yard/pump") == "pump"
    made = proposal_for(_Proposed, "", "duckdb://x/fixident/yard/crane_rev")
    assert made == _Proposed(entity="Crane", entity_key=["crane_no"], latest_by="rev_no")
    assert proposal_for(_Proposed, "no relation here") is None


def test_the_fake_l2_stage_names_the_shared_entity() -> None:
    from unittest.mock import MagicMock

    client = IdentityFakeClient(llm=MagicMock())
    for table, entity in L2_ENTITIES.items():
        named = client.invoke("", StageAResult, table_ref=f"duckdb://x/fixident/yard/{table}")
        assert named.primary_entity == entity
    assert L2_ENTITIES["hull_rev"] == L2_ENTITIES["hull_cargo"] == PROPOSALS["hull_rev"].entity

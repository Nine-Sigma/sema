"""Land verifier: the land result checked against the landed data and the run's policy."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import duckdb
import pytest

from sema.ingest.duckdb_staging import Staging
from sema.intake.landing import LandResult
from sema.intake.loaders.tabular import land_tabular
from tests.unit.intake.shapes.shapes_env import Source, staging_source, write_workbook
from tests.unit.intake.verify.conftest import Landed
from tests.unit.intake.verify.verify_env import (
    LAND_CHECKS,
    assert_named,
    assert_passes,
    failures,
    tools_for,
    verifier,
)

pytestmark = pytest.mark.unit

ACRIS_RELATIONS = 65
COLUMN_INFO_HEADER_ROW = 2
MAIN: list[list[object]] = [["code", "label"], ["A", "Alpha"], ["B", "Beta"], ["C", "Gamma"]]
FIGURES: list[list[object]] = [[1, 2, 3], [4, 5, 6]]
POSITIONAL = ("_col_1", "_col_2", "_col_3")


def _verify(land: LandResult, source: Source, ledger: Path, **kwargs: Any) -> Any:
    return verifier("land").verify_land(land, tools_for(source, ledger, **kwargs))


def _with_policy_class(land: LandResult, policy_class: str) -> LandResult:
    relations = tuple(
        dataclasses.replace(
            r, provenance=dataclasses.replace(r.provenance, policy_class=policy_class)
        )
        for r in land.relations
    )
    return dataclasses.replace(land, policy_class=policy_class, relations=relations)


def _replace_relation(land: LandResult, name: str, **changes: Any) -> LandResult:
    relations = tuple(
        dataclasses.replace(r, **changes) if r.name == name else r for r in land.relations
    )
    return dataclasses.replace(land, relations=relations)


@pytest.fixture
def headerless(tmp_path: Path) -> Iterator[Source]:
    path = write_workbook(tmp_path / "book.xlsx", {"Main": MAIN, "Figures": FIGURES})
    with Staging(str(tmp_path / "staging.duckdb"), schemas=()) as staging:
        land = land_tabular(path, staging, schema="land_h")
        yield staging_source(staging, land, tmp_path / "ledger" / "landing")


def _memory(tmp_path: Path, *statements: str) -> tuple[duckdb.DuckDBPyConnection, Source]:
    from tests.unit.intake.shapes.shapes_env import warehouse_source

    con = duckdb.connect(":memory:")
    con.execute("CREATE SCHEMA tl")
    for statement in statements:
        con.execute(statement)
    return con, warehouse_source(con, "tl", tmp_path / "ledger" / "landing")


def test_checks_are_the_documented_list() -> None:
    assert set(verifier("land").CHECKS) == LAND_CHECKS


def test_acris_land_passes_with_65_relations(acris: Landed, tmp_path: Path) -> None:
    land = acris.source.land
    verdict = _verify(land, acris.source, tmp_path / "ledger" / "verify")

    assert_passes(verdict, LAND_CHECKS)
    assert len(land.relations) == ACRIS_RELATIONS
    column_info = [r for r in land.relations if r.name.endswith("__Column_Info")]
    assert len(column_info) == 15
    assert {r.header_row for r in column_info} == {COLUMN_INFO_HEADER_ROW}


def test_a2_land_passes_and_lists_the_empty_named_column(a2: Landed, tmp_path: Path) -> None:
    land = a2.source.land
    verdict = _verify(land, a2.source, tmp_path / "ledger" / "verify")

    assert_passes(verdict, LAND_CHECKS)
    concepts = f"{land.relation('codes_a2__Concepts').schema}.codes_a2__Concepts"
    listed = [a for a in verdict.ambiguities if a.subject == concepts]
    assert listed and all(a.stage == "land" for a in listed)
    assert "_empty_3" in {member for a in listed for member in a.members}


def test_loader_declaring_full_under_metadata_only_fails(a2: Landed, tmp_path: Path) -> None:
    land = _with_policy_class(a2.source.land, "full")
    verdict = _verify(land, a2.source, tmp_path / "ledger" / "verify")

    assert verdict.status == "fail"
    assert failures(verdict, "policy_class_permitted")
    assert_named(verdict, LAND_CHECKS)


def test_loader_declaring_full_under_a_full_run_is_permitted(a2: Landed, tmp_path: Path) -> None:
    land = _with_policy_class(a2.source.land, "full")
    verdict = _verify(land, a2.source, tmp_path / "ledger" / "verify", tier="full")

    assert not failures(verdict, "policy_class_permitted")
    assert_named(verdict, LAND_CHECKS)


def test_headerless_relation_is_listed_with_its_positional_columns_not_failed(
    headerless: Source, tmp_path: Path
) -> None:
    land = headerless.land
    figures = f"{land.relation('book__Figures').schema}.book__Figures"
    verdict = _verify(land, headerless, tmp_path / "ledger" / "verify")

    assert land.headerless == ("book__Figures",)
    assert_passes(verdict, LAND_CHECKS)
    (listed,) = [a for a in verdict.ambiguities if a.subject == figures]
    assert listed.stage == "land"
    assert set(listed.members) == set(POSITIONAL)
    assert listed.choice is None


def test_recorded_row_count_differing_from_the_landed_relation_fails(
    a2: Landed, tmp_path: Path
) -> None:
    landed = a2.source.land.relation("codes_a2__Values")
    land = _replace_relation(a2.source.land, landed.name, row_count=landed.row_count + 1)
    verdict = _verify(land, a2.source, tmp_path / "ledger" / "verify")

    (found,) = failures(verdict, "row_counts_match")
    assert found.relation == f"{landed.schema}.{landed.name}"
    assert_named(verdict, LAND_CHECKS)


def test_all_numeric_header_row_fails(tmp_path: Path) -> None:
    con, source = _memory(
        tmp_path, "CREATE TABLE tl.numbered AS SELECT 'a' AS \"1\", 'b' AS \"2\" FROM range(3)"
    )
    land = _replace_relation(source.land, "numbered", header_row=1)
    verdict = _verify(land, source, tmp_path / "ledger" / "verify")
    con.close()

    (found,) = failures(verdict, "header_row_valid")
    assert found.relation == "tl.numbered"
    assert_named(verdict, LAND_CHECKS)


def test_header_value_recurring_in_its_column_body_fails(tmp_path: Path) -> None:
    con, source = _memory(
        tmp_path,
        "CREATE TABLE tl.echoed AS SELECT * FROM (VALUES ('A', 'Alpha'), ('code', 'Beta'),"
        " ('C', 'Gamma')) t(code, label)",
    )
    land = _replace_relation(source.land, "echoed", header_row=1)
    verdict = _verify(land, source, tmp_path / "ledger" / "verify")
    con.close()

    (found,) = failures(verdict, "header_row_valid")
    assert (found.relation, found.column) == ("tl.echoed", "code")
    assert_named(verdict, LAND_CHECKS)


def test_row_locator_missing_on_a_row_fails(tmp_path: Path) -> None:
    con, source = _memory(
        tmp_path,
        "CREATE TABLE tl.located AS SELECT * FROM (VALUES ('A', 'f!1'), ('B', ''),"
        " ('C', 'f!3')) t(code, _sema_locator)",
    )
    located = source.land.relation("located")
    row_kind = dataclasses.replace(located.provenance, locator_kind="row", locator=None)
    land = _replace_relation(source.land, "located", provenance=row_kind)
    verdict = _verify(land, source, tmp_path / "ledger" / "verify")
    con.close()

    (found,) = failures(verdict, "locator_present")
    assert found.relation == "tl.located"
    assert_named(verdict, LAND_CHECKS)


def test_relation_locator_kind_needs_no_locator_column(tmp_path: Path) -> None:
    con, source = _memory(
        tmp_path, "CREATE TABLE tl.plain AS SELECT * FROM (VALUES ('A'), ('B')) t(code)"
    )
    verdict = _verify(source.land, source, tmp_path / "ledger" / "verify")
    con.close()

    assert not failures(verdict, "locator_present")
    assert_named(verdict, LAND_CHECKS)


def test_land_verifier_reads_no_value(a2: Landed, tmp_path: Path) -> None:
    ledger = tmp_path / "ledger" / "verify"
    _verify(a2.source.land, a2.source, ledger)
    lines = Source(a2.source.land, a2.source.tools, "staging", ledger).value_bearing()

    assert lines == []

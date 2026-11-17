"""The coded-column fixture holds each case and tempts each decoy's predicate, on its data."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import duckdb
import pytest

from sema.binding.importer import import_concept_source
from sema.binding.spec import load_concept_source_spec
from sema.connectors.duckdb import DuckDBConnector
from sema.ingest.duckdb_staging import Staging
from sema.intake.land.relations import RelationProvider
from sema.models.config import DuckDBConfig
from tests.support.connector_fixture_codes import (
    CODE_LISTS,
    CODED_TABLES,
    CODES_CATALOG,
    CODES_SCHEMA,
    STATED_DOMAINS,
    UNPLACED_PROPERTY,
    column_values,
    write_coded_database,
    write_dictionary,
)

pytestmark = pytest.mark.unit

TABLES = {table.name: table for table in CODED_TABLES}


@pytest.fixture(scope="module")
def database(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_coded_database(tmp_path_factory.mktemp("codes"))


@pytest.fixture
def warehouse(database: Path) -> Iterator[duckdb.DuckDBPyConnection]:
    con = duckdb.connect(str(database), read_only=True)
    yield con
    con.close()


def _observed(table: str, column: str) -> set[object]:
    return {value for value in column_values(table, column) if value is not None}


def _key(table: str) -> set[object]:
    return _observed(table, TABLES[table].primary_key[0])


def _value_share(table: str, column: str, listed: set[object]) -> float:
    observed = _observed(table, column)
    return len(observed & listed) / len(observed)


def test_the_file_holds_every_row_the_python_spells(warehouse: duckdb.DuckDBPyConnection) -> None:
    for table in CODED_TABLES:
        stored = warehouse.execute(
            f"SELECT {', '.join(table.column_names)} "
            f"FROM {CODES_CATALOG}.{CODES_SCHEMA}.{table.name}"
        ).fetchall()
        spelled = zip(*(column_values(table.name, c) for c in table.column_names))
        assert set(stored) == set(spelled) and len(stored) == len(set(stored)), table.name


def test_declared_keys_are_read_one_column_each(database: Path) -> None:
    connector = DuckDBConnector(DuckDBConfig(path=str(database)))
    try:
        found = {
            (table.name, key.referenced_table, key.column_pairs)
            for table in CODED_TABLES
            for key in connector.extract_foreign_keys_dto(CODES_CATALOG, CODES_SCHEMA, table.name)
        }
    finally:
        connector.close()
    assert found == {
        ("panel", "finish", (("finish_code", "finish_code"),)),
        ("panel", "grit", (("grit_code", "grit_code"),)),
        ("fitting", "tone", (("tone_code", "tone_code"),)),
    }


def test_code_relations_have_unique_keys_and_their_text_columns() -> None:
    text_columns = {
        name: [c for c, kind in table.columns[1:] if kind == "VARCHAR"]
        for name, table in TABLES.items() if name not in ("panel", "fitting")
    }
    assert text_columns == {
        "finish": ["finish_label"], "tone": [], "grit": ["grit_note", "grit_alias"],
        "shift_slot": ["slot_title"], "rack": ["rack_title"], "rack_spare": ["spare_title"],
        "bay": ["bay_title"], "crew": ["crew_name"],
    }
    for name in text_columns:
        keys = column_values(name, TABLES[name].primary_key[0])
        assert None not in keys and len(set(keys)) == len(keys), name


def test_the_sibling_label_is_a_function_of_the_code() -> None:
    pairs = set(zip(column_values("fitting", "tone_code"), column_values("fitting", "tone_name")))
    assert len(pairs) == len({code for code, _ in pairs}) == len(_key("tone"))


def test_each_decoy_tempts_its_predicate() -> None:
    assert _observed("panel", "shift") <= _key("shift_slot")
    assert _observed("panel", "crew_size") <= _key("bay")
    assert _observed("panel", "rack_no") <= _key("rack") & _key("rack_spare")
    assert _observed("panel", "assigned_crew") <= _key("crew")
    assert len(_observed("panel", "assigned_crew")) == 1100
    every_key = set().union(*(_key(name) for name in TABLES))
    assert not _observed("panel", "finish_cd") & every_key
    assert not _observed("fitting", "grit_code") & _key("grit")


def test_stated_domains_cover_the_floor_from_both_sides() -> None:
    coats = set(CODE_LISTS["coat_kinds"])
    assert _value_share("panel", "coat_kind", coats) == 5 / 6
    assert _value_share("fitting", "coat_kind", coats) == 6 / 10
    assert _value_share("panel", "trim", set(CODE_LISTS["trim_kinds"])) == 1 / 6
    assert column_values("panel", "coat_kind").count(None) == 100
    assert UNPLACED_PROPERTY.replace(" ", "_") not in TABLES["panel"].column_names
    for entity, domains in STATED_DOMAINS.items():
        placed = set(domains) - {UNPLACED_PROPERTY}
        assert placed <= set(TABLES[entity].column_names)


def test_the_dictionary_imports_its_code_lists(tmp_path: Path) -> None:
    files = write_dictionary(tmp_path)
    spec = load_concept_source_spec(files.spec)
    staging = Staging(str(files.staging), schemas=())
    try:
        imported = import_concept_source(
            spec, RelationProvider(warehouse=None, staging=staging), dataset_map={}
        )
    finally:
        staging.close()
    terms = {(t["vocabulary"], t["code"], t["display"]) for t in imported.manifest["terms"]}
    assert terms == {
        (vocabulary, code, label)
        for vocabulary, codes in CODE_LISTS.items() for code, label in codes.items()
    }

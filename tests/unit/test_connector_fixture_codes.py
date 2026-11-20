"""The coded-column fixture holds each case and tempts each decoy's predicate, on its data."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from collections.abc import Set as AbstractSet
from pathlib import Path

import duckdb
import pytest

from sema.binding.importer import import_concept_source
from sema.binding.spec import load_concept_source_spec
from sema.connectors.duckdb import DuckDBConnector
from sema.engine.join_detector import COVERAGE_THRESHOLD
from sema.ingest.duckdb_staging import Staging
from sema.intake.land.relations import RelationProvider
from sema.models.config import DuckDBConfig
from tests.support.connector_fixture_codes import (
    CODE_LISTS,
    CODED_TABLES,
    CODES_CATALOG,
    CODES_SCHEMA,
    DICTIONARY_ENTRIES,
    FACT_TABLES,
    LOW_DISTINCT_BOUND,
    UNPLACED_PROPERTY,
    column_values,
    named_table,
    normalized_name,
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


def _value_share(table: str, column: str, listed: AbstractSet[object]) -> float:
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
        ("panel", "pallet", (("pallet_no", "pallet_no"),)),
        ("fitting", "tone", (("tone_code", "tone_code"),)),
        ("fitting", "sheen", (("sheen_code", "sheen_code"),)),
    }


def test_code_relations_have_unique_keys_and_their_text_columns() -> None:
    text_columns = {
        name: [c for c, kind in table.columns[1:] if kind == "VARCHAR"]
        for name, table in TABLES.items() if name not in FACT_TABLES
    }
    assert text_columns == {
        "finish": ["finish_label"], "tone": [], "sheen": [], "glaze": ["glaze_title"],
        "grit": ["grit_note", "grit_alias"], "shift_slot": ["slot_title"],
        "rack": ["rack_title"], "rack_spare": ["spare_title"], "bay": ["bay_title"],
        "crew": ["crew_name"], "pallet": ["pallet_title"],
    }
    for name in text_columns:
        keys = column_values(name, TABLES[name].primary_key[0])
        assert None not in keys and len(set(keys)) == len(keys), name


def _violation_rate(table: str, code: str, text: str) -> float:
    """Share of rows whose text differs from the most common text for their code."""
    pairs = list(zip(column_values(table, code), column_values(table, text)))
    by_code: dict[object, Counter[object]] = {}
    for code_value, text_value in pairs:
        by_code.setdefault(code_value, Counter())[text_value] += 1
    agreeing = sum(counts.most_common(1)[0][1] for counts in by_code.values())
    return 1 - agreeing / len(pairs)


def _determined_pairs() -> set[tuple[str, str, str]]:
    found = set()
    for name in FACT_TABLES:
        table = TABLES[name]
        texts = [c for c, kind in table.columns if kind == "VARCHAR"]
        for code in table.column_names[1:]:
            for text in texts:
                if text != code and _violation_rate(name, code, text) <= 1 - COVERAGE_THRESHOLD:
                    found.add((name, code, text))
    return found


def test_only_the_sibling_label_cases_hold_a_text_column_the_code_determines() -> None:
    siblings = {("fitting", "tone_code", "tone_name"), ("fitting", "tone_name", "tone_code")}
    sheen = ("sheen_code", "sheen_name", "sheen_hint")
    siblings |= {("fitting", a, b) for a in sheen for b in sheen if a != b}
    assert _determined_pairs() == siblings


def test_each_decoy_tempts_its_predicate() -> None:
    assert _observed("panel", "shift") <= _key("shift_slot")
    assert _observed("panel", "crew_size") <= _key("bay")
    assert _observed("panel", "rack_no") <= _key("rack") & _key("rack_spare")
    every_key = set().union(*(_key(name) for name in TABLES))
    assert not _observed("panel", "finish_cd") & every_key
    assert not _observed("fitting", "grit_code") & _key("grit")


def test_name_match_cases_share_their_key_column_name() -> None:
    assert _observed("fitting", "glaze_code") <= _key("glaze")
    keyed_glaze = [t for t in TABLES.values() if t.primary_key == ("glaze_code",)]
    assert [t.name for t in keyed_glaze] == ["glaze"]
    assert TABLES["rack"].primary_key == TABLES["rack_spare"].primary_key == ("rack_no",)


def test_the_bound_cases_sit_at_and_one_past_the_bound() -> None:
    assert len(_key("crew")) == len(_observed("panel", "crew_ref")) == LOW_DISTINCT_BOUND
    assert _observed("panel", "crew_ref") <= _key("crew")
    assert len(_observed("panel", "assigned_crew")) == LOW_DISTINCT_BOUND + 1
    assert len(_observed("panel", "assigned_crew") - _key("crew")) == 1
    assert len(_key("pallet")) == LOW_DISTINCT_BOUND + 1
    assert _observed("panel", "pallet_no") <= _key("pallet")
    assert len(_observed("panel", "pallet_no")) < LOW_DISTINCT_BOUND


def _placements(entry_property: str) -> set[tuple[str, str]]:
    """Every (table, column) a stated property lands on by the normalized-name rule."""
    (entry,) = (e for e in DICTIONARY_ENTRIES if entry_property in e.domains)
    named = named_table(entry)
    tables = [named] if named else list(CODED_TABLES)
    wanted = normalized_name(entry_property)
    return {(t.name, c) for t in tables for c in t.column_names if normalized_name(c) == wanted}


def test_stated_domains_are_spelled_as_a_document_spells_them() -> None:
    placed = {prop: _placements(prop) for e in DICTIONARY_ENTRIES for prop in e.domains}
    assert placed == {
        "Coat Kind": {("panel", "coat_kind")}, "TRIM": {("panel", "trim")},
        "Edge Treatment": set(), "coat-kind": {("fitting", "coat_kind")},
        "Seal / Kind": {("panel", "seal_kind"), ("fitting", "seal_kind")},
        "Bevel Kind.": {("panel", "bevel_kind")},
    }
    assert all(column not in placed for found in placed.values() for _, column in found)
    assert [e.entity for e in DICTIONARY_ENTRIES if named_table(e) is None] == [
        None, "Wall Fitting",
    ]
    assert normalized_name(UNPLACED_PROPERTY) not in {
        normalized_name(c) for t in CODED_TABLES for c in t.column_names
    }


def test_stated_domains_cover_the_floor_from_both_sides() -> None:
    coats, trims = set(CODE_LISTS["coat_kinds"]), set(CODE_LISTS["trim_kinds"])
    assert _value_share("panel", "coat_kind", coats) == 5 / 6
    assert _value_share("fitting", "coat_kind", coats) == 6 / 10
    assert _value_share("panel", "trim", trims) == 1 / 7
    assert _value_share("fitting", "trim", trims) == 1.0
    assert column_values("panel", "coat_kind").count(None) == 96
    for column in (("panel", "seal_kind"), ("fitting", "seal_kind")):
        assert _value_share(*column, set(CODE_LISTS["seal_kinds"])) == 1.0
    assert _value_share("panel", "bevel_kind", set(CODE_LISTS["bevel_kinds"])) == 1.0


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
    stated = {
        domain["property_name"]: domain["domain_id"]
        for entity in imported.manifest["entities"] for domain in entity.get("domains", [])
    }
    assert stated == {prop: vocabulary for e in DICTIONARY_ENTRIES
                      for prop, vocabulary in e.domains.items()}

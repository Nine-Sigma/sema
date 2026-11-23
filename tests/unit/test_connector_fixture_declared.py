"""The declared-join fixture's own facts, counted with DuckDB on the written files and never
through the code under test: each decoy is what it claims to be."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator, Sequence
from pathlib import Path

import duckdb
import pytest

from sema.binding.spec import ConceptSourceSpec, load_concept_source_spec
from sema.engine.join_detector_utils import normalized_name
from sema.intake.land.table_schema_utils import inline_values
from tests.support.connector_fixture_declared import (
    ACCEPTED,
    CRATE_ENTRY,
    DECLARED_BY_NAME,
    DECLARED_DICTIONARY_SCHEMA,
    DICTIONARY_ENTRIES,
    LID_LIST,
    LOST_PROPERTY,
    MISSPELLED_PROPERTY,
    OPEN,
    REJECTED,
    STRAP_LIST,
    TRAY_LIST,
    list_note,
    write_declared_database,
    write_declared_dictionary,
    write_intake_run,
)
from tests.support.connector_fixture_identity import IDENTITY_SCHEMA

pytestmark = pytest.mark.unit

Pair = tuple[object, ...]


@pytest.fixture(scope="module")
def con(tmp_path_factory: pytest.TempPathFactory) -> Iterator[duckdb.DuckDBPyConnection]:
    database = write_declared_database(tmp_path_factory.mktemp("declared"))
    with duckdb.connect(str(database), read_only=True) as opened:
        yield opened


def _rows(con: duckdb.DuckDBPyConnection, sql: str) -> list[Pair]:
    return [tuple(row) for row in con.execute(sql).fetchall()]


def _pairs(con: duckdb.DuckDBPyConnection, table: str, columns: Sequence[str]) -> set[Pair]:
    """Distinct tuples with every part non-NULL."""
    listed = ", ".join(columns)
    nulls = " AND ".join(f"{c} IS NOT NULL" for c in columns)
    return set(_rows(con, f"SELECT DISTINCT {listed} FROM {IDENTITY_SCHEMA}.{table} WHERE {nulls}"))


def _repeats(con: duckdb.DuckDBPyConnection, table: str, columns: Sequence[str]) -> int:
    """Non-NULL tuples held by more than one row."""
    listed, nulls = ", ".join(columns), " AND ".join(f"{c} IS NOT NULL" for c in columns)
    return len(_rows(con, f"SELECT {listed} FROM {IDENTITY_SCHEMA}.{table} WHERE {nulls} "
                          f"GROUP BY {listed} HAVING COUNT(*) > 1"))


def _columns(con: duckdb.DuckDBPyConnection, table: str) -> list[str]:
    return [str(name) for (name,) in _rows(
        con, "SELECT column_name FROM information_schema.columns "
             f"WHERE table_schema = '{IDENTITY_SCHEMA}' AND table_name = '{table}' "
             "ORDER BY ordinal_position")]


def _values(con: duckdb.DuckDBPyConnection, table: str, column: str) -> set[object]:
    return {v for (v,) in _rows(con, f"SELECT DISTINCT {column} FROM {IDENTITY_SCHEMA}.{table}")}


def _join_columns(child: str) -> tuple[str, ...]:
    (entry,) = [e for e in DICTIONARY_ENTRIES if e.relation == child]
    return tuple(local for local, _ in entry.pairs)


def test_no_relation_declares_a_key(con: duckdb.DuckDBPyConnection) -> None:
    assert _rows(con, "SELECT COUNT(*) FROM duckdb_constraints() "
                      "WHERE constraint_type IN ('PRIMARY KEY', 'UNIQUE', 'FOREIGN KEY')") == [(0,)]


def test_each_accepted_child_holds_only_its_own_referenced_pairs(
    con: duckdb.DuckDBPyConnection,
) -> None:
    for child, referenced in ACCEPTED.items():
        cols = _join_columns(child)
        assert _pairs(con, child, cols) <= _pairs(con, referenced, cols), child
        assert _repeats(con, referenced, cols) == 0, referenced
    assert not _pairs(con, "east_load", ("pallet_no", "pack_day")) & _pairs(
        con, "pallet_west", ("pallet_no", "pack_day"))
    assert not _pairs(con, "west_load", ("pallet_no", "pack_day")) & _pairs(
        con, "pallet_east", ("pallet_no", "pack_day"))
    assert _columns(con, "pallet_east") == _columns(con, "pallet_west")


def test_the_tie_decoy_has_the_referenced_column_set_and_none_of_the_childs_pairs(
    con: duckdb.DuckDBPyConnection,
) -> None:
    cols = ("crate_no", "seal_day")
    assert _columns(con, "crate_head_spare") == _columns(con, "crate_head")
    assert _repeats(con, "crate_head_spare", cols) == 0
    assert not _pairs(con, "crate_line", cols) & _pairs(con, "crate_head_spare", cols)


def test_the_referenced_side_has_one_null_part_row_whose_first_part_is_unique(
    con: duckdb.DuckDBPyConnection,
) -> None:
    nulls = _rows(con, f"SELECT crate_no FROM {IDENTITY_SCHEMA}.crate_head WHERE seal_day IS NULL")
    assert len(nulls) == 1
    ((crate_no,),) = nulls
    assert _rows(con, f"SELECT COUNT(*) FROM {IDENTITY_SCHEMA}.crate_head "
                      f"WHERE crate_no = {crate_no}") == [(1,)]


def test_the_rejected_referenced_side_repeats_a_pair_its_child_holds(
    con: duckdb.DuckDBPyConnection,
) -> None:
    cols = ("bale_no", "weigh_day")
    assert _repeats(con, "bale_head", cols) == 1
    assert _pairs(con, "bale_line", cols) <= _pairs(con, "bale_head", cols)


def test_both_open_candidates_are_unique_and_hold_every_pair_of_the_child(
    con: duckdb.DuckDBPyConnection,
) -> None:
    cols = _join_columns(OPEN[0])
    assert _columns(con, "drum_head") == _columns(con, "drum_head_copy")
    for referenced in ("drum_head", "drum_head_copy"):
        assert _repeats(con, referenced, cols) == 0
        assert _pairs(con, OPEN[0], cols) <= _pairs(con, referenced, cols)


def test_no_column_family_names_a_column_of_another_case() -> None:
    families = [("crate_head", "crate_head_spare", "crate_line"), ("bale_head", "bale_line"),
                ("drum_head", "drum_head_copy", "drum_line"),
                ("pallet_east", "pallet_west", "east_load", "west_load"),
                ("cask_head", "cask_line"), ("tray_stock", "tray_return")]
    owners: dict[str, int] = {}
    for index, family in enumerate(families):
        for table in family:
            for column in DECLARED_BY_NAME[table].column_names:
                assert owners.setdefault(column, index) == index, column
    assert {t for f in families for t in f} == set(DECLARED_BY_NAME)
    assert set(ACCEPTED) | set(REJECTED) | set(OPEN) == {
        e.relation for e in DICTIONARY_ENTRIES if e.joins_to}


def test_entries_list_every_property_and_the_two_misspellings_sit_either_side_of_nine() -> None:
    for entry in DICTIONARY_ENTRIES:
        columns = set(DECLARED_BY_NAME[entry.relation].column_names)
        missing = set(entry.properties) - columns
        if entry is CRATE_ENTRY:
            assert missing == {MISSPELLED_PROPERTY} and len(entry.properties) >= 9
            assert (len(columns) - 1) / (len(columns) + 1) >= 0.8
        elif LOST_PROPERTY in entry.properties:
            assert missing == {LOST_PROPERTY} and len(entry.properties) < 9
            assert (len(columns) - 1) / (len(columns) + 1) < 0.8
        else:
            assert set(entry.properties) == columns, entry.relation
        assert len(entry.properties) == len(columns)


def test_no_entity_or_header_name_is_a_table_name() -> None:
    tables = {normalized_name(t) for t in DECLARED_BY_NAME}
    names = [n for e in DICTIONARY_ENTRIES for n in (e.entity, e.header, e.joins_to) if n]
    assert names and not {normalized_name(n) for n in names} & tables


def test_referencing_entries_state_no_entity_and_a_header() -> None:
    for entry in DICTIONARY_ENTRIES:
        assert (entry.entity is None) == bool(entry.joins_to) == bool(entry.header), entry


def test_the_held_list_covers_its_column_and_the_missed_list_two_of_three(
    con: duckdb.DuckDBPyConnection,
) -> None:
    assert _values(con, "crate_head", "lid_kind") == set(LID_LIST)
    assert _values(con, "crate_head", "wrap_kind") == set(LID_LIST)
    strap = _values(con, "crate_head", "strap_kind")
    assert len(strap) == 3 and len(strap & set(STRAP_LIST)) == 2


def test_the_placement_decoy_holds_the_listed_codes_on_a_same_named_column(
    con: duckdb.DuckDBPyConnection,
) -> None:
    assert "lid_kind" in _columns(con, "crate_head_spare")
    assert _values(con, "crate_head_spare", "lid_kind") <= set(LID_LIST)


def test_the_unbound_entitys_domain_column_is_on_two_relations_within_its_list(
    con: duckdb.DuckDBPyConnection,
) -> None:
    for table in ("tray_stock", "tray_return"):
        assert _values(con, table, "tray_kind") <= set(TRAY_LIST)
        assert len(_rows(con, f"SELECT tray_kind FROM {IDENTITY_SCHEMA}.{table}")) > len(
            _values(con, table, "tray_kind"))


def test_the_list_notes_are_in_the_form_the_importer_reads_and_two_are_equal() -> None:
    assert CRATE_ENTRY.lists["lid_kind"] == CRATE_ENTRY.lists["wrap_kind"]
    for codes in CRATE_ENTRY.lists.values():
        assert len(codes) >= 2
        assert inline_values(list_note(codes)) == codes


def test_the_dictionary_staging_holds_each_entry_and_the_spec_parses(tmp_path: Path) -> None:
    files = write_declared_dictionary(tmp_path)
    spec = load_concept_source_spec(files.spec)
    with duckdb.connect(str(files.staging), read_only=True) as staged:
        tables = {t for (t,) in staged.execute(
            "SELECT table_name FROM information_schema.tables "
            f"WHERE table_schema = '{DECLARED_DICTIONARY_SCHEMA}'").fetchall()}
    assert {f"{e.relation}_fields" for e in DICTIONARY_ENTRIES} <= tables
    assert spec.entity_model is not None and spec.entity_model.entities is not None
    assert len(spec.entity_model.entities) == len(DICTIONARY_ENTRIES)


def test_write_intake_run_leaves_the_folder_intake_leaves(tmp_path: Path) -> None:
    approved = write_intake_run(tmp_path, stage="import", status="approved")
    waiting = write_intake_run(tmp_path, stage="review", status="waiting_review")

    with sqlite3.connect(tmp_path / "registry.sqlite") as registry:
        rows = dict(((r, (s, t)) for r, s, t in registry.execute(
            "SELECT run_id, stage, status FROM runs")))
    assert rows == {approved.name: ("import", "approved"),
                    waiting.name: ("review", "waiting_review")}
    for folder in (approved, waiting):
        assert folder.parent == tmp_path
        assert (folder / "staging.duckdb").is_file()
        spec = ConceptSourceSpec.model_validate_json(
            (folder / "artifacts" / "spec.json").read_text(encoding="utf-8"))
        assert spec == load_concept_source_spec(write_declared_dictionary(
            tmp_path / f"{folder.name}-yaml").spec)

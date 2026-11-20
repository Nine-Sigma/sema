"""Tabular loader: CSV, XLSX, and Parquet land as staging relations."""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path

import openpyxl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from openpyxl.styles import Font

from sema.ingest.duckdb_staging import Staging

pytestmark = pytest.mark.unit

WORKBOOK = Path(__file__).parents[4] / "fixtures" / "intake" / "tabular" / "codes_a2.xlsx"
SCHEMA = "land_a2"


@pytest.fixture
def staging(tmp_path: Path) -> Staging:
    handle = Staging(str(tmp_path / "staging.duckdb"), schemas=())
    yield handle
    handle.close()


def _land(source: Path, staging: Staging, schema: str = SCHEMA):  # type: ignore[no-untyped-def]
    from sema.intake.land.loaders.tabular import land_tabular

    return land_tabular(source, staging, schema=schema)


def _schemas(staging: Staging) -> set[str]:
    return {row[0] for row in staging.execute("SELECT schema_name FROM duckdb_schemas()").fetchall()}


def _rows(staging: Staging, relation: str, columns: str) -> list[tuple[object, ...]]:
    sql = f'SELECT {columns} FROM {SCHEMA}."{relation}" ORDER BY _sema_locator'
    rows: list[tuple[object, ...]] = staging.execute(sql).fetchall()
    return rows


def test_workbook_lands_one_relation_per_sheet_in_a_new_schema(staging: Staging) -> None:
    assert SCHEMA not in _schemas(staging)

    result = _land(WORKBOOK, staging)

    names = [r.name for r in result.relations]
    assert names == ["codes_a2__Concepts", "codes_a2__Hierarchy", "codes_a2__Values"]
    assert SCHEMA in _schemas(staging)
    assert {r.schema for r in result.relations} == {SCHEMA}
    assert result.medium == "tabular"


def test_header_rows_are_recorded(staging: Staging) -> None:
    result = _land(WORKBOOK, staging)

    headers = {r.name: r.header_row for r in result.relations}
    assert headers == {
        "codes_a2__Concepts": 3,
        "codes_a2__Values": 1,
        "codes_a2__Hierarchy": 1,
    }


def test_empty_column_lands_as_nulls_and_is_listed(staging: Staging) -> None:
    result = _land(WORKBOOK, staging)

    concepts = result.relation("codes_a2__Concepts")
    assert concepts.empty_columns == ("_empty_3",)
    assert [name for name, _ in concepts.columns][:4] == [
        "concept_code",
        "concept_name",
        "_empty_3",
        "definition",
    ]
    values = _rows(staging, "codes_a2__Concepts", "_empty_3")
    assert values and all(value is None for (value,) in values)


def test_all_integer_column_is_typed_integer(staging: Staging) -> None:
    result = _land(WORKBOOK, staging)

    types = dict(result.relation("codes_a2__Values").columns)
    assert types["rank"] == "BIGINT"
    assert types["code"] == "VARCHAR"


def test_every_row_carries_its_sheet_and_physical_row(staging: Staging) -> None:
    result = _land(WORKBOOK, staging)

    concepts = _rows(staging, "codes_a2__Concepts", "concept_code, _sema_locator")
    assert concepts[0] == ("ATTR", "Concepts!4")
    assert [loc for _, loc in concepts] == [f"Concepts!{n}" for n in range(4, 9)]
    hierarchy = _rows(staging, "codes_a2__Hierarchy", "_sema_locator")
    assert [loc for (loc,) in hierarchy] == ["Hierarchy!2", "Hierarchy!3", "Hierarchy!4"]
    assert dict(result.relation("codes_a2__Concepts").columns)["_sema_locator"] == "VARCHAR"


def test_row_counts_per_relation(staging: Staging) -> None:
    result = _land(WORKBOOK, staging)

    counts = {r.name: r.row_count for r in result.relations}
    assert counts == {
        "codes_a2__Concepts": 5,
        "codes_a2__Values": 9,
        "codes_a2__Hierarchy": 3,
    }


def test_provenance_record_per_relation(staging: Staging) -> None:
    from sema.intake.land.loaders.tabular import LOADER, LOADER_VERSION

    result = _land(WORKBOOK, staging)

    for relation in result.relations:
        record = relation.provenance
        assert record.loader == LOADER == "tabular"
        assert record.loader_version == LOADER_VERSION
        assert record.policy_class == result.policy_class == "metadata_only"
        assert record.locator_kind == "row"
        assert record.locator is None
        assert record.source_uri == WORKBOOK.resolve().as_uri()
        landed = datetime.fromisoformat(record.landed_at)
        assert landed.utcoffset() is not None and landed.utcoffset().total_seconds() == 0


def _write_csv(path: Path, text: str) -> str:
    path.write_text(text)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_directory_of_csvs_lands_each_with_sorted_identity(
    tmp_path: Path, staging: Staging
) -> None:
    folder = tmp_path / "codes"
    folder.mkdir()
    state = _write_csv(folder / "state.csv", "code,name\nNY,New York\nCA,California\n")
    country = _write_csv(folder / "country.csv", "code,name\nUS,United States\n")

    result = _land(folder, staging)

    assert [r.name for r in result.relations] == ["country", "state"]
    assert result.source_identity == (("country.csv", country), ("state.csv", state))
    assert _rows(staging, "state", "code, _sema_locator") == [
        ("NY", "state!2"),
        ("CA", "state!3"),
    ]


def test_name_collision_fails_before_writing(tmp_path: Path, staging: Staging) -> None:
    folder = tmp_path / "clash"
    folder.mkdir()
    _write_csv(folder / "codes.csv", "code,name\nA,Alpha\n")
    pq.write_table(pa.table({"code": ["B"], "name": ["Beta"]}), folder / "codes.parquet")

    with pytest.raises(ValueError, match=r"codes\.csv.*codes\.parquet"):
        _land(folder, staging)

    assert SCHEMA not in _schemas(staging)


def test_parquet_file_lands_with_its_own_schema(tmp_path: Path, staging: Staging) -> None:
    path = tmp_path / "terms.parquet"
    pq.write_table(pa.table({"code": ["A", "B"], "rank": [1, 2]}), path)

    result = _land(path, staging)

    relation = result.relation("terms")
    assert relation.header_row is None
    assert dict(relation.columns)["rank"] == "BIGINT"
    assert _rows(staging, "terms", "code, rank, _sema_locator") == [
        ("A", 1, "terms!1"),
        ("B", 2, "terms!2"),
    ]


def test_glob_lands_matching_files(tmp_path: Path, staging: Staging) -> None:
    _write_csv(tmp_path / "a_codes.csv", "code,name\nA,Alpha\n")
    _write_csv(tmp_path / "b_codes.csv", "code,name\nB,Beta\n")
    _write_csv(tmp_path / "other.csv", "code,name\nC,Gamma\n")

    result = _land(Path(str(tmp_path / "*_codes.csv")), staging)

    assert [r.name for r in result.relations] == ["a_codes", "b_codes"]


def _write_workbook(path: Path, sheets: dict[str, list[list[object]]]) -> Path:
    workbook = openpyxl.Workbook()
    workbook.remove(workbook.active)
    for title, rows in sheets.items():
        worksheet = workbook.create_sheet(title)
        for row in rows:
            worksheet.append(row)
    workbook.save(path)
    return path


def test_csv_without_a_header_lands_with_positional_columns(
    tmp_path: Path, staging: Staging
) -> None:
    path = tmp_path / "numbers.csv"
    path.write_text("1,2\n3,4\n")

    result = _land(path, staging)

    relation = result.relation("numbers")
    assert [name for name, _ in relation.columns] == ["_col_1", "_col_2", "_sema_locator"]
    assert relation.header_row is None
    assert relation.row_count == 2
    assert result.headerless == ("numbers",)
    assert _rows(staging, "numbers", "_col_1, _col_2, _sema_locator") == [
        (1, 2, "numbers!1"),
        (3, 4, "numbers!2"),
    ]


def test_headerless_sheet_lands_beside_the_other_sheets(
    tmp_path: Path, staging: Staging
) -> None:
    path = _write_workbook(
        tmp_path / "book.xlsx",
        {
            "Codes": [["code", "label"], ["A", "Alpha"], ["B", "Beta"]],
            "Figures": [[1, 2, 3], [4, 5, 6], [None, None, None], [7, 8, 9]],
        },
    )

    result = _land(path, staging)

    assert [r.name for r in result.relations] == ["book__Codes", "book__Figures"]
    assert result.headerless == ("book__Figures",)
    codes, figures = result.relation("book__Codes"), result.relation("book__Figures")
    assert (codes.header_row, codes.row_count) == (1, 2)
    assert figures.header_row is None
    assert [n for n, _ in figures.columns] == ["_col_1", "_col_2", "_col_3", "_sema_locator"]
    assert _rows(staging, "book__Figures", "_col_1, _sema_locator") == [
        (1, "Figures!1"),
        (4, "Figures!2"),
        (7, "Figures!4"),
    ]


def test_trailing_empty_columns_are_not_landed(tmp_path: Path, staging: Staging) -> None:
    path = tmp_path / "log.xlsx"
    _write_workbook(
        path, {"Log": [["Revision history"], ["Version", "Date", "Change", "Comments"]]}
    )
    workbook = openpyxl.load_workbook(path)
    for row in range(1, 6):
        workbook["Log"].cell(row=row, column=18).font = Font(bold=True)
    workbook.save(path)

    result = _land(path, staging)

    relation = result.relation("log__Log")
    assert relation.header_row == 2
    names = [name for name, _ in relation.columns]
    assert names == ["Version", "Date", "Change", "Comments", "_sema_locator"]
    assert relation.row_count == 0
    assert relation.empty_columns == ()
    assert result.headerless == ()


def test_header_line_break_is_collapsed_in_the_column_name(
    tmp_path: Path, staging: Staging
) -> None:
    path = _write_workbook(
        tmp_path / "notes.xlsx",
        {"Info": [["Column Name", "Additional Notes \n(where applicable)"], ["a", "b"]]},
    )

    result = _land(path, staging)

    names = [name for name, _ in result.relation("notes__Info").columns]
    assert names == ["Column Name", "Additional Notes (where applicable)", "_sema_locator"]


def test_a_workbook_whose_sheets_all_have_headers_lists_none_as_headerless(
    staging: Staging,
) -> None:
    assert _land(WORKBOOK, staging).headerless == ()


def test_two_landings_produce_equal_snapshots(tmp_path: Path) -> None:
    first = Staging(str(tmp_path / "one.duckdb"), schemas=())
    second = Staging(str(tmp_path / "two.duckdb"), schemas=())

    one, two = _land(WORKBOOK, first), _land(WORKBOOK, second)

    assert one.schema_snapshot == two.schema_snapshot
    assert one.source_identity == two.source_identity
    first.close()
    second.close()


def test_non_word_characters_in_names_become_underscores(
    tmp_path: Path, staging: Staging
) -> None:
    path = tmp_path / "OMOP_CDMv5.4 fields.csv"
    path.write_text("code,name\nA,Alpha\n")

    result = _land(path, staging)

    assert [r.name for r in result.relations] == ["OMOP_CDMv5_4_fields"]
    rows = staging.execute(f"SELECT code FROM {SCHEMA}.OMOP_CDMv5_4_fields").fetchall()
    assert rows == [("A",)]


def test_csv_title_block_keeps_the_physical_header_row_and_locators(
    tmp_path: Path, staging: Staging
) -> None:
    path = tmp_path / "titled.csv"
    path.write_text("My code list\n\ncode,name,parent\nA,Alpha,\nB,Beta,A\n")

    result = _land(path, staging)

    assert result.relation("titled").header_row == 3
    assert _rows(staging, "titled", "code, _sema_locator") == [
        ("A", "titled!4"),
        ("B", "titled!5"),
    ]


def test_csv_blank_lines_and_multiline_cells_keep_physical_rows(
    tmp_path: Path, staging: Staging
) -> None:
    path = tmp_path / "gaps.csv"
    path.write_text('code,label\nA,One\n\nB,"Two\nlines"\nC,Three\n')

    _land(path, staging)

    assert _rows(staging, "gaps", "code, label, _sema_locator") == [
        ("A", "One", "gaps!2"),
        ("B", "Two\nlines", "gaps!4"),
        ("C", "Three", "gaps!6"),
    ]


def test_csv_delimiter_is_sniffed(tmp_path: Path, staging: Staging) -> None:
    path = tmp_path / "semi.csv"
    path.write_text("code;label\nA;Alpha\nB;Beta\n")

    _land(path, staging)

    assert _rows(staging, "semi", "code, label") == [("A", "Alpha"), ("B", "Beta")]


def test_text_that_would_change_as_a_number_lands_as_text(
    tmp_path: Path, staging: Staging
) -> None:
    path = tmp_path / "codes.csv"
    path.write_text(
        "code,id,score\n1,12345678901234567890,0.5\n1.1,2,1.25\n1.10,3,2.0\n"
    )

    result = _land(path, staging)

    assert dict(result.relation("codes").columns) == {
        "code": "VARCHAR",
        "id": "VARCHAR",
        "score": "DOUBLE",
        "_sema_locator": "VARCHAR",
    }
    assert _rows(staging, "codes", "code, id") == [
        ("1", "12345678901234567890"),
        ("1.1", "2"),
        ("1.10", "3"),
    ]


def test_headers_differing_only_in_case_land_the_sheet_headerless(
    tmp_path: Path, staging: Staging
) -> None:
    path = tmp_path / "cased.csv"
    path.write_text("Code,code,label\nA,a,Alpha\n")

    result = _land(path, staging)

    assert result.headerless == ("cased",)
    assert result.relation("cased").header_row is None
    assert _rows(staging, "cased", "_col_1, _col_2, _col_3") == [
        ("Code", "code", "label"),
        ("A", "a", "Alpha"),
    ]


def test_header_named_like_the_locator_column_fails(
    tmp_path: Path, staging: Staging
) -> None:
    path = tmp_path / "loc.csv"
    path.write_text("code,_SEMA_LOCATOR\nA,x\n")

    with pytest.raises(ValueError, match="_sema_locator"):
        _land(path, staging)

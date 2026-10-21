"""Tabular loader: a sniffed CSV delimiter is kept only when it splits a row."""

from __future__ import annotations

import csv
from pathlib import Path

import duckdb
import pytest

from sema.ingest.duckdb_staging import Staging

pytestmark = pytest.mark.unit

SCHEMA = "land_ragged"
FIXTURES = Path(__file__).parents[3] / "fixtures" / "intake"

RAGGED_COMMA = (
    'H,"batch one",2024,v1,"opened, late"\n'
    'D,100,"Widget, large",each,12.5,"first note",Y\n'
    'D,101,"Gadget small",each,3.25,"second, note",N\n'
    'T,2,"closed on time",v1,end\n'
    'D,102,"Bolt, hex",box,0.75,"third note",Y\n'
)
RAGGED_COMMA_WITH_HEADER = (
    "kind,code,label,unit,price,note,flag\n"
    'D,100,"Widget, large",each,12.5,"first note",Y\n'
    'T,2,"closed on time",v1,end\n'
    'D,101,"Gadget small",each,3.25,"second, note",N\n'
    'T,3,"closed late",v2,end\n'
)
RAGGED_SEMICOLON = (
    'H;"batch one";2024;v1;"opened; late"\n'
    'D;100;"Widget; large";each;12.5;"first note";Y\n'
    'D;101;"Gadget small";each;3.25;"second; note";N\n'
    'T;2;"closed on time";v1;end\n'
)
RAGGED_TAB = (
    'H\t"batch one"\t2024\tv1\t"opened, late"\n'
    'D\t100\t"Widget, large"\teach\t12.5\t"first note"\tY\n'
    'D\t101\t"Gadget small"\teach\t3.25\t"second, note"\tN\n'
    'T\t2\t"closed on time"\tv1\tend\n'
)
POSITIONAL = [f"_col_{n}" for n in range(1, 8)]
FIRST_TWO_ROWS = [
    ("H", "batch one", "2024", "v1", "opened, late", None, None),
    ("D", "100", "Widget, large", "each", "12.5", "first note", "Y"),
]


@pytest.fixture
def staging(tmp_path: Path) -> Staging:
    handle = Staging(str(tmp_path / "staging.duckdb"), schemas=())
    yield handle
    handle.close()


def _land(source: Path, staging: Staging):  # type: ignore[no-untyped-def]
    from sema.intake.loaders.tabular import land_tabular

    return land_tabular(source, staging, schema=SCHEMA)


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text)
    return path


def _rows(staging: Staging, relation: str, columns: list[str]) -> list[tuple[object, ...]]:
    selected = ", ".join(f'"{name}"' for name in columns)
    sql = f'SELECT {selected} FROM {SCHEMA}."{relation}" ORDER BY _sema_locator'
    rows: list[tuple[object, ...]] = staging.execute(sql).fetchall()
    return rows


def _sniffed_delimiter(path: Path) -> str:
    with duckdb.connect(":memory:") as con:
        sniffed = con.execute("SELECT Delimiter FROM sniff_csv(?)", [str(path)]).fetchone()
    assert sniffed is not None
    return str(sniffed[0])


def _splits_some_row(path: Path, delimiter: str) -> bool:
    with path.open(newline="") as handle:
        return any(len(record) > 1 for record in csv.reader(handle, delimiter=delimiter))


def _column_names(result, relation: str) -> list[str]:  # type: ignore[no-untyped-def]
    return [name for name, _ in result.relation(relation).columns]


def test_precondition_the_sniffer_picks_a_delimiter_that_splits_no_row(
    tmp_path: Path,
) -> None:
    for name, text in [
        ("ragged.csv", RAGGED_COMMA),
        ("ragged_header.csv", RAGGED_COMMA_WITH_HEADER),
        ("ragged_semi.csv", RAGGED_SEMICOLON),
        ("ragged_tab.csv", RAGGED_TAB),
    ]:
        path = _write(tmp_path, name, text)
        assert not _splits_some_row(path, _sniffed_delimiter(path)), name


def test_ragged_comma_file_lands_comma_split_with_positional_columns(
    tmp_path: Path, staging: Staging
) -> None:
    path = _write(tmp_path, "ragged.csv", RAGGED_COMMA)

    result = _land(path, staging)

    assert _column_names(result, "ragged") == [*POSITIONAL, "_sema_locator"]
    assert result.relation("ragged").header_row is None
    assert result.relation("ragged").row_count == 5
    assert result.headerless == ("ragged",)
    assert _rows(staging, "ragged", POSITIONAL)[:2] == FIRST_TWO_ROWS
    assert _rows(staging, "ragged", ["_col_1", "_sema_locator"])[3] == ("T", "ragged!4")


def test_ragged_comma_file_with_a_header_row_keeps_its_header(
    tmp_path: Path, staging: Staging
) -> None:
    path = _write(tmp_path, "ragged_header.csv", RAGGED_COMMA_WITH_HEADER)

    result = _land(path, staging)

    names = ["kind", "code", "label", "unit", "price", "note", "flag"]
    assert _column_names(result, "ragged_header") == [*names, "_sema_locator"]
    assert result.relation("ragged_header").header_row == 1
    assert result.headerless == ()
    assert _rows(staging, "ragged_header", ["kind", "label", "note", "_sema_locator"]) == [
        ("D", "Widget, large", "first note", "ragged_header!2"),
        ("T", "closed on time", None, "ragged_header!3"),
        ("D", "Gadget small", "second, note", "ragged_header!4"),
        ("T", "closed late", None, "ragged_header!5"),
    ]


@pytest.mark.parametrize(
    ("name", "text", "quoted_cell"),
    [
        ("ragged_semi", RAGGED_SEMICOLON, "opened; late"),
        ("ragged_tab", RAGGED_TAB, "opened, late"),
    ],
    ids=["semicolon", "tab"],
)
def test_ragged_file_with_another_delimiter_lands_split_on_that_delimiter(
    tmp_path: Path, staging: Staging, name: str, text: str, quoted_cell: str
) -> None:
    path = _write(tmp_path, f"{name}.csv", text)

    result = _land(path, staging)

    assert _column_names(result, name) == [*POSITIONAL, "_sema_locator"]
    assert result.headerless == (name,)
    first = _rows(staging, name, POSITIONAL)[0]
    assert first == ("H", "batch one", "2024", "v1", quoted_cell, None, None)


def test_a_sniffed_delimiter_that_splits_no_row_is_rejected(tmp_path: Path) -> None:
    from sema.intake.loaders.tabular import _csv_dialect

    path = _write(tmp_path, "ragged.csv", RAGGED_COMMA)
    sniffed = _sniffed_delimiter(path)

    delimiter, _ = _csv_dialect(path)

    assert delimiter != sniffed
    assert delimiter == ","
    assert _splits_some_row(path, delimiter)


def test_a_sniffed_delimiter_that_splits_rows_is_kept(
    tmp_path: Path, staging: Staging
) -> None:
    from sema.intake.loaders.tabular import _csv_dialect

    path = _write(tmp_path, "piped.csv", 'code|label\nA|"Alpha, one"\nB|Beta\n')

    assert _csv_dialect(path)[0] == "|"
    _land(path, staging)
    assert _rows(staging, "piped", ["code", "label"]) == [("A", "Alpha, one"), ("B", "Beta")]


def test_a_single_column_file_still_lands_one_column(
    tmp_path: Path, staging: Staging
) -> None:
    path = _write(tmp_path, "codes.csv", "code\nA\nB\nC\n")

    result = _land(path, staging)

    assert _column_names(result, "codes") == ["code", "_sema_locator"]
    assert _rows(staging, "codes", ["code"]) == [("A",), ("B",), ("C",)]


@pytest.mark.parametrize(
    ("relative", "width", "header_row", "row_count"),
    [
        ("tabular/ambiguity_a8.csv", 4, 1, 40),
        ("omop/OMOP_CDMv5.4_Field_Level.csv", 13, 1, 432),
        ("acris/Country_Codes.csv", 3, 1, 250),
        ("acris/Document_Control_Codes.csv", 7, 1, 126),
        ("acris/Property_Types_Codes.csv", 3, 1, 49),
        ("acris/State_Codes.csv", 3, 1, 54),
        ("acris/UCC_Collateral_Codes.csv", 3, 1, 8),
    ],
)
def test_existing_csv_goldens_land_unchanged(
    staging: Staging, relative: str, width: int, header_row: int, row_count: int
) -> None:
    result = _land(FIXTURES / relative, staging)

    (relation,) = result.relations
    assert len(relation.columns) - 1 == width
    assert relation.header_row == header_row
    assert relation.row_count == row_count
    assert result.headerless == ()

"""Tabular loader: a sniffed CSV delimiter is kept only when it splits a row.

A CSV is decoded by the encoding its bytes declare or admit, not assumed UTF-8.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path

import duckdb
import pytest

from sema.ingest.duckdb_staging import Staging
from sema.intake.land.loaders import tabular_utils

pytestmark = pytest.mark.unit

SCHEMA = "land_ragged"
FIXTURES = Path(__file__).parents[4] / "fixtures" / "intake"

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
    from sema.intake.land.loaders.tabular import land_tabular

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
    from sema.intake.land.loaders.tabular import _csv_dialect

    path = _write(tmp_path, "ragged.csv", RAGGED_COMMA)
    sniffed = _sniffed_delimiter(path)

    delimiter, _ = _csv_dialect(path)

    assert delimiter != sniffed
    assert delimiter == ","
    assert _splits_some_row(path, delimiter)


def test_a_sniffed_delimiter_that_splits_rows_is_kept(
    tmp_path: Path, staging: Staging
) -> None:
    from sema.intake.land.loaders.tabular import _csv_dialect

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


def _one_column_text(separator: str, *, header: bool) -> str:
    values = ["alpha", f"beta{separator} two", "gamma"]
    values += ["delta", f"epsilon{separator}five", "zeta"]
    return ("label\n" if header else "") + "\n".join(values) + "\n"


@pytest.mark.parametrize("header", [True, False], ids=["header", "headerless"])
@pytest.mark.parametrize("separator", [";", "|", "\t"], ids=["semicolon", "pipe", "tab"])
def test_a_one_column_file_whose_minority_of_values_hold_a_delimiter_lands_one_column(
    tmp_path: Path, staging: Staging, separator: str, header: bool
) -> None:
    from sema.intake.land.loaders.tabular import _csv_dialect

    path = _write(tmp_path, "labels.csv", _one_column_text(separator, header=header))

    assert _csv_dialect(path)[0] != separator
    result = _land(path, staging)

    (column, _) = _column_names(result, "labels")
    assert column == "label" or not header
    assert (f"beta{separator} two",) in _rows(staging, "labels", [column])


CRLF = "\r\n"
UNDEFINED_IN_CP1252 = chr(0x81)
WINDOWS_ROWS = [
    ("code", "city", "price"),
    ("Q1", "Québec", "€ 12"),
    ("M2", "Montréal, centre", "€ 7"),
    ("T3", "Trois-Rivières", "€ 3"),
]
LATIN_ROWS = [
    ("code", "label"),
    ("A1", f"mark{UNDEFINED_IN_CP1252}one"),
    ("B2", "café, noir"),
    ("C3", "crème"),
]


def _csv_text(rows: Sequence[tuple[str, ...]], newline: str = "\n") -> str:
    def cell(value: str) -> str:
        return f'"{value}"' if "," in value else value

    return "".join(",".join(cell(value) for value in row) + newline for row in rows)


def _write_bytes(tmp_path: Path, name: str, data: bytes) -> Path:
    path = tmp_path / name
    path.write_bytes(data)
    return path


@pytest.mark.parametrize(
    ("data", "codec"),
    [
        (b"\xef\xbb\xbfcode,label\n", "utf-8-sig"),
        ("code,label\n".encode("utf-16"), "utf-16"),
        ("année,prix\n".encode("utf-8"), "utf-8-sig"),
        ("price\n€ 12\n".encode("cp1252"), "cp1252"),
        (f"label\nmark{UNDEFINED_IN_CP1252}one\n".encode("latin-1"), "latin-1"),
    ],
    ids=["utf8-bom", "utf16-bom", "utf8-plain", "cp1252", "latin1-fallback"],
)
def test_detect_text_encoding_follows_the_ladder(
    tmp_path: Path, data: bytes, codec: str
) -> None:
    path = _write_bytes(tmp_path, "sample.csv", data)

    assert tabular_utils.detect_text_encoding(path) == codec


def _assert_lands_decoded(
    path: Path, staging: Staging, rows: Sequence[tuple[str, ...]]
) -> None:
    header, *data = rows
    relation = path.stem

    result = _land(path, staging)

    assert _column_names(result, relation) == [*header, "_sema_locator"]
    assert result.relation(relation).header_row == 1
    assert _rows(staging, relation, list(header)) == data


def test_windows_1252_csv_lands_decoded(tmp_path: Path, staging: Staging) -> None:
    data = _csv_text(WINDOWS_ROWS, CRLF).encode("cp1252")
    assert b"\x80" in data and not data.startswith(b"\xef\xbb\xbf")
    path = _write_bytes(tmp_path, "windows.csv", data)

    _assert_lands_decoded(path, staging, WINDOWS_ROWS)


def test_latin_1_csv_lands_through_the_fallback(tmp_path: Path, staging: Staging) -> None:
    data = _csv_text(LATIN_ROWS, CRLF).encode("latin-1")
    assert b"\x81" in data
    path = _write_bytes(tmp_path, "latin.csv", data)

    _assert_lands_decoded(path, staging, LATIN_ROWS)


@pytest.mark.parametrize("codec", ["utf-8-sig", "utf-8"], ids=["bom", "plain"])
def test_utf8_csvs_land_unchanged(tmp_path: Path, staging: Staging, codec: str) -> None:
    rows = [("année", "label"), ("A1", "Québec"), ("B2", "€ 12, net"), ("C3", "Zürich")]
    path = _write_bytes(tmp_path, "unicode.csv", _csv_text(rows).encode(codec))

    _assert_lands_decoded(path, staging, rows)

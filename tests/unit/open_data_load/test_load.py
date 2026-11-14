from __future__ import annotations

import datetime
import hashlib
import json
from decimal import Decimal
from pathlib import Path

import duckdb
import pytest

from showcase.open_data_load.load import main
from tests.unit.open_data_load.conftest import ORDERS_CSV, Source, neutral_profile

pytestmark = pytest.mark.unit


def _rows(db: Path, sql: str) -> list[tuple[object, ...]]:
    with duckdb.connect(str(db), read_only=True) as conn:
        return conn.execute(sql).fetchall()


def _manifest(source: Source) -> dict[str, dict[str, object]]:
    loaded: dict[str, dict[str, object]] = json.loads(
        source.db.with_suffix(".manifest.json").read_text()
    )
    return loaded


def test_load_lands_every_relation_under_an_unqualified_name_with_schema_types(
    source: Source,
) -> None:
    assert main(source.args("load")) == 0
    rows = _rows(source.db, "SELECT * FROM orders ORDER BY order_no")
    assert rows == [
        ("A1", datetime.date(2020, 1, 2), Decimal("10.50"), 3, "007"),
        ("A2", datetime.date(2021, 12, 31), None, 4, "010"),
        ("A3", None, Decimal("2.25"), None, None),
    ]
    types = _rows(
        source.db,
        "SELECT column_name, data_type FROM duckdb_columns() "
        "WHERE table_name = 'orders' ORDER BY column_index",
    )
    assert types == [
        ("order_no", "VARCHAR"), ("ship_date", "DATE"), ("amount", "DECIMAL(18,2)"),
        ("qty", "INTEGER"), ("region", "VARCHAR"),
    ]  # fmt: skip
    assert _rows(source.db, "SELECT * FROM colors ORDER BY 1") == [("G", "Green"), ("R", "Red")]


def test_the_manifest_records_publisher_count_and_file_digest(source: Source) -> None:
    assert main(source.args("load")) == 0
    manifest = _manifest(source)
    downloaded = source.db.parent / "downloads" / source.db.stem / "orders.csv"
    assert downloaded.read_text() == ORDERS_CSV
    assert manifest["orders"]["publisher_count"] == 3
    assert manifest["orders"]["sha256"] == hashlib.sha256(ORDERS_CSV.encode()).hexdigest()
    assert manifest["colors"]["publisher_count"] == 2


@pytest.mark.parametrize(
    ("value", "column"),
    [("x1", "qty"), ("2.5", "qty"), ("1.234", "amount"), ("2020-01-02", "ship_date")],
)
def test_a_value_that_does_not_cast_losslessly_stops_only_its_relation(
    source: Source, capsys: pytest.CaptureFixture[str], value: str, column: str
) -> None:
    cells = {"qty": "1", "amount": "1.00", "ship_date": "01.01.2020"} | {column: value}
    bad = (
        "ORDER NO,SHIP DATE,AMOUNT,QTY,Unused Flag,Region Code\n"
        + "".join(f"B{i},{cells['ship_date']},{cells['amount']},{cells['qty']},,\n"
                  for i in range(4))
    )  # fmt: skip
    source.publish("ord-0001", bad, 4)
    assert main(source.args("load")) == 1
    err = capsys.readouterr().err
    assert f"orders: column {column}" in err
    assert "4 values" in err and err.count(repr(value)) == 3
    tables = _rows(source.db, "SELECT table_name FROM duckdb_tables()")
    assert tables == [("colors",)]


def test_a_header_that_maps_to_no_column_stops_its_relation(
    source: Source, capsys: pytest.CaptureFixture[str]
) -> None:
    extra = (
        "ORDER NO,SHIP DATE,AMOUNT,QTY,Unused Flag,Region Code,Extra\n"
        "A1,02.01.2020,10.50,3,,007,x\n"
    )
    source.publish("ord-0001", extra, 1)
    assert main(source.args("load")) == 1
    assert "orders: header 'Extra' maps to no column" in capsys.readouterr().err


def test_a_column_with_no_header_stops_its_relation(
    source: Source, capsys: pytest.CaptureFixture[str]
) -> None:
    source.publish("ord-0001", ORDERS_CSV.replace("QTY", "COUNT", 1), 3)
    assert main(source.args("load")) == 1
    err = capsys.readouterr().err
    assert "orders: column qty has no header 'QTY'" in err
    assert "orders: header 'COUNT' maps to no column" in err


def test_an_ignored_header_that_holds_a_value_stops_its_relation(
    source: Source, capsys: pytest.CaptureFixture[str]
) -> None:
    source.publish("ord-0001", ORDERS_CSV.replace("3,,007", "3,Y,007"), 3)
    assert main(source.args("load")) == 1
    assert "orders: column Unused Flag (ignored): 1 values" in capsys.readouterr().err


def test_a_row_count_other_than_the_publishers_fails_the_load(
    source: Source, capsys: pytest.CaptureFixture[str]
) -> None:
    source.publish("ord-0001", ORDERS_CSV, 4)
    assert main(source.args("load")) == 1
    assert "orders: loaded 3 rows, publisher count 4" in capsys.readouterr().err


def test_verify_reports_each_relation_and_the_total(
    source: Source, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(source.args("load")) == 0
    capsys.readouterr()
    assert main(source.args("verify")) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines == ["OK orders rows=3", "OK colors rows=2", "2/2 relations match"]


def test_verify_fails_a_relation_whose_count_differs_from_the_manifest(
    source: Source, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(source.args("load")) == 0
    with duckdb.connect(str(source.db)) as conn:
        conn.execute("DELETE FROM orders WHERE order_no = 'A3'")
    capsys.readouterr()
    assert main(source.args("verify")) == 1
    out = capsys.readouterr().out
    assert "FAIL orders: 2 rows, expected 3" in out and out.endswith("1/2 relations match\n")


def test_verify_fails_a_relation_whose_type_differs_from_the_schema(
    source: Source, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(source.args("load")) == 0
    with duckdb.connect(str(source.db)) as conn:
        conn.execute("ALTER TABLE colors ALTER description TYPE VARCHAR[] USING [description]")
    capsys.readouterr()
    assert main(source.args("verify")) == 1
    assert "FAIL colors: columns" in capsys.readouterr().out


def test_a_second_load_skips_a_relation_that_already_verifies(source: Source) -> None:
    assert main(source.args("load")) == 0
    (source.published / "ord-0001.csv").unlink()
    (source.published / "ord-0001.count.json").unlink()
    assert main(source.args("load")) == 0
    assert _rows(source.db, "SELECT count(*) FROM orders") == [(3,)]


def test_a_profile_whose_relations_differ_from_the_schema_is_refused(
    source: Source, capsys: pytest.CaptureFixture[str]
) -> None:
    profile = neutral_profile(source.published)
    relations = profile["relations"]
    assert isinstance(relations, dict)
    relations["extra"] = {"file": "local/colors.csv"}
    source.write_profile(profile)
    assert main(source.args("load")) == 1
    assert "profile relations differ from schema: extra" in capsys.readouterr().err

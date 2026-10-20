"""Tabular loader: CSV, XLSX, and Parquet land as staging relations."""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from sema.ingest.duckdb_staging import Staging

pytestmark = pytest.mark.unit

WORKBOOK = Path(__file__).parents[2] / "fixtures" / "intake" / "tabular" / "codes_a2.xlsx"
SCHEMA = "land_a2"


@pytest.fixture
def staging(tmp_path: Path) -> Staging:
    handle = Staging(str(tmp_path / "staging.duckdb"), schemas=())
    yield handle
    handle.close()


def _land(source: Path, staging: Staging, schema: str = SCHEMA):  # type: ignore[no-untyped-def]
    from sema.intake.loaders.tabular import land_tabular

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
    from sema.intake.loaders.tabular import LOADER, LOADER_VERSION

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


def test_sheet_without_a_header_fails_naming_the_sheet(
    tmp_path: Path, staging: Staging
) -> None:
    path = tmp_path / "numbers.csv"
    path.write_text("1,2\n3,4\n")

    with pytest.raises(ValueError, match=r"numbers.*non-empty strings.*unique.*recurs"):
        _land(path, staging)


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

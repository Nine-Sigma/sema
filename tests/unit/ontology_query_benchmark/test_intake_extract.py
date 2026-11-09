"""The workload projection reads only declared rows, completely, into a landable DuckDB file."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import pytest

from sema.ingest.duckdb_staging import Staging
from sema.intake.loaders.detect import land_source
from showcase.ontology_query_benchmark.intake_extract import extract
from showcase.ontology_query_benchmark.profile import Extract, load_profile
from tests.unit.ontology_query_benchmark.conftest import PROFILE_PATH

pytestmark = pytest.mark.unit

SPEC: Extract = load_profile(PROFILE_PATH).extract
DE = SPEC.elements.relation
CONCEPT = SPEC.members.relation
ALTERED = datetime(2026, 11, 9, 6, 27, 4, tzinfo=UTC)
LONG_VARCHAR = "varchar(2147483647)"
COLUMNS = [
    (CONCEPT, "concept_id", "string"),
    (CONCEPT, "concept_type_label", LONG_VARCHAR),
    (CONCEPT, "synonym_name", "array<string>"),
    (DE, "element_row_id", "string"),
    (DE, "element_id", "string"),
    (DE, "element_sensitive", "boolean"),
    (DE, "element_modified_on", "timestamp"),
    (DE, "member_id", "string"),
]
DE_ROWS = [
    ("s1", "L1", False, ALTERED, "L10"),
    ("s2", "L1", False, ALTERED, "L11"),
    ("s3", "L2", True, ALTERED, None),
]
CONCEPT_ROWS = [
    ("L1", "Data Element", ["one"]),
    ("L10", "Permissible Value", []),
    ("L11", "Permissible Value", ["eleven", "xi"]),
    ("L3", "Permissible Value", None),
]


class FakeCursor:
    """Answers the projection's catalog, count and select statements from fixed rows."""

    def __init__(self, *, counts: dict[str, int] | None = None, altered_after: datetime = ALTERED,
                 de_rows: list[tuple[Any, ...]] = DE_ROWS) -> None:  # fmt: skip
        self.de_rows = de_rows
        self.counts = counts or {DE: len(de_rows), CONCEPT: len(CONCEPT_ROWS)}
        self.altered = [ALTERED, altered_after]
        self.statements: list[str] = []
        self.closed = False
        self._rows: list[Any] = []

    def execute(self, sql: str, parameters: Any = None) -> None:
        self.statements.append(sql)
        self._rows = self._answer(sql)

    def fetchall(self) -> list[Any]:
        return self._rows

    def close(self) -> None:
        self.closed = True

    def _answer(self, sql: str) -> list[Any]:
        if "information_schema.columns" in sql:
            return list(COLUMNS)
        if "information_schema.tables" in sql:
            when = self.altered.pop(0)
            return [(CONCEPT, when, "concepts"), (DE, when, "data elements")]
        relation = CONCEPT if sql.split(" WHERE ")[0].endswith(CONCEPT) else DE
        if "COUNT(*)" in sql:
            return [(self.counts[relation],)]
        rows = self.de_rows if relation == DE else CONCEPT_ROWS
        return [tuple(_as_numpy_array(v) for v in row) for row in rows]


def _as_numpy_array(value: Any) -> Any:
    import numpy as np

    return np.array(value) if isinstance(value, list) else value


def _field(column: str, de_id: str | None = None) -> dict[str, Any]:
    field = {
        "column_id": f"{column}-id", "column_name": column,
        "physical_type": "STRING", "strategy": "scalar",
    }
    return field if de_id is None else {**field, "de_id": de_id}


def _write_spec(path: Path, fields: list[dict[str, Any]]) -> Path:
    path.write_text(json.dumps({"product_id": path.stem, "fields": fields}), encoding="utf-8")
    return path


def _specs(tmp_path: Path) -> list[Path]:
    return [
        _write_spec(
            tmp_path / f"{name}_spec.json",
            [_field("rid")] + [_field(f"c_{i}", i) for i in ids + ["L1"]],
        )
        for name, ids in (("people", ["L1", "L3"]), ("events", ["L2", "L4"]))
    ]


def _extract(tmp_path: Path, cursor: FakeCursor, **kwargs: Any) -> dict[str, Any]:
    return extract(lambda: cursor, SPEC, _specs(tmp_path), tmp_path / "out", **kwargs)


def test_complete_projection_lands_both_relations_under_source_names(tmp_path: Path) -> None:
    receipt = _extract(tmp_path, FakeCursor())
    duckdb_path = tmp_path / "out" / receipt["duckdb"]["path"]

    with Staging(str(tmp_path / "staging.duckdb"), schemas=()) as staging:
        land = land_source(duckdb_path, staging, schema="ignored")

    assert receipt["complete"] is True
    assert land.medium == "database"
    landed = {r.name: (r.schema, r.row_count, dict(r.columns)) for r in land.relations}
    assert landed[DE][:2] == (SPEC.schema_name, 3)
    assert landed[CONCEPT][:2] == (SPEC.schema_name, 4)
    assert landed[DE][2]["element_sensitive"] == "BOOLEAN"
    assert landed[DE][2]["element_modified_on"] == "TIMESTAMP WITH TIME ZONE"
    assert landed[CONCEPT][2] == {
        "concept_id": "VARCHAR", "concept_type_label": "VARCHAR", "synonym_name": "VARCHAR[]",
    }


def test_rows_keep_values_and_warehouse_order(tmp_path: Path) -> None:
    receipt = _extract(tmp_path, FakeCursor())

    with duckdb.connect(str(tmp_path / "out" / receipt["duckdb"]["path"]), read_only=True) as con:
        concepts = con.execute(
            f"SELECT concept_id, synonym_name FROM {SPEC.schema_name}.{CONCEPT}"
        ).fetchall()
        comment = con.execute(
            "SELECT comment FROM duckdb_tables() WHERE table_name = ?", [DE]
        ).fetchone()

    assert concepts == [
        ("L1", ["one"]), ("L10", []), ("L11", ["eleven", "xi"]), ("L3", None),
    ]
    assert comment == ("data elements",)


def test_receipt_states_scope_assistance_and_identity(tmp_path: Path) -> None:
    cursor = FakeCursor()
    receipt = _extract(tmp_path, cursor)
    on_disk = json.loads((tmp_path / "out" / "extract_receipt.json").read_text(encoding="utf-8"))
    import hashlib

    digest = hashlib.sha256((tmp_path / "out" / receipt["duckdb"]["path"]).read_bytes()).hexdigest()

    assert on_disk == receipt
    assert receipt["assistance"]["kind"] == "supplied"
    assert receipt["workload"]["de_ids_requested"] == ["L1", "L2", "L3", "L4"]
    assert receipt["workload"]["found_as_data_element"] == ["L1", "L2"]
    assert receipt["workload"]["found_as_concept_only"] == ["L3"]
    assert receipt["workload"]["missing"] == ["L4"]
    assert receipt["duckdb"]["sha256"] == digest
    assert receipt["relations"][DE]["updated_at"] == "2026-11-09T06:27:04+00:00"
    assert receipt["relations"][DE]["expected_rows"] == receipt["relations"][DE]["returned_rows"]
    assert [s["sql"] for s in receipt["statements"]] == cursor.statements
    assert cursor.closed


def test_every_statement_is_a_bounded_ordered_read(tmp_path: Path) -> None:
    cursor = FakeCursor()
    _extract(tmp_path, cursor, row_limit=50)

    selects = [s for s in cursor.statements if "information_schema" not in s and "COUNT" not in s]

    assert all(s.startswith("SELECT ") for s in cursor.statements)
    assert len(selects) == 2
    assert all(" ORDER BY " in s and s.endswith(" LIMIT 51") for s in selects)
    assert all("'L4'" in s for s in selects)


@pytest.mark.parametrize(
    ("cursor", "reason"),
    [
        (FakeCursor(counts={DE: 3, CONCEPT: 4}), "row_limit"),
        (FakeCursor(counts={DE: 3, CONCEPT: 5}), "returned_rows"),
        (FakeCursor(altered_after=datetime(2026, 11, 10, tzinfo=UTC)), "updated_at"),
    ],
    ids=["row-limit-hit", "count-moved", "source-refreshed"],
)
def test_an_incomplete_read_writes_no_database(
    tmp_path: Path, cursor: FakeCursor, reason: str
) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / "workload_projection.duckdb").write_bytes(b"stale")
    row_limit = 3 if reason == "row_limit" else 100

    receipt = _extract(tmp_path, cursor, row_limit=row_limit)

    assert receipt["complete"] is False
    assert receipt["duckdb"] is None
    assert any(reason in finding for finding in receipt["incomplete"])
    assert not (out / "workload_projection.duckdb").exists()


def test_a_malformed_declared_id_is_refused_before_any_statement(tmp_path: Path) -> None:
    spec = _write_spec(tmp_path / "spec.json", [_field("c", "L1' OR '1'='1")])
    cursor = FakeCursor()

    with pytest.raises(ValueError, match="de_id"):
        extract(lambda: cursor, SPEC, [spec], tmp_path / "out")
    assert cursor.statements == []


def test_receipt_names_each_source_with_its_catalog(tmp_path: Path) -> None:
    receipt = _extract(tmp_path, FakeCursor())

    sources = {name: entry["source"] for name, entry in receipt["relations"].items()}
    assert sources == {name: f"{SPEC.catalog}.{SPEC.schema_name}.{name}" for name in (DE, CONCEPT)}


def test_receipt_points_at_the_database_relative_to_itself(tmp_path: Path) -> None:
    receipt = _extract(tmp_path, FakeCursor())

    assert receipt["duckdb"]["path"] == "workload_projection.duckdb"
    assert (tmp_path / "out" / receipt["duckdb"]["path"]).is_file()


def test_duplicate_order_keys_make_the_read_incomplete(tmp_path: Path) -> None:
    rows = [*DE_ROWS, DE_ROWS[0]]

    receipt = _extract(tmp_path, FakeCursor(de_rows=rows))

    assert receipt["complete"] is False and receipt["duckdb"] is None
    assert any(f.startswith(f"{DE}: 1 duplicate order key") for f in receipt["incomplete"])


def test_no_declared_id_is_refused_before_any_statement(tmp_path: Path) -> None:
    spec = _write_spec(tmp_path / "spec.json", [_field("rid")])
    cursor = FakeCursor()

    with pytest.raises(ValueError, match="no de_id"):
        extract(lambda: cursor, SPEC, [spec], tmp_path / "out")
    assert cursor.statements == []


def test_a_spec_that_breaks_the_declared_format_is_refused_before_any_statement(
    tmp_path: Path,
) -> None:
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"fields": [{"de_id": "L1"}]}), encoding="utf-8")
    cursor = FakeCursor()

    with pytest.raises(ValueError, match="declared format"):
        extract(lambda: cursor, SPEC, [spec], tmp_path / "out")
    assert cursor.statements == []

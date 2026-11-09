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

pytestmark = pytest.mark.unit

ROOT = Path(__file__).parents[3]
README = ROOT / "showcase" / "partner_timbr_benchmark" / "config" / "intake_README.md"
DE = "standard_library_data_element"
CONCEPT = "standard_library_concept"
ALTERED = datetime(2026, 11, 9, 6, 27, 4, tzinfo=UTC)
LONG_VARCHAR = "varchar(2147483647)"
COLUMNS = [
    (CONCEPT, "concept_prt_id", "string"),
    (CONCEPT, "concept_type_label", LONG_VARCHAR),
    (CONCEPT, "synonym_name", "array<string>"),
    (DE, "data_element_enriched_surrogate_id", "string"),
    (DE, "data_element_prt_id", "string"),
    (DE, "data_element_phi", "boolean"),
    (DE, "data_element_modified_on", "timestamp"),
    (DE, "permissible_value_prt_id", "string"),
]
DE_ROWS = [
    ("s1", "PRT1", False, ALTERED, "PRT10"),
    ("s2", "PRT1", False, ALTERED, "PRT11"),
    ("s3", "PRT2", True, ALTERED, None),
]
CONCEPT_ROWS = [
    ("PRT1", "Data Element", ["one"]),
    ("PRT10", "Permissible Value", []),
    ("PRT11", "Permissible Value", ["eleven", "xi"]),
    ("PRT3", "Permissible Value", None),
]


class FakeCursor:
    """Answers the projection's catalog, count and select statements from fixed rows."""

    def __init__(self, *, counts: dict[str, int] | None = None, altered_after: datetime = ALTERED):
        self.counts = counts or {DE: len(DE_ROWS), CONCEPT: len(CONCEPT_ROWS)}
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
        rows = DE_ROWS if relation == DE else CONCEPT_ROWS
        return [tuple(_as_numpy_array(v) for v in row) for row in rows]


def _as_numpy_array(value: Any) -> Any:
    import numpy as np

    return np.array(value) if isinstance(value, list) else value


def _specs(tmp_path: Path) -> list[Path]:
    paths = []
    for name, ids in (("demographics", ["PRT1", "PRT3"]), ("cancer_registry", ["PRT2", "PRT4"])):
        path = tmp_path / f"{name}_product_spec_cpd_v2.json"
        fields = [{"column_name": "mrn"}] + [{"de_id": de_id} for de_id in ids + ["PRT1"]]
        path.write_text(json.dumps({"fields": fields}), encoding="utf-8")
        paths.append(path)
    return paths


def _extract(tmp_path: Path, cursor: FakeCursor, **kwargs: Any) -> dict[str, Any]:
    from showcase.partner_timbr_benchmark.intake_extract import extract

    return extract(lambda: cursor, _specs(tmp_path), tmp_path / "out", **kwargs)


def test_complete_projection_lands_both_relations_under_source_names(tmp_path: Path) -> None:
    receipt = _extract(tmp_path, FakeCursor())
    duckdb_path = Path(receipt["duckdb"]["path"])

    with Staging(str(tmp_path / "staging.duckdb"), schemas=()) as staging:
        land = land_source(duckdb_path, staging, schema="ignored")

    assert receipt["complete"] is True
    assert land.medium == "database"
    landed = {r.name: (r.schema, r.row_count, dict(r.columns)) for r in land.relations}
    assert landed[DE][:2] == ("standard_library", 3)
    assert landed[CONCEPT][:2] == ("standard_library", 4)
    assert landed[DE][2]["data_element_phi"] == "BOOLEAN"
    assert landed[DE][2]["data_element_modified_on"] == "TIMESTAMP WITH TIME ZONE"
    assert landed[CONCEPT][2] == {
        "concept_prt_id": "VARCHAR", "concept_type_label": "VARCHAR", "synonym_name": "VARCHAR[]",
    }


def test_rows_keep_values_and_warehouse_order(tmp_path: Path) -> None:
    receipt = _extract(tmp_path, FakeCursor())

    with duckdb.connect(receipt["duckdb"]["path"], read_only=True) as con:
        concepts = con.execute(
            "SELECT concept_prt_id, synonym_name FROM standard_library.standard_library_concept"
        ).fetchall()
        comment = con.execute(
            "SELECT comment FROM duckdb_tables() WHERE table_name = ?", [DE]
        ).fetchone()

    assert concepts == [
        ("PRT1", ["one"]), ("PRT10", []), ("PRT11", ["eleven", "xi"]), ("PRT3", None),
    ]
    assert comment == ("data elements",)


def test_receipt_states_scope_assistance_and_identity(tmp_path: Path) -> None:
    cursor = FakeCursor()
    receipt = _extract(tmp_path, cursor)
    on_disk = json.loads((tmp_path / "out" / "extract_receipt.json").read_text(encoding="utf-8"))
    import hashlib

    digest = hashlib.sha256(Path(receipt["duckdb"]["path"]).read_bytes()).hexdigest()

    assert on_disk == receipt
    assert receipt["assistance"]["kind"] == "supplied"
    assert receipt["workload"]["de_ids_requested"] == ["PRT1", "PRT2", "PRT3", "PRT4"]
    assert receipt["workload"]["found_as_data_element"] == ["PRT1", "PRT2"]
    assert receipt["workload"]["found_as_concept_only"] == ["PRT3"]
    assert receipt["workload"]["missing"] == ["PRT4"]
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
    assert all("'PRT4'" in s for s in selects)


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
    from showcase.partner_timbr_benchmark.intake_extract import extract

    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"fields": [{"de_id": "PRT1' OR '1'='1"}]}), encoding="utf-8")
    cursor = FakeCursor()

    with pytest.raises(ValueError, match="de_id"):
        extract(lambda: cursor, [spec], tmp_path / "out")
    assert cursor.statements == []


def test_recording_discovers_the_projection_file_not_the_warehouse() -> None:
    text = README.read_text(encoding="utf-8")

    assert "intake_extract" in text
    assert 'sema target discover "$SOURCE"' in text
    assert "--warehouse" not in text
    assert "DATABRICKS_HTTP_PATH" not in text

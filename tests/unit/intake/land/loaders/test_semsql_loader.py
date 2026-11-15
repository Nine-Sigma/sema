"""Semsql loader: a prebuilt semantic-sql SQLite database lands as staging relations."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from sema.ingest.duckdb_staging import Staging

pytestmark = pytest.mark.unit

DB = Path(__file__).parents[4] / "fixtures" / "intake" / "semsql" / "ontology.db"
SCHEMA = "land_a3"
DERIVED_COLUMNS = {
    "semsql_terms": [
        "subject",
        "ontology",
        "label",
        "definition",
        "deprecated",
        "replaced_by",
        "synonyms",
        "_sema_locator",
    ],
    "semsql_edges": ["subject", "predicate", "object", "_sema_locator"],
    "semsql_synonyms": ["subject", "value", "predicate", "_sema_locator"],
    "semsql_mappings": [
        "subject_id",
        "predicate_id",
        "object_id",
        "mapping_justification",
        "_sema_locator",
    ],
}


@pytest.fixture
def staging(tmp_path: Path) -> Staging:
    handle = Staging(str(tmp_path / "staging.duckdb"), schemas=())
    yield handle
    handle.close()


def _land(path: Path, staging: Staging):  # type: ignore[no-untyped-def]
    from sema.intake.land.loaders.semsql import land_semsql

    return land_semsql(path, staging, schema=SCHEMA)


def _rows(staging: Staging, sql: str) -> list[tuple[object, ...]]:
    rows: list[tuple[object, ...]] = staging.execute(sql).fetchall()
    return rows


def test_fixture_lands_base_tables_views_and_derived_relations(staging: Staging) -> None:
    result = _land(DB, staging)

    names = {r.name for r in result.relations}
    assert {"statements", "entailed_edge", "prefix", "rdfs_label_statement"} <= names
    assert {
        "has_exact_synonym_statement",
        "has_mapping_statement",
        "has_dbxref_statement",
        "deprecated_node",
        "edge",
    } <= names
    assert set(DERIVED_COLUMNS) <= names
    for name, columns in DERIVED_COLUMNS.items():
        assert [c for c, _ in result.relation(name).columns] == columns
    assert result.relation("statements").row_count == 172
    assert result.medium == "semsql"


def test_deprecated_classes_carry_their_replacements(staging: Staging) -> None:
    _land(DB, staging)

    rows = _rows(staging, f"SELECT subject, deprecated, replaced_by FROM {SCHEMA}.semsql_terms")
    deprecated = {(s, r) for s, d, r in rows if d == "true"}
    assert deprecated == {("EX:0049", "EX:0002"), ("EX:0050", "EX:0003")}
    assert {d for _, d, _ in rows} == {"true", "false"}
    assert len(rows) == 50


def test_mappings_keep_their_predicate_and_unspecified_justification(
    staging: Staging,
) -> None:
    _land(DB, staging)

    rows = _rows(
        staging,
        "SELECT subject_id, predicate_id, object_id, mapping_justification"
        f" FROM {SCHEMA}.semsql_mappings ORDER BY 1",
    )
    assert rows == [
        ("EX:0002", "skos:exactMatch", "NCIT:C1000", "semapv:UnspecifiedMatching"),
        ("EX:0003", "skos:exactMatch", "NCIT:C1001", "semapv:UnspecifiedMatching"),
        ("EX:0004", "skos:closeMatch", "NCIT:C1002", "semapv:UnspecifiedMatching"),
        ("EX:0005", "oio:hasDbXref", "MESH:D000001", "semapv:UnspecifiedMatching"),
    ]


def test_locator_is_the_subject_on_every_relation_with_one(staging: Staging) -> None:
    result = _land(DB, staging)

    for relation in result.relations:
        columns = dict(relation.columns)
        assert "_sema_locator" in columns
        assert relation.provenance.locator_kind == "row"
        subject = next((c for c in ("subject", "subject_id", "id") if c in columns), None)
        if subject is None:
            continue
        mismatched = _rows(
            staging,
            f"SELECT count(*) FROM {SCHEMA}.{relation.name}"
            f" WHERE _sema_locator IS DISTINCT FROM {subject}",
        )
        assert mismatched == [(0,)], relation.name
    statement = _rows(
        staging,
        f"SELECT _sema_locator FROM {SCHEMA}.statements WHERE subject = 'EX:0001' LIMIT 1",
    )
    assert statement == [("EX:0001",)]


def test_no_duckdb_extension_is_loaded(staging: Staging) -> None:
    _land(DB, staging)

    loaded = _rows(
        staging,
        "SELECT loaded FROM duckdb_extensions() WHERE extension_name = 'sqlite_scanner'",
    )
    assert loaded in ([], [(False,)])


def test_file_without_statements_fails_before_any_write(
    tmp_path: Path, staging: Staging
) -> None:
    path = tmp_path / "other.db"
    with sqlite3.connect(path) as con:
        con.execute("CREATE TABLE prefix (prefix TEXT, base TEXT)")
    con.close()

    with pytest.raises(ValueError, match="statements"):
        _land(path, staging)

    schemas = _rows(staging, "SELECT schema_name FROM duckdb_schemas()")
    assert (SCHEMA,) not in schemas


def test_source_identity_carries_both_iris_and_no_hash(staging: Staging) -> None:
    result = _land(DB, staging)

    assert result.source_identity == (
        ("ontology", "http://example.org/ex.owl"),
        ("version", "http://example.org/ex/2026-10-01/ex.owl"),
    )


def test_terms_carry_exact_synonyms_as_a_json_array(staging: Staging) -> None:
    _land(DB, staging)

    rows = _rows(
        staging,
        f"SELECT subject, synonyms FROM {SCHEMA}.semsql_terms"
        " WHERE subject IN ('EX:0001', 'EX:0002') ORDER BY 1",
    )
    assert rows == [("EX:0001", "[]"), ("EX:0002", '["Category 1 synonym"]')]

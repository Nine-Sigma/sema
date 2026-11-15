"""Database-file loader: a DuckDB file lands its base tables under their own schema names.

Pinned API (``sema.intake.land.loaders.database``)::

    land_database(path, staging, *, schema=None, policy_class=DEFAULT_POLICY_CLASS)
        -> LandResult

- The file is opened read-only; its bytes and modification time are unchanged and no
  write-ahead log appears beside it. A file without write permission lands.
- Every base table of every non-system schema lands as ``<schema>.<table>`` in staging,
  with the source schema and table names kept: the ``schema`` keyword (the ``Loader``
  call shape) does not rename anything, and no relation lands in it.
- System schemas are the catalog's internal ones (``information_schema``,
  ``pg_catalog``); a user table in ``main`` lands as ``main.<table>``. Views do not land.
- Rows are copied unchanged: landed columns equal the source columns and types (no
  ``_sema_locator`` column), and row counts equal the source counts.
- ``LandResult.medium == "database"``; ``source_identity == (("sha256", <file digest>),)``.
- Provenance per relation: ``loader == "database"``, the file URI, the default
  ``metadata_only`` policy class, ``locator_kind == "relation"`` with ``<schema>.<table>``
  as the locator, like the Athena loader whose rows also land unchanged.

Detection (``sema.intake.land.loaders.detect.detect_medium(path) -> Medium | None``) reads a
file's header bytes, not its suffix: a DuckDB file is ``"database"`` under any name, a
semantic-sql SQLite file is still ``"semsql"``, a CSV is still ``"tabular"``.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import duckdb
import pytest

from sema.ingest.duckdb_staging import Staging

pytestmark = pytest.mark.unit

FIXTURES = Path(__file__).parents[4] / "fixtures" / "intake"
LIBRARY = FIXTURES / "partner_sl" / "library.duckdb"
LIBRARY_SPEC = FIXTURES / "partner_sl" / "standard_library.spec.yaml"
SEMSQL_DB = FIXTURES / "semsql" / "ontology.db"
LIBRARY_SCHEMA = "standard_library"
LIBRARY_TABLES = (
    "data_element_mappings",
    "permissible_value_mappings_edited",
    "standard_library_concept",
    "standard_library_concept_relationship",
    "standard_library_data_element",
)
LANDING_KEYWORD = "land_unused"
SMALL_TABLES = {
    ("sales", "orders"): 3,
    ("sales", "customers"): 2,
    ("catalog", "items"): 4,
    ("main", "notes"): 1,
}
SMALL_VIEWS = (("sales", "big_orders"), ("main", "note_view"))
SMALL_COLUMNS = {
    ("sales", "orders"): (("order_id", "INTEGER"), ("customer_id", "INTEGER"),
                          ("amount", "DOUBLE"), ("tags", "VARCHAR[]")),
    ("sales", "customers"): (("customer_id", "INTEGER"), ("name", "VARCHAR")),
    ("catalog", "items"): (("code", "VARCHAR"), ("active", "BOOLEAN")),
    ("main", "notes"): (("body", "VARCHAR"),),
}
EXPECTED_IMPORT = {
    "concepts": 12,
    "value_sets": 12,
    "terms": 36,
    "relationships": 3,
    "column_bindings": 2,
    "value_mappings": 3,
}


def _build_small(path: Path) -> Path:
    with duckdb.connect(str(path)) as con:
        con.execute("CREATE SCHEMA sales")
        con.execute("CREATE SCHEMA catalog")
        con.execute(
            "CREATE TABLE sales.orders (order_id INTEGER, customer_id INTEGER, "
            "amount DOUBLE, tags VARCHAR[])"
        )
        con.execute(
            "INSERT INTO sales.orders VALUES (1, 10, 5.5, ['a']), "
            "(2, 10, 120.0, ['a', 'b']), (3, 11, 80.25, NULL)"
        )
        con.execute("CREATE TABLE sales.customers (customer_id INTEGER, name VARCHAR)")
        con.execute("INSERT INTO sales.customers VALUES (10, 'Ann'), (11, 'Bo')")
        con.execute("CREATE TABLE catalog.items (code VARCHAR, active BOOLEAN)")
        con.execute(
            "INSERT INTO catalog.items VALUES ('X1', true), ('X2', false), "
            "('X3', NULL), ('X4', true)"
        )
        con.execute("CREATE TABLE main.notes (body VARCHAR)")
        con.execute("INSERT INTO main.notes VALUES ('first')")
        con.execute("CREATE VIEW sales.big_orders AS SELECT * FROM sales.orders WHERE amount > 50")
        con.execute("CREATE VIEW main.note_view AS SELECT body FROM main.notes")
    return path


def _read_only_copy(source: Path, target: Path) -> Path:
    shutil.copyfile(source, target)
    target.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    return target


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _land(path: Path, staging: Staging, **kwargs: Any) -> Any:
    from sema.intake.land.loaders.database import land_database

    return land_database(path, staging, **kwargs)


def _keys(result: Any) -> set[tuple[str, str]]:
    return {(r.schema, r.name) for r in result.relations}


def _landed(result: Any, schema: str, name: str) -> Any:
    return next(r for r in result.relations if (r.schema, r.name) == (schema, name))


def _staged_columns(staging: Staging, schema: str, name: str) -> tuple[tuple[str, str], ...]:
    rows = staging.execute(
        "SELECT column_name, data_type FROM duckdb_columns() "
        "WHERE schema_name = ? AND table_name = ? ORDER BY column_index",
        [schema, name],
    ).fetchall()
    return tuple((str(column), str(kind)) for column, kind in rows)


def _staged_rows(staging: Staging, schema: str, name: str) -> list[tuple[Any, ...]]:
    rows: list[tuple[Any, ...]] = staging.execute(
        f'SELECT * FROM "{schema}"."{name}" ORDER BY ALL'
    ).fetchall()
    return rows


def _source_rows(path: Path, schema: str, name: str) -> list[tuple[Any, ...]]:
    with duckdb.connect(str(path), read_only=True) as con:
        rows: list[tuple[Any, ...]] = con.execute(
            f'SELECT * FROM "{schema}"."{name}" ORDER BY ALL'
        ).fetchall()
    return rows


@pytest.fixture
def staging(tmp_path: Path) -> Iterator[Staging]:
    handle = Staging(str(tmp_path / "staging.duckdb"), schemas=())
    yield handle
    handle.close()


@pytest.fixture
def small(tmp_path: Path) -> Path:
    built = _build_small(tmp_path / "built.duckdb")
    (tmp_path / "source").mkdir()
    return _read_only_copy(built, tmp_path / "source" / "small.duckdb")


# --- 1. Landing a database file --------------------------------------------------------


def test_every_base_table_of_every_user_schema_lands_under_its_own_name(
    small: Path, staging: Staging
) -> None:
    result = _land(small, staging)

    assert result.medium == "database"
    assert _keys(result) == set(SMALL_TABLES)
    for (schema, name), count in SMALL_TABLES.items():
        landed = _landed(result, schema, name)
        assert landed.row_count == count
        assert landed.columns == SMALL_COLUMNS[(schema, name)]
        assert _staged_columns(staging, schema, name) == SMALL_COLUMNS[(schema, name)]
        assert _staged_rows(staging, schema, name) == _source_rows(small, schema, name)


def test_views_and_system_schemas_do_not_land(small: Path, staging: Staging) -> None:
    result = _land(small, staging)

    keys = _keys(result)
    assert not keys & set(SMALL_VIEWS)
    assert not {schema for schema, _ in keys} & {"information_schema", "pg_catalog"}
    staged_views = staging.execute(
        "SELECT schema_name, view_name FROM duckdb_views() WHERE NOT internal"
    ).fetchall()
    assert not set(staged_views) & set(SMALL_VIEWS)


def test_the_schema_keyword_renames_nothing(small: Path, staging: Staging) -> None:
    result = _land(small, staging, schema=LANDING_KEYWORD)

    assert _keys(result) == set(SMALL_TABLES)
    assert LANDING_KEYWORD not in {r.schema for r in result.relations}
    tables = staging.execute(
        "SELECT count(*) FROM duckdb_tables() WHERE schema_name = ?", [LANDING_KEYWORD]
    ).fetchone()
    assert tables == (0,)


def test_each_relation_records_land_provenance(small: Path, staging: Staging) -> None:
    from sema.intake.land.loaders.base import DEFAULT_POLICY_CLASS

    result = _land(small, staging)

    assert result.source_identity == (("sha256", _sha256(small)),)
    assert result.policy_class == DEFAULT_POLICY_CLASS
    for relation in result.relations:
        record = relation.provenance
        assert record.source_uri == small.resolve().as_uri()
        assert record.loader == "database"
        assert record.loader_version
        assert record.landed_at
        assert record.policy_class == DEFAULT_POLICY_CLASS
        assert record.locator_kind == "relation"
        assert record.locator == f"{relation.schema}.{relation.name}"


def test_the_source_file_is_opened_read_only_and_never_written(
    small: Path, staging: Staging
) -> None:
    before = (_sha256(small), os.stat(small).st_mtime_ns)
    siblings = set(small.parent.iterdir())

    _land(small, staging)

    assert (_sha256(small), os.stat(small).st_mtime_ns) == before
    assert set(small.parent.iterdir()) == siblings


def test_the_library_fixture_lands_its_five_relations_unchanged(
    tmp_path: Path, staging: Staging
) -> None:
    source = _read_only_copy(LIBRARY, tmp_path / LIBRARY.name)

    result = _land(source, staging)

    assert _keys(result) == {(LIBRARY_SCHEMA, name) for name in LIBRARY_TABLES}
    for name in LIBRARY_TABLES:
        landed = _landed(result, LIBRARY_SCHEMA, name)
        source_rows = _source_rows(source, LIBRARY_SCHEMA, name)
        assert landed.row_count == len(source_rows) > 0
        assert _staged_columns(staging, LIBRARY_SCHEMA, name) == landed.columns
        assert _staged_rows(staging, LIBRARY_SCHEMA, name) == source_rows


# --- 2. Detection by header bytes ------------------------------------------------------


def _detect(path: Path) -> Any:
    from sema.intake.land.loaders.detect import detect_medium

    return detect_medium(path)


@pytest.mark.parametrize("name", ["library.duckdb", "library.db", "library.bin", "library"])
def test_a_duckdb_file_is_detected_as_a_database_under_any_name(
    tmp_path: Path, name: str
) -> None:
    source = tmp_path / name
    shutil.copyfile(LIBRARY, source)

    assert _detect(source) == "database"


@pytest.mark.parametrize("name", ["ontology.db", "ontology.duckdb"])
def test_a_semsql_file_is_still_semsql(tmp_path: Path, name: str) -> None:
    source = tmp_path / name
    shutil.copyfile(SEMSQL_DB, source)

    assert _detect(source) == "semsql"


def test_a_csv_is_still_tabular(tmp_path: Path) -> None:
    source = tmp_path / "codes.csv"
    source.write_text("code,label\nA,Alpha\nB,Beta\n")

    assert _detect(source) == "tabular"


def test_a_text_file_with_a_database_suffix_is_not_a_database(tmp_path: Path) -> None:
    source = tmp_path / "notes.duckdb"
    source.write_text("not a database\n")

    assert _detect(source) != "database"


# --- 3. A1 end to end without the model ------------------------------------------------


def _import(staging: Staging) -> Any:
    from sema.binding.importer import import_concept_source
    from sema.binding.spec import load_concept_source_spec, load_dataset_map
    from sema.intake.land.relations import RelationProvider

    spec = load_concept_source_spec(LIBRARY_SPEC)
    assert spec.authored_bindings is not None
    dataset_map = load_dataset_map(spec.authored_bindings.dataset_map)
    provider = RelationProvider(warehouse=None, staging=staging)
    return import_concept_source(spec, provider, dataset_map=dataset_map)


def _counts(imported: Any) -> dict[str, int]:
    manifest = imported.manifest
    (scheme,) = manifest["entities"]
    return {
        "concepts": len(scheme["properties"]),
        "value_sets": len(manifest["vocabularies"]),
        "terms": len(manifest["terms"]),
        "relationships": len(manifest["relationships"]),
        "column_bindings": len(imported.authored.column_bindings),
        "value_mappings": len(imported.authored.value_mappings),
    }


def test_the_hand_written_spec_imports_from_the_landed_library_as_from_the_fixture(
    tmp_path: Path, staging: Staging
) -> None:
    source = _read_only_copy(LIBRARY, tmp_path / LIBRARY.name)
    direct_copy = tmp_path / "direct" / LIBRARY.name
    direct_copy.parent.mkdir()
    shutil.copyfile(LIBRARY, direct_copy)
    with Staging(str(direct_copy), schemas=()) as direct:
        expected = _import(direct)

    _land(source, staging)
    landed = _import(staging)

    assert _counts(landed) == EXPECTED_IMPORT
    assert _counts(expected) == EXPECTED_IMPORT
    assert landed.manifest == expected.manifest
    assert landed.authored == expected.authored

"""Dictionary loader: a DDL file lands as one row per column."""

from __future__ import annotations

from pathlib import Path

import pytest

from sema.ingest.duckdb_staging import Staging

pytestmark = pytest.mark.unit

DDL = Path(__file__).parents[4] / "fixtures" / "intake" / "omop" / "OMOPCDM_postgresql_5.4_ddl.sql"
SCHEMA = "land_ddl"
RELATION = "OMOPCDM_postgresql_5_4_ddl__columns"


@pytest.fixture
def staging(tmp_path: Path) -> Staging:
    handle = Staging(str(tmp_path / "staging.duckdb"), schemas=())
    yield handle
    handle.close()


def _land(staging: Staging):  # type: ignore[no-untyped-def]
    from sema.intake.land.loaders.dictionary import land_ddl

    return land_ddl(DDL, staging, schema=SCHEMA)


def test_ddl_lands_one_row_per_column(staging: Staging) -> None:
    from sema.intake.land.loaders.dictionary_utils import parse_postgres_ddl

    result = _land(staging)

    relation = result.relation(RELATION)
    assert [name for name, _ in relation.columns] == [
        "entity",
        "property",
        "raw_type",
        "nullable",
        "ordinal",
        "_sema_locator",
    ]
    rows = staging.execute(
        f"SELECT entity, property, nullable FROM {SCHEMA}.{RELATION}"
    ).fetchall()
    parsed = parse_postgres_ddl(DDL.read_text())
    expected = {
        (table, column.name, column.nullable)
        for table, columns in parsed.items()
        for column in columns
    }
    assert set(rows) == expected
    assert relation.row_count == len(expected)
    assert result.medium == "dictionary"


def test_locator_names_file_entity_and_property(staging: Staging) -> None:
    _land(staging)

    rows = staging.execute(
        f"SELECT raw_type, ordinal, _sema_locator FROM {SCHEMA}.{RELATION}"
        " WHERE entity = 'person' AND property = 'person_id'"
    ).fetchall()

    assert rows == [("integer", 1, f"{DDL.name}!person.person_id")]


def test_provenance_is_row_kind(staging: Staging) -> None:
    result = _land(staging)

    record = result.relation(RELATION).provenance
    assert record.locator_kind == "row"
    assert record.loader == "dictionary"
    assert record.locator is None


def test_showcase_re_exports_the_core_parser() -> None:
    from sema.intake.land.loaders import dictionary_utils
    from showcase.cbioportal_to_omop import omop_ingest, omop_ingest_utils

    assert omop_ingest_utils.parse_postgres_ddl is dictionary_utils.parse_postgres_ddl
    assert omop_ingest_utils.postgres_to_duckdb_type is dictionary_utils.postgres_to_duckdb_type
    assert omop_ingest.parse_postgres_ddl is dictionary_utils.parse_postgres_ddl
    comments = omop_ingest_utils.load_field_level_comments
    assert comments.__module__ == "showcase.cbioportal_to_omop.omop_ingest_utils"

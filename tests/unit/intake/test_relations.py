"""RelationProvider: a spec relation resolves to a warehouse or staging query."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest

from sema.binding.spec import ConceptSourceSpec, RelationRef
from sema.ingest.duckdb_staging import Staging
from tests.unit.intake.warehouse_env import (
    CATALOG,
    CODES_COLUMNS,
    CONCEPT_COLUMNS,
    SCHEMA,
    TABLE,
    UNREADABLE_REASON,
    FakeConnector,
    schema_metadata,
    unreadable,
)

pytestmark = pytest.mark.unit

HOSTILE = "x'); DROP TABLE t; --"
FIXED_SELECT = re.compile(r"^SELECT [\w\", ]+ FROM [\w\".]+( WHERE .+)?$", re.DOTALL)


def _staging(tmp_path: Path) -> Staging:
    staging = Staging(str(tmp_path / "staging.duckdb"), schemas=("land",))
    rows = pa.table(
        {"code": ["A", "B"], "label": ["Alpha", HOSTILE], "parent": [None, "A"]}
    )
    staging.write_table(
        "land",
        "codes",
        rows,
        {"code": "VARCHAR", "label": "VARCHAR", "parent": "VARCHAR"},
        {},
        None,
    )
    return staging


def test_staging_ref_queries_the_staging_table(tmp_path: Path) -> None:
    from sema.intake.relations import RelationProvider

    staging = _staging(tmp_path)
    provider = RelationProvider(warehouse=None, staging=staging)

    query = provider.query(RelationRef(provider="staging", name="land.codes"))

    assert query("SELECT code FROM land.codes ORDER BY code") == [("A",), ("B",)]
    staging.close()


def test_warehouse_ref_calls_the_supplied_query_function() -> None:
    from sema.intake.relations import RelationProvider

    calls: list[str] = []

    def warehouse(sql: str) -> list[tuple[Any, ...]]:
        calls.append(sql)
        return [("row",)]

    provider = RelationProvider(warehouse=warehouse, staging=None)
    query = provider.query(RelationRef(provider="warehouse", name="cat.s.t"))

    assert query("SELECT a FROM cat.s.t") == [("row",)]
    assert calls == ["SELECT a FROM cat.s.t"]


def test_require_names_provider_and_first_relation_before_any_query() -> None:
    from sema.intake.relations import MissingProviderError, RelationProvider

    calls: list[str] = []

    def warehouse(sql: str) -> list[tuple[Any, ...]]:
        calls.append(sql)
        return []

    provider = RelationProvider(warehouse=warehouse, staging=None)
    refs = [
        RelationRef(provider="warehouse", name="cat.s.t"),
        RelationRef(provider="staging", name="land.first"),
        RelationRef(provider="staging", name="land.second"),
    ]

    with pytest.raises(MissingProviderError, match=r"staging.*land\.first"):
        provider.require(refs)
    assert calls == []


def test_query_text_stays_fixed_select_forms_for_hostile_cells(tmp_path: Path) -> None:
    from sema.binding.importer import import_concept_source
    from sema.intake.relations import RelationProvider

    staging = _staging(tmp_path)
    statements: list[str] = []
    run = staging.execute

    def recording(sql: str, params: Any = None) -> Any:
        statements.append(sql)
        return run(sql, params)

    staging.execute = recording  # type: ignore[method-assign]
    spec = ConceptSourceSpec.model_validate(
        {
            "spec_version": 2,
            "id": "codes",
            "identity_system": "ex",
            "release": "1",
            "default_provider": "staging",
            "scheme": {"qualified_name": "ex.codes", "description": "Codes"},
            "concepts": {
                "relation": "land.codes",
                "group_by": "code",
                "roles": {"concept_id": "code", "name": "label"},
            },
            "hierarchy": {"parent_column": "parent"},
        }
    )

    result = import_concept_source(
        spec, RelationProvider(warehouse=None, staging=staging), dataset_map={}
    )

    assert statements
    assert all(FIXED_SELECT.match(sql) for sql in statements), statements
    assert not [sql for sql in statements if "DROP" in sql]
    names = {p["name"] for e in result.manifest["entities"] for p in e["properties"]}
    assert HOSTILE in names
    staging.close()


STAGING_REF = RelationRef(provider="staging", name="land.codes")
WAREHOUSE_REF = RelationRef(provider="warehouse", name="cat.schema.concept")
PUNCTUATED = "Term, Acronym, or Code Definitions"


def _provider() -> Any:
    from sema.intake.relations import RelationProvider

    return RelationProvider(warehouse=lambda sql: [], staging=None)


@pytest.mark.parametrize(
    ("ref", "identifier", "expected"),
    [
        (STAGING_REF, "code", '"code"'),
        (STAGING_REF, "Column Name", '"Column Name"'),
        (STAGING_REF, PUNCTUATED, f'"{PUNCTUATED}"'),
        (STAGING_REF, "table", '"table"'),
        (STAGING_REF, 'say "hi"', '"say ""hi"""'),
        (WAREHOUSE_REF, "code", "`code`"),
        (WAREHOUSE_REF, "Column Name", "`Column Name`"),
        (WAREHOUSE_REF, 'say "hi"', '`say "hi"`'),
    ],
)
def test_quote_uses_the_provider_identifier_quote(
    ref: RelationRef, identifier: str, expected: str
) -> None:
    assert _provider().quote(ref, identifier) == expected


def test_warehouse_rejects_an_identifier_holding_a_backtick() -> None:
    with pytest.raises(ValueError, match=r"(?i)backtick.*odd`name|odd`name.*backtick"):
        _provider().quote(WAREHOUSE_REF, "odd`name")


@pytest.mark.parametrize(
    ("ref", "expected"),
    [
        (STAGING_REF, '"land"."codes"'),
        (
            RelationRef(provider="staging", name="land_a2.codes_a2__Concepts"),
            '"land_a2"."codes_a2__Concepts"',
        ),
        (RelationRef(provider="staging", name="codes"), '"codes"'),
        (WAREHOUSE_REF, "`cat`.`schema`.`concept`"),
    ],
)
def test_relation_sql_splits_the_dotted_name_and_quotes_each_part(
    ref: RelationRef, expected: str
) -> None:
    assert _provider().relation_sql(ref) == expected


def test_relation_sql_rejects_a_warehouse_part_holding_a_backtick() -> None:
    ref = RelationRef(provider="warehouse", name="cat.sch`ema.concept")

    with pytest.raises(ValueError, match=r"(?i)backtick"):
        _provider().relation_sql(ref)


def test_warehouse_statement_names_the_relation_with_backticks() -> None:
    from sema.binding.importer import import_concept_source
    from sema.intake.relations import RelationProvider

    statements: list[str] = []

    def warehouse(sql: str) -> list[tuple[Any, ...]]:
        statements.append(sql)
        return [("A", "Alpha")]

    spec = ConceptSourceSpec.model_validate(
        {
            "spec_version": 2,
            "id": "codes",
            "identity_system": "ex",
            "release": "1",
            "default_provider": "warehouse",
            "scheme": {"qualified_name": "ex.codes", "description": "Codes"},
            "concepts": {
                "relation": "cat.schema.concept",
                "group_by": "code",
                "roles": {"concept_id": "code", "name": "label"},
            },
        }
    )

    import_concept_source(
        spec, RelationProvider(warehouse=warehouse, staging=None), dataset_map={}
    )

    assert statements == ["SELECT `code`, `label` FROM `cat`.`schema`.`concept`"]


# warehouse Land (design Decision 8)

CATALOG_STATEMENT = re.compile(r"\bDESCRIBE\b|\bSHOW\b|\binformation_schema\b", re.IGNORECASE)


def test_warehouse_land_reads_the_schema_once_and_fills_columns_types_comments(
    tmp_path: Path,
) -> None:
    from sema.intake.ledger import Ledger
    from sema.intake.loaders.warehouse import land_warehouse

    connector = FakeConnector(schema_metadata(), query=lambda sql: [(3,)])

    land = land_warehouse(
        connector, catalog=CATALOG, schema=SCHEMA, ledger=Ledger(tmp_path).run("r")
    )

    assert connector.metadata_reads == [(CATALOG, SCHEMA)]
    assert land.medium == "warehouse"
    namespace = f"{CATALOG}.{SCHEMA}"
    assert sorted((r.schema, r.name) for r in land.relations) == [
        (namespace, "codes"),
        (namespace, TABLE),
    ]
    concept = land.relation(TABLE)
    assert concept.columns == tuple((c.name, c.data_type) for c in CONCEPT_COLUMNS)
    assert dict(concept.comments) == {c.name: c.comment for c in CONCEPT_COLUMNS if c.comment}
    codes = land.relation("codes")
    assert codes.columns == tuple((c.name, c.data_type) for c in CODES_COLUMNS)
    assert dict(codes.comments) == {"code": "a code"}
    assert not [q for q in connector.queries if CATALOG_STATEMENT.search(q)]


def test_warehouse_land_counts_rows_through_the_logged_path(tmp_path: Path) -> None:
    from sema.intake.ledger import Ledger
    from sema.intake.loaders.warehouse import land_warehouse

    connector = FakeConnector(schema_metadata(), query=lambda sql: [(3,)])

    land = land_warehouse(
        connector, catalog=CATALOG, schema=SCHEMA, ledger=Ledger(tmp_path).run("r")
    )

    assert {r.name: r.row_count for r in land.relations} == {TABLE: 3, "codes": 3}
    logged = (tmp_path / "r" / "sql.jsonl").read_text().splitlines()
    lines = [json.loads(line) for line in logged]
    assert [line["statement"] for line in lines] == connector.queries
    assert all(re.fullmatch(r"SELECT COUNT\(\*\) FROM \S+", q) for q in connector.queries)
    assert {(line["stage"], line["provider"], line["value_bearing"]) for line in lines} == {
        ("land", "warehouse", False)
    }


def test_warehouse_land_fails_naming_the_unreadable_reason(tmp_path: Path) -> None:
    from sema.intake.ledger import Ledger
    from sema.intake.loaders.warehouse import land_warehouse

    connector = FakeConnector(unreadable())

    with pytest.raises(ValueError, match=re.escape(UNREADABLE_REASON)):
        land_warehouse(
            connector, catalog=CATALOG, schema=SCHEMA, ledger=Ledger(tmp_path).run("r")
        )

    assert connector.metadata_reads == [(CATALOG, SCHEMA)]
    assert connector.queries == []
    assert not (tmp_path / "r" / "sql.jsonl").exists()

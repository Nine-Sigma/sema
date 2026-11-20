"""Importer statements quote every spec identifier with the provider's quote."""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

from sema.binding.spec import ConceptSourceSpec
from sema.ingest.duckdb_staging import Staging
from sema.targets.adapters.manifest_parser import parse_manifest_raw
from tests.unit.binding.fixtures.concept_library import (
    SPEC_PATH,
    build_library,
    load_dataset_map,
    provider_for,
    query_fn,
)

pytestmark = pytest.mark.unit

PUNCTUATED = "Term, Acronym, or Code Definitions"
DICTIONARY = f"""\
table,column,Column Name,Data Type,"{PUNCTUATED}"
VISIT,VISIT_ID,Visit key,integer,Identifies one visit
VISIT,CARE_SITE,Care site,text,Where the visit took place
PERSON,PERSON_ID,Person key,integer,Identifies one person
PERSON,NOTE,Free note,skip,Not part of the model
"""
STATEMENT_KEYWORDS = {"SELECT", "FROM", "WHERE", "IN", "OR", "AND"}
DOUBLE_QUOTED = re.compile(r'"(?:[^"]|"")*"')
BACKTICKED = re.compile(r"`(?:[^`]|``)*`")
WORD = re.compile(r"[A-Za-z_]\w*")


def _spec(**sections: Any) -> ConceptSourceSpec:
    raw: dict[str, Any] = {
        "spec_version": 2,
        "id": "quoted",
        "identity_system": "ex",
        "release": "1",
        "default_provider": "staging",
    }
    raw.update(sections)
    return ConceptSourceSpec.model_validate(raw)


def _concepts(**extra: Any) -> dict[str, Any]:
    return {
        "scheme": {"qualified_name": "ex.dictionary", "description": "Dictionary"},
        "concepts": {
            "relation": "land.dictionary",
            "group_by": "Column Name",
            "roles": {
                "concept_id": "Column Name",
                "name": "Column Name",
                "definition": PUNCTUATED,
            },
            **extra,
        },
    }


ENTITY_MODEL = {
    "entity_model": {
        "relation": "land.dictionary",
        "entity_case": "lower",
        "roles": {
            "entity": "table",
            "property": "column",
            "type": "Data Type",
            "description": PUNCTUATED,
        },
    }
}


@pytest.fixture
def landed(tmp_path: Path) -> Iterator[tuple[Staging, list[str]]]:
    from sema.intake.land.loaders.tabular import land_tabular

    path = tmp_path / "dictionary.csv"
    path.write_text(DICTIONARY)
    with Staging(str(tmp_path / "staging.duckdb"), schemas=()) as staging:
        land_tabular(path, staging, schema="land")
        statements: list[str] = []
        run = staging.execute

        def recording(sql: str, params: Any = None) -> Any:
            statements.append(sql)
            return run(sql, params)

        staging.execute = recording  # type: ignore[method-assign]
        yield staging, statements


def _import(staging: Staging, spec: ConceptSourceSpec) -> Any:
    from sema.binding.importer import import_concept_source
    from sema.intake.land.relations import RelationProvider

    return import_concept_source(
        spec, RelationProvider(warehouse=None, staging=staging), dataset_map={}
    )


def _parsed(result: Any) -> Any:
    return parse_manifest_raw(yaml.safe_load(yaml.safe_dump(result.manifest)))


def _unquoted_words(sql: str, quoted: re.Pattern[str]) -> set[str]:
    return set(WORD.findall(quoted.sub(" ", sql))) - STATEMENT_KEYWORDS


def _selects(statements: list[str]) -> list[str]:
    return [sql for sql in statements if sql.startswith("SELECT ")]


def test_columns_with_spaces_and_punctuation_import_and_are_quoted(
    landed: tuple[Staging, list[str]],
) -> None:
    staging, statements = landed

    result = _import(staging, _spec(**_concepts()))

    props = {p.name: p for e in _parsed(result).entities for p in e.properties}
    assert props["Visit key"].definition == "Identifies one visit"
    assert props["Person key"].definition == "Identifies one person"
    selects = _selects(statements)
    assert selects
    assert all('"Column Name"' in sql for sql in selects), selects
    assert any(f'"{PUNCTUATED}"' in sql for sql in selects), selects
    assert all('FROM "land"."dictionary"' in sql for sql in selects), selects


def test_sql_keyword_columns_import_without_a_syntax_error(
    landed: tuple[Staging, list[str]],
) -> None:
    staging, statements = landed

    result = _import(staging, _spec(**ENTITY_MODEL))

    entities = {e.qualified_name: e for e in _parsed(result).entities}
    assert sorted(entities) == ["ex.person", "ex.visit"]
    visit = {p.name: p for p in entities["ex.visit"].properties}
    assert sorted(visit) == ["care_site", "visit_id"]
    assert visit["care_site"].definition == "Where the visit took place"
    assert all('"table"' in sql and '"column"' in sql for sql in _selects(statements))


def test_every_staging_identifier_is_double_quoted(
    landed: tuple[Staging, list[str]],
) -> None:
    staging, statements = landed

    _import(staging, _spec(**_concepts(), **ENTITY_MODEL))

    selects = _selects(statements)
    assert selects
    assert [_unquoted_words(sql, DOUBLE_QUOTED) for sql in selects] == [set()] * len(
        selects
    ), selects


def test_filter_fragment_stays_verbatim(landed: tuple[Staging, list[str]]) -> None:
    staging, statements = landed
    fragment = "\"Data Type\" <> 'skip'"

    result = _import(staging, _spec(**_concepts(filter=fragment)))

    names = {p.name for e in _parsed(result).entities for p in e.properties}
    assert "Free note" not in names
    assert "Visit key" in names
    filtered = [sql for sql in _selects(statements) if " WHERE " in sql]
    assert filtered
    assert all(sql.endswith(f" WHERE {fragment}") for sql in filtered), filtered


def test_every_version_1_warehouse_identifier_is_backtick_quoted() -> None:
    from sema.binding.importer import import_concept_source
    from sema.binding.spec import load_concept_source_spec

    con = build_library()
    run = query_fn(con)
    statements: list[str] = []

    def recording(sql: str) -> list[tuple[Any, ...]]:
        statements.append(sql)
        return run(sql)

    try:
        import_concept_source(
            load_concept_source_spec(SPEC_PATH),
            provider_for(con, recording),
            dataset_map=load_dataset_map(),
        )
    finally:
        con.close()

    assert statements
    assert [_unquoted_words(sql, BACKTICKED) for sql in statements] == [set()] * len(
        statements
    ), statements
    assert all("FROM `lib`." in sql for sql in statements), statements

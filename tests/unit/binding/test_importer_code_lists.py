"""Importer over `code_lists`: standalone code tables become vocabularies and terms."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

from sema.binding.spec import ConceptSourceSpec
from sema.ingest.duckdb_staging import Staging
from sema.targets.adapters.manifest_parser import parse_manifest_raw

pytestmark = pytest.mark.unit

NAMESPACE = "vocabulary_ex_codes"
CODE_TABLES: dict[str, tuple[tuple[str, str], ...]] = {
    "Colour_Codes": (("R", "Red"), ("G", "Green"), ("B", "Blue")),
    "Size_Codes": (("S", "Small"), ("M", "Medium")),
    "Shape_Codes": (("SQ", "Square"), ("CI", "Circle"), ("TR", "Triangle"), ("ST", "Star")),
    "Unit_Codes": (("KG", "Kilogram"),),
    "Grade_Codes": (("1", "First"), ("2", "Second"), ("3", "Third"), ("4", "Fourth"),
                    ("5", "Fifth")),
}


@pytest.fixture
def staging(tmp_path: Path) -> Iterator[Staging]:
    with Staging(str(tmp_path / "codes.duckdb"), schemas=()) as handle:
        handle.execute("CREATE SCHEMA land")
        for name, rows in CODE_TABLES.items():
            handle.execute(f'CREATE TABLE land."{name}" ("KIND" VARCHAR, "CODE" VARCHAR,'
                           ' "LABEL" VARCHAR)')
            for code, label in rows:
                handle.execute(
                    f'INSERT INTO land."{name}" VALUES (?, ?, ?)', ["X", code, label]
                )
        yield handle


def _entry(name: str, **extra: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "vocabulary_id": name,
        "relation": f"land.{name}",
        "roles": {"code": "CODE", "display": "LABEL"},
    }
    entry.update(extra)
    return entry


def _spec(entries: list[dict[str, Any]], **extra: Any) -> ConceptSourceSpec:
    raw: dict[str, Any] = {
        "spec_version": 2,
        "id": "ex-codes",
        "identity_system": "ex",
        "release": "1",
        "default_provider": "staging",
        "code_lists": entries,
    }
    raw.update(extra)
    return ConceptSourceSpec.model_validate(raw)


def _import(spec: ConceptSourceSpec, source: Staging, staging: Staging | None = None) -> Any:
    from sema.binding.importer import import_concept_source
    from sema.intake.relations import RelationProvider

    provider = RelationProvider(warehouse=None, staging=source)
    return import_concept_source(spec, provider, dataset_map={}, staging=staging)


def _fetch(staging: Staging, sql: str) -> list[tuple[Any, ...]]:
    rows: list[tuple[Any, ...]] = staging.execute(sql).fetchall()
    return rows


def test_five_code_lists_yield_five_vocabularies_and_no_scheme(staging: Staging) -> None:
    result = _import(_spec([_entry(name) for name in CODE_TABLES]), staging)

    manifest = result.manifest
    vocabularies = {v["name"]: v["source"] for v in manifest["vocabularies"]}
    assert vocabularies == {name: "INLINE" for name in CODE_TABLES}
    counts = Counter(term["vocabulary"] for term in manifest["terms"])
    assert counts == {name: len(rows) for name, rows in CODE_TABLES.items()}
    assert manifest["entities"] == []
    assert manifest["relationships"] == []


def test_code_list_terms_carry_code_and_display(staging: Staging) -> None:
    result = _import(_spec([_entry("Colour_Codes")]), staging)

    terms = {(t["vocabulary"], t["code"], t["display"]) for t in result.manifest["terms"]}
    assert terms == {("Colour_Codes", code, label) for code, label in CODE_TABLES["Colour_Codes"]}


def test_code_list_with_concept_adds_one_scheme_property(staging: Staging) -> None:
    entries = [
        _entry("Colour_Codes", concept={"concept_id": "colour", "name": "Colour"}),
        _entry("Size_Codes"),
    ]
    scheme = {"qualified_name": "ex.codes", "description": "Code tables"}

    result = _import(_spec(entries, scheme=scheme), staging)

    (scheme_entity,) = result.manifest["entities"]
    assert scheme_entity["kind"] == "CONCEPT_SCHEME"
    assert scheme_entity["qualified_name"] == "ex.codes"
    (prop,) = scheme_entity["properties"]
    assert prop["concept_ref"]["id"] == "colour"
    assert prop["name"] == "Colour"
    assert prop["vocabulary_binding"] == {"vocabulary": "Colour_Codes"}


def test_code_list_manifest_parses(staging: Staging) -> None:
    entries = [_entry("Colour_Codes", concept={"concept_id": "colour", "name": "Colour"})]
    scheme = {"qualified_name": "ex.codes", "description": "Code tables"}

    result = _import(_spec(entries, scheme=scheme), staging)

    parsed = parse_manifest_raw(yaml.safe_load(yaml.safe_dump(result.manifest)))
    assert [v.name for v in parsed.vocabularies] == ["Colour_Codes"]
    assert len(parsed.terms) == 3


def test_code_list_above_inline_cap_is_staged_under_the_spec_namespace(
    staging: Staging,
) -> None:
    spec = _spec(
        [_entry("Shape_Codes"), _entry("Size_Codes")], term_storage={"inline_cap": 3}
    )

    result = _import(spec, staging, staging)

    vocabularies = {v["name"]: v["source"] for v in result.manifest["vocabularies"]}
    assert vocabularies == {"Shape_Codes": "EXTERNAL", "Size_Codes": "INLINE"}
    assert {t["vocabulary"] for t in result.manifest["terms"]} == {"Size_Codes"}
    staged = _fetch(
        staging,
        f"SELECT vocabulary_id, concept_code FROM {NAMESPACE}.concept ORDER BY concept_code",
    )
    assert staged == sorted(
        [("Shape_Codes", code) for code, _ in CODE_TABLES["Shape_Codes"]],
        key=lambda row: row[1],
    )


def test_code_list_above_inline_cap_without_staging_names_the_vocabulary(
    staging: Staging,
) -> None:
    spec = _spec([_entry("Shape_Codes")], term_storage={"inline_cap": 3})

    with pytest.raises(ValueError, match=r"Shape_Codes.*inline_cap 3"):
        _import(spec, staging)

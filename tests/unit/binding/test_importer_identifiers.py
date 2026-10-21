"""Imported entities get valid identifiers and card versions (bug-756)."""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from sema.binding.spec import ConceptSourceSpec
from sema.ingest.duckdb_staging import Staging

pytestmark = pytest.mark.unit

FIXTURES = Path(__file__).parents[2] / "fixtures" / "intake"
QUALIFIED_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*(\.[A-Za-z0-9_]+)+$")
EN_DASH = "–"

DICTIONARY = f"""\
table_name,column_name,table_note
Visit Log - Daily,visit_id,Daily visits
Care Site {EN_DASH} Main,site_id,Care sites
Person (v2)/Core,person_id,People
LAB,lab_id,Labs
"""
COLLIDING = """\
table_name,column_name,table_note
Real-Property Master,doc_id,Hyphenated
Real Property Master,doc_id,Spaced
"""

ENTITY_MODEL = {
    "relation": "land.dictionary",
    "roles": {
        "entity": "table_name",
        "property": "column_name",
        "entity_description": "table_note",
    },
}
CONVERTED = {
    "ex.visit_log_daily": "Visit Log - Daily",
    "ex.care_site_main": f"Care Site {EN_DASH} Main",
    "ex.person_v2_core": "Person (v2)/Core",
}
EXPECTED = CONVERTED | {"ex.LAB": "LAB"}


def _spec(release: str = "1") -> ConceptSourceSpec:
    return ConceptSourceSpec.model_validate(
        {
            "spec_version": 2,
            "id": "ex-model",
            "identity_system": "ex",
            "release": release,
            "default_provider": "staging",
            "entity_model": ENTITY_MODEL,
        }
    )


def _import(tmp_path: Path, spec: ConceptSourceSpec, dictionary: str = DICTIONARY) -> Any:
    from sema.binding.importer import import_concept_source
    from sema.intake.loaders.tabular import land_tabular
    from sema.intake.relations import RelationProvider

    path = tmp_path / "dictionary.csv"
    path.write_text(dictionary, encoding="utf-8")
    with Staging(str(tmp_path / "staging.duckdb"), schemas=()) as staging:
        land_tabular(path, staging, schema="land")
        return import_concept_source(
            spec, RelationProvider(warehouse=None, staging=staging), dataset_map={}
        )


def _normalize(result: Any, path: Path) -> Any:
    from sema.binding.importer import write_import
    from sema.targets.adapters.manifest import ManifestTargetAdapter
    from sema.targets.normalizer import TargetModelNormalizer

    write_import(result, path, path.with_suffix(".evidence.json"))
    return TargetModelNormalizer.normalize(ManifestTargetAdapter(path))


@pytest.mark.parametrize(
    ("source", "identifier"),
    [
        ("person", "person"),
        ("VISIT", "VISIT"),
        ("Caf\u00e9", "cafe"),
        ("ACRIS - Real Property Master", "acris_real_property_master"),
        (f"ACRIS {EN_DASH} Country Codes", "acris_country_codes"),
        ("Person (v2)/Core", "person_v2_core"),
        ("a...b", "a_b"),
        ("snake_case_name", "snake_case_name"),
    ],
)
def test_entity_identifier_keeps_valid_names_and_converts_others(
    source: str, identifier: str
) -> None:
    from sema.binding.identifier_utils import entity_identifier

    assert entity_identifier(source) == identifier
    assert QUALIFIED_NAME.match(f"ns.{entity_identifier(source)}")


@pytest.mark.parametrize(
    ("release", "version"),
    [
        ("2026-10-20", "2026.10.20"),
        ("2026-01-05", "2026.01.05"),
        ("1", "1"),
        ("1.0.0", "1.0.0"),
        ("2026.10", "2026.10"),
    ],
)
def test_card_version_maps_dates_and_keeps_valid_releases(
    release: str, version: str
) -> None:
    from sema.binding.identifier_utils import card_version

    assert card_version(release) == version


@pytest.mark.parametrize("release", ["latest", "2026/10/20", "release 7", "2026-13-45"])
def test_card_version_rejects_any_other_invalid_release(release: str) -> None:
    from sema.binding.identifier_utils import card_version

    with pytest.raises(ValueError, match=re.escape(release)):
        card_version(release)


def test_imported_entities_get_valid_qualified_names(tmp_path: Path) -> None:
    result = _import(tmp_path, _spec())

    names = [e["qualified_name"] for e in result.manifest["entities"]]
    assert sorted(names) == sorted(EXPECTED)
    assert all(QUALIFIED_NAME.match(name) for name in names)
    assert sorted(result.table_schemas) == sorted(EXPECTED)


def test_source_name_is_kept_as_the_context_card_label(tmp_path: Path) -> None:
    result = _import(tmp_path, _spec())

    cards = {e["qualified_name"]: e["context_card"] for e in result.manifest["entities"]}
    for qualified_name, source in CONVERTED.items():
        assert source in cards[qualified_name]["curated_synonyms"]
    assert "curated_synonyms" not in cards["ex.LAB"]


def test_two_names_converting_to_one_identifier_fail_naming_both(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError) as raised:
        _import(tmp_path, _spec(), COLLIDING)

    message = str(raised.value)
    assert "Real-Property Master" in message
    assert "Real Property Master" in message


def test_date_release_becomes_a_dotted_card_version(tmp_path: Path) -> None:
    result = _import(tmp_path, _spec(release="2026-10-20"))

    versions = {e["context_card"]["card_version"] for e in result.manifest["entities"]}
    assert versions == {"2026.10.20"}
    normalized = _normalize(result, tmp_path / "manifest.yaml")
    assert len(normalized.entities) == len(EXPECTED)


def test_valid_release_is_kept_as_the_card_version(tmp_path: Path) -> None:
    result = _import(tmp_path, _spec(release="3.1"))

    versions = {e["context_card"]["card_version"] for e in result.manifest["entities"]}
    assert versions == {"3.1"}


def test_invalid_release_fails_the_import_naming_it(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not-a-version"):
        _import(tmp_path, _spec(release="not-a-version"))


@pytest.fixture(scope="module")
def acris(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Any]:
    from sema.binding.importer import import_concept_source
    from sema.binding.spec import load_concept_source_spec
    from sema.intake.loaders.tabular import land_tabular
    from sema.intake.relations import RelationProvider

    directory = FIXTURES / "acris"
    spec = load_concept_source_spec(directory / "acris.spec.yaml")
    path = tmp_path_factory.mktemp("acris_ids") / "staging.duckdb"
    with Staging(str(path), schemas=()) as staging:
        land_tabular(directory, staging, schema="land_acris")
        provider = RelationProvider(warehouse=None, staging=staging)
        yield import_concept_source(spec, provider, dataset_map={})


def test_acris_import_normalizes_with_its_declared_joins(
    acris: Any, tmp_path: Path
) -> None:
    normalized = _normalize(acris, tmp_path / "acris.manifest.yaml")

    names = {e.ref.qualified_name for e in normalized.entities}
    assert len(names) == 15
    assert "acris.acris_real_property_master" in names
    assert "acris.acris_country_codes" in names
    joins = [j for e in normalized.entities for j in e.declared_joins]
    assert len(joins) == 8
    assert {j.referenced_entity for j in joins} == {
        "acris.acris_real_property_master",
        "acris.acris_personal_property_master",
    }
    assert {c.card_version for c in normalized.context_cards} == {"2026.10.20"}


def test_acris_cards_keep_the_source_names(acris: Any) -> None:
    cards = {e["qualified_name"]: e["context_card"] for e in acris.manifest["entities"]}

    assert "ACRIS - Real Property Master" in (
        cards["acris.acris_real_property_master"]["curated_synonyms"]
    )
    assert f"ACRIS {EN_DASH} Country Codes" in (
        cards["acris.acris_country_codes"]["curated_synonyms"]
    )


def test_semsql_import_normalizes(tmp_path: Path) -> None:
    from sema.binding.importer import import_concept_source
    from sema.binding.spec import load_concept_source_spec
    from sema.intake.loaders.semsql import land_semsql
    from sema.intake.relations import RelationProvider

    directory = FIXTURES / "semsql"
    with Staging(str(tmp_path / "staging.duckdb"), schemas=()) as staging:
        land_semsql(directory / "ontology.db", staging, schema="land_a3")
        result = import_concept_source(
            load_concept_source_spec(directory / "ontology.spec.yaml"),
            RelationProvider(warehouse=None, staging=staging),
            dataset_map={},
            staging=staging,
        )

    normalized = _normalize(result, tmp_path / "ontology.manifest.yaml")

    [scheme] = normalized.entities
    assert scheme.ref.qualified_name == "ex.ontology"
    assert [c.card_version for c in normalized.context_cards] == ["2026.10.01"]

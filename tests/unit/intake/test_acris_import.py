"""A10: the ACRIS directory imported with its hand-written per-file spec."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jsonschema
import pytest
import yaml

from sema.ingest.duckdb_staging import Staging
from sema.targets.adapters.manifest_parser import parse_manifest_raw

pytestmark = pytest.mark.unit

FIXTURES = Path(__file__).parents[2] / "fixtures" / "intake"
ACRIS = FIXTURES / "acris"
SPEC = ACRIS / "acris.spec.yaml"
RUN = ACRIS / "acris.run.yaml"
QUALIFIED_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*(\.[A-Za-z0-9_]+)+$")
JSON_SCHEMA = json.loads((FIXTURES / "frictionless" / "table-schema.json").read_text())
SCHEMA = "land_acris"
ENTITIES = 15
PROPERTIES = 125
TERMS = {
    "Country_Codes": 250,
    "Document_Control_Codes": 126,
    "Property_Types_Codes": 49,
    "State_Codes": 54,
    "UCC_Collateral_Codes": 8,
}
JOIN_COLUMNS = ["Document ID", "Good Through Date"]
REAL_MASTER = "acris_real_property_master"
PERSONAL_MASTER = "acris_personal_property_master"
FOREIGN_KEYS = {
    ("acris_real_property_legals", REAL_MASTER),
    ("acris_real_property_parties", REAL_MASTER),
    ("acris_real_property_remarks", REAL_MASTER),
    ("acris_real_property_references", REAL_MASTER),
    ("acris_personal_property_legals", PERSONAL_MASTER),
    ("acris_personal_property_parties", PERSONAL_MASTER),
    ("acris_personal_property_remarks", PERSONAL_MASTER),
    ("acris_personal_property_references", PERSONAL_MASTER),
}
DOMAINS = {
    ("acris_real_property_legals", "Property Type", "Property_Types_Codes"),
    ("acris_personal_property_legals", "Property Type", "Property_Types_Codes"),
    ("acris_real_property_master", "Docoument Type", "Document_Control_Codes"),
    ("acris_personal_property_master", "Docoument Type", "Document_Control_Codes"),
    ("acris_personal_property_master", "UCC Collateral", "UCC_Collateral_Codes"),
    ("acris_real_property_parties", "Country", "Country_Codes"),
    ("acris_personal_property_parties", "Country", "Country_Codes"),
    ("acris_real_property_parties", "State", "State_Codes"),
    ("acris_personal_property_parties", "State", "State_Codes"),
}
BOROUGHS = {"1": "Manhattan", "2": "Bronx", "3": "Brooklyn", "4": "Queens"}
LOTS = {"P": "partial", "E": "entire"}
INLINE_VALUES = {
    (REAL_MASTER, "Recorded Borough"): BOROUGHS,
    ("acris_real_property_legals", "Borough"): BOROUGHS,
    ("acris_real_property_legals", "Partial Lot"): LOTS,
    (PERSONAL_MASTER, "Recorded Borough"): BOROUGHS,
    ("acris_personal_property_legals", "Borough"): BOROUGHS,
    ("acris_personal_property_legals", "Partial Lot"): LOTS,
}
LEGALS_DESCRIPTION = (
    "The properties under which real property documents recorded in ACRIS are indexed."
)


@dataclass(frozen=True)
class _Imported:
    spec: Any
    landed: Any
    result: Any


@pytest.fixture(scope="module")
def imported(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Imported]:
    from sema.binding.importer import import_concept_source
    from sema.binding.spec import load_concept_source_spec
    from sema.intake.loaders.tabular import land_tabular
    from sema.intake.relations import RelationProvider

    spec = load_concept_source_spec(SPEC)
    path = tmp_path_factory.mktemp("acris_import") / "staging.duckdb"
    with Staging(str(path), schemas=()) as staging:
        landed = land_tabular(ACRIS, staging, schema=SCHEMA)
        provider = RelationProvider(warehouse=None, staging=staging)
        result = import_concept_source(spec, provider, dataset_map={})
        yield _Imported(spec=spec, landed=landed, result=result)


def _table_rows(result: Any) -> list[dict[str, Any]]:
    return [e for e in result.manifest["entities"] if e["kind"] == "TABLE_ROW"]


def _dataset_names(landed: Any) -> set[str]:
    headers = [r for r in landed.relations if r.name.endswith("__Dataset_Info")]
    return {r.columns[1][0] for r in headers}


def test_spec_matches_the_run_descriptor(imported: _Imported) -> None:
    run = yaml.safe_load(RUN.read_text())
    spec = imported.spec

    assert spec.spec_version == 2
    assert (spec.id, spec.identity_system, spec.release) == (
        run["id"],
        run["identity_system"],
        str(run["release"]),
    )
    assert spec.term_storage.inline_cap == run["term_storage"]["inline_cap"]
    assert spec.external_namespace == run["term_storage"]["external_namespace"]
    assert spec.entity_model.entity_case == run["entity_case"]


def test_spec_is_per_file_over_staging_with_headers_and_notes(imported: _Imported) -> None:
    spec = imported.spec
    model = spec.entity_model

    assert spec.default_provider == "staging"
    assert {ref.provider for ref in spec.relation_refs()} == {"staging"}
    assert model.layout == "per_file"
    assert model.header_keys is None
    assert model.roles.type is None
    assert len(model.roles.notes) == 2
    assert len(model.entities) == ENTITIES
    for entry in model.entities:
        assert entry.relation.name.endswith("__Column_Info")
        assert entry.header is not None
        assert entry.header.name.endswith("__Dataset_Info")
        stem = entry.relation.name.removesuffix("__Column_Info")
        assert entry.header.name.removesuffix("__Dataset_Info") == stem


def test_spec_declares_five_code_lists_and_the_shape_entry(imported: _Imported) -> None:
    spec = imported.spec

    assert {entry.vocabulary_id for entry in spec.code_lists} == set(TERMS)
    assert len(spec.code_lists) == len(TERMS)
    assert {entry.relation.name for entry in spec.code_lists} == {
        f"{SCHEMA}.{name}" for name in TERMS
    }
    assert spec.concepts is None
    shape = {(entry.section, entry.template) for entry in spec.shape or []}
    assert ("code_lists.Document_Control_Codes", "concept_list") in shape


def test_fifteen_entities_named_from_their_headers(imported: _Imported) -> None:
    entities = _table_rows(imported.result)

    assert len(entities) == ENTITIES
    names = {e["qualified_name"] for e in entities}
    assert len(names) == ENTITIES
    assert all(QUALIFIED_NAME.match(name) for name in names)
    labels = {label for e in entities for label in e["context_card"]["curated_synonyms"]}
    assert labels == _dataset_names(imported.landed)
    kinds = {e["kind"] for e in imported.result.manifest["entities"]}
    assert kinds == {"TABLE_ROW"}


def test_entities_hold_125_properties(imported: _Imported) -> None:
    entities = _table_rows(imported.result)

    assert sum(len(e["properties"]) for e in entities) == PROPERTIES


def test_header_description_reaches_the_context_card(imported: _Imported) -> None:
    entities = {e["qualified_name"]: e for e in _table_rows(imported.result)}

    card = entities["acris.acris_real_property_legals"]["context_card"]
    assert card["description"] == LEGALS_DESCRIPTION
    assert card["curated_synonyms"] == ["ACRIS - Real Property Legals"]


def test_five_vocabularies_with_their_term_counts(imported: _Imported) -> None:
    manifest = imported.result.manifest

    vocabularies = {v["name"]: v["source"] for v in manifest["vocabularies"]}
    assert vocabularies == {name: "INLINE" for name in TERMS}
    assert Counter(t["vocabulary"] for t in manifest["terms"]) == TERMS


def test_no_entity_has_an_obligation(imported: _Imported) -> None:
    entities = _table_rows(imported.result)

    assert [e["qualified_name"] for e in entities if "obligation" in e] == []
    assert len(entities) == ENTITIES


def test_eight_declared_foreign_keys_across_the_table_schemas(imported: _Imported) -> None:
    schemas = imported.result.table_schemas

    found = [
        (name.split(".", 1)[1], fk.reference.resource, fk.fields, fk.reference.fields)
        for name, schema in schemas.items()
        for fk in schema.foreign_keys or []
    ]
    assert len(found) == 8
    assert {(owner, target) for owner, target, _, _ in found} == FOREIGN_KEYS
    assert all(local == remote == JOIN_COLUMNS for _, _, local, remote in found)


def test_nine_domains_across_the_table_schemas(imported: _Imported) -> None:
    schemas = imported.result.table_schemas

    found = {
        (name.split(".", 1)[1], field.name, field.x_sema_domain)
        for name, schema in schemas.items()
        for field in schema.fields
        if field.x_sema_domain
    }
    assert found == DOMAINS


def test_table_schemas_are_untyped_and_validate(imported: _Imported) -> None:
    schemas = imported.result.table_schemas

    assert len(schemas) == ENTITIES
    for schema in schemas.values():
        assert {f.type for f in schema.fields} == {"any"}
        assert {f.x_sema_raw_type for f in schema.fields} == {None}
        jsonschema.validate(schema.descriptor(), JSON_SCHEMA)


def test_manifest_parses(imported: _Imported) -> None:
    parsed = parse_manifest_raw(yaml.safe_load(yaml.safe_dump(imported.result.manifest)))

    assert len(parsed.entities) == ENTITIES
    assert len(parsed.vocabularies) == len(TERMS)
    assert all(entity.obligation is None for entity in parsed.entities)


def test_six_fields_carry_inline_values(imported: _Imported) -> None:
    schemas = imported.result.table_schemas

    found = {
        (name.split(".", 1)[1], field.name): field.x_sema_values
        for name, schema in schemas.items()
        for field in schema.fields
        if field.x_sema_values
    }
    assert len(found) == 6
    assert found == INLINE_VALUES
    assert all(list(found[key]) == list(INLINE_VALUES[key]) for key in found)


def test_schemas_with_inline_values_record_the_extractor_version(
    imported: _Imported,
) -> None:
    from sema.intake.lexicon import PATTERNS_VERSION

    schemas = imported.result.table_schemas
    owners = {f"acris.{owner}" for owner, _ in INLINE_VALUES}

    for name in owners:
        descriptor = schemas[name].descriptor()
        assert descriptor["x-sema-values-extractor"] == PATTERNS_VERSION
        jsonschema.validate(descriptor, JSON_SCHEMA)


def test_the_same_six_manifest_properties_carry_decoded_values(
    imported: _Imported,
) -> None:
    parsed = parse_manifest_raw(yaml.safe_load(yaml.safe_dump(imported.result.manifest)))

    found = {
        (entity.qualified_name.split(".", 1)[1], prop.name): prop.decoded_values
        for entity in parsed.entities
        for prop in entity.properties
        if prop.decoded_values
    }
    assert found == INLINE_VALUES

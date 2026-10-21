"""A10: the ACRIS directory imported with its hand-written per-file spec."""

from __future__ import annotations

import json
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
REAL_MASTER = "ACRIS - Real Property Master"
PERSONAL_MASTER = "ACRIS - Personal Property Master"
FOREIGN_KEYS = {
    ("ACRIS - Real Property Legals", REAL_MASTER),
    ("ACRIS - Real Property Parties", REAL_MASTER),
    ("ACRIS - Real Property Remarks", REAL_MASTER),
    ("ACRIS - Real Property References", REAL_MASTER),
    ("ACRIS - Personal Property Legals", PERSONAL_MASTER),
    ("ACRIS - Personal Property Parties", PERSONAL_MASTER),
    ("ACRIS - Personal Property Remarks", PERSONAL_MASTER),
    ("ACRIS - Personal Property References", PERSONAL_MASTER),
}
DOMAINS = {
    ("ACRIS - Real Property Legals", "Property Type", "Property_Types_Codes"),
    ("ACRIS - Personal Property Legals", "Property Type", "Property_Types_Codes"),
    ("ACRIS - Real Property Master", "Docoument Type", "Document_Control_Codes"),
    ("ACRIS - Personal Property Master", "Docoument Type", "Document_Control_Codes"),
    ("ACRIS - Personal Property Master", "UCC Collateral", "UCC_Collateral_Codes"),
    ("ACRIS - Real Property Parties", "Country", "Country_Codes"),
    ("ACRIS - Personal Property Parties", "Country", "Country_Codes"),
    ("ACRIS - Real Property Parties", "State", "State_Codes"),
    ("ACRIS - Personal Property Parties", "State", "State_Codes"),
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
    assert names == {f"acris.{name}" for name in _dataset_names(imported.landed)}
    kinds = {e["kind"] for e in imported.result.manifest["entities"]}
    assert kinds == {"TABLE_ROW"}


def test_entities_hold_125_properties(imported: _Imported) -> None:
    entities = _table_rows(imported.result)

    assert sum(len(e["properties"]) for e in entities) == PROPERTIES


def test_header_description_reaches_the_context_card(imported: _Imported) -> None:
    entities = {e["qualified_name"]: e for e in _table_rows(imported.result)}

    card = entities["acris.ACRIS - Real Property Legals"]["context_card"]
    assert card["description"] == LEGALS_DESCRIPTION


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

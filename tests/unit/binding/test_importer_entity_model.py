"""Importer over an entity_model section: a landed dictionary becomes entities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from sema.binding.spec import ConceptSourceSpec
from sema.ingest.duckdb_staging import Staging
from sema.targets.adapters.manifest_parser import parse_manifest_raw

pytestmark = pytest.mark.unit

DICTIONARY = """\
table_name,column_name,datatype,required,pk,fk_table,fk_column,domain,guidance
VISIT,VISIT_ID,integer,Yes,Yes,,,,Visit key
VISIT,PERSON_ID,integer,Yes,No,PERSON,PERSON_ID,,
VISIT,VISIT_TYPE_ID,integer,No,No,TYPE,TYPE_ID,Visit Type,
PERSON,PERSON_ID,integer,Yes,Yes,,,,
PERSON,BIRTH_YEAR,integer,No,No,,,,
"""

ENTITY_MODEL = {
    "relation": "land.dictionary",
    "entity_case": "lower",
    "roles": {
        "entity": "table_name",
        "property": "column_name",
        "type": "datatype",
        "required": "required",
        "primary_key": "pk",
        "fk_entity": "fk_table",
        "fk_property": "fk_column",
        "domain": "domain",
        "description": "guidance",
    },
}


def _spec(**sections: Any) -> ConceptSourceSpec:
    raw: dict[str, Any] = {
        "spec_version": 2,
        "id": "ex-model",
        "identity_system": "ex",
        "release": "1",
        "default_provider": "staging",
        "entity_model": ENTITY_MODEL,
    }
    raw.update(sections)
    return ConceptSourceSpec.model_validate(raw)


def _import(tmp_path: Path, spec: ConceptSourceSpec) -> Any:
    from sema.binding.importer import import_concept_source
    from sema.intake.loaders.tabular import land_tabular
    from sema.intake.relations import RelationProvider

    path = tmp_path / "dictionary.csv"
    path.write_text(DICTIONARY)
    with Staging(str(tmp_path / "staging.duckdb"), schemas=()) as staging:
        land_tabular(path, staging, schema="land")
        staging.execute("CREATE TABLE land.codes AS SELECT 'A' AS code, 'Alpha' AS label")
        return import_concept_source(
            spec, RelationProvider(warehouse=None, staging=staging), dataset_map={}
        )


def test_entity_model_only_produces_entities_and_no_scheme(tmp_path: Path) -> None:
    result = _import(tmp_path, _spec())

    parsed = parse_manifest_raw(yaml.safe_load(yaml.safe_dump(result.manifest)))
    names = [e.qualified_name for e in parsed.entities]
    assert names == ["ex.person", "ex.visit"]
    assert {e.kind.value for e in parsed.entities} == {"TABLE_ROW"}
    visit = next(e for e in parsed.entities if e.qualified_name == "ex.visit")
    assert visit.obligation is not None
    assert visit.obligation.required_fields == ["person_id", "visit_id"]
    assert [(fk.referenced_entity, fk.join_keys) for fk in visit.obligation.foreign_keys] == [
        ("ex.person", [("person_id", "person_id")]),
        ("ex.type", [("visit_type_id", "type_id")]),
    ]
    assert [(d.property_name, d.domain_id) for d in visit.obligation.domain_constraints] == [
        ("visit_type_id", "Visit Type")
    ]
    props = {p.name: p for p in visit.properties}
    assert props["visit_id"].definition == "Visit key"
    assert props["visit_type_id"].nullable is True


def test_spec_with_concepts_and_entity_model_produces_both(tmp_path: Path) -> None:
    spec = _spec(
        scheme={"qualified_name": "ex.codes", "description": "Codes"},
        concepts={
            "relation": "land.codes",
            "group_by": "code",
            "roles": {"concept_id": "code", "name": "label"},
        },
    )

    result = _import(tmp_path, spec)

    kinds = [(e["qualified_name"], e["kind"]) for e in result.manifest["entities"]]
    assert kinds == [
        ("ex.codes", "CONCEPT_SCHEME"),
        ("ex.person", "TABLE_ROW"),
        ("ex.visit", "TABLE_ROW"),
    ]

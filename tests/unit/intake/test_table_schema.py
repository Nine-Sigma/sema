"""Table Schema descriptors from dictionary rows, and their projection to entities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema
import pytest

from sema.targets.adapters.manifest_models import ManifestEntity
from tests.unit.binding.fixtures.concept_library_v2 import captured_log

pytestmark = pytest.mark.unit

JSON_SCHEMA = json.loads(
    (
        Path(__file__).parents[2] / "fixtures" / "intake" / "frictionless" / "table-schema.json"
    ).read_text()
)


def _row(name: str, raw: str = "integer", **extra: Any) -> Any:
    from sema.intake.table_schema_utils import DictionaryRow

    return DictionaryRow(property=name, raw_type=raw, **extra)


def _schema(entity: str, rows: list[Any]) -> Any:
    from sema.intake.table_schema_utils import table_schema_from_rows

    return table_schema_from_rows(entity, rows)


def _project(schema: Any, name: str = "ex.visit") -> ManifestEntity:
    from sema.intake.table_schema import to_manifest_entity

    return ManifestEntity.model_validate(to_manifest_entity(schema, name, namespace="ex"))


def _visit_rows() -> list[Any]:
    return [
        _row("visit_id", required=True, primary_key=True, description="Visit key"),
        _row("person_id", required=True, fk_entity="person", fk_property="person_id"),
        _row("visit_type", "varchar(50)", domain="Visit Type"),
        _row("visit_date", "date", required=True),
        _row("cost", "float"),
    ]


def test_descriptors_validate_against_the_vendored_json_schema() -> None:
    schemas = [
        _schema("visit", _visit_rows()),
        _schema("pair", [_row("a", required=True, primary_key=True), _row("b", "text")]),
        _schema("odd", [_row("blob", "geometry")]),
    ]

    for schema in schemas:
        jsonschema.validate(schema.descriptor(), JSON_SCHEMA)


def test_varchar_maps_to_string_and_keeps_the_raw_type() -> None:
    descriptor = _schema("visit", _visit_rows()).descriptor()

    field = next(f for f in descriptor["fields"] if f["name"] == "visit_type")
    assert field["type"] == "string"
    assert field["x-sema-raw-type"] == "varchar(50)"
    assert field["x-sema-domain"] == "Visit Type"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Integer", "integer"),
        ("bigint", "integer"),
        ("float", "number"),
        ("numeric(10,2)", "number"),
        ("varchar(MAX)", "string"),
        ("text", "string"),
        ("date", "date"),
        ("datetime", "datetime"),
        ("timestamp", "datetime"),
        ("boolean", "boolean"),
    ],
)
def test_raw_types_map_deterministically(raw: str, expected: str) -> None:
    from sema.intake.table_schema_utils import frictionless_type

    assert frictionless_type(raw, "t.c") == expected


def test_unknown_raw_type_maps_to_any_and_logs_the_column() -> None:
    with captured_log() as messages:
        descriptor = _schema("odd", [_row("blob", "geometry")]).descriptor()

    assert descriptor["fields"][0]["type"] == "any"
    assert [m for m in messages if "odd.blob" in m and "geometry" in m]


def test_projection_natural_key_foreign_key_and_required_fields() -> None:
    entity = _project(_schema("visit", _visit_rows()))

    assert entity.kind.value == "TABLE_ROW"
    props = {p.name: p for p in entity.properties}
    assert [p.name for p in entity.properties] == [
        "visit_id",
        "person_id",
        "visit_type",
        "visit_date",
        "cost",
    ]
    assert props["visit_id"].type == "integer"
    assert props["visit_id"].nullable is False
    assert props["visit_type"].nullable is True
    assert props["visit_id"].definition == "Visit key"
    obligation = entity.obligation
    assert obligation is not None
    assert obligation.required_fields == ["visit_id", "person_id", "visit_date"]
    assert obligation.primary_key.value == "NATURAL_KEY"
    assert [(fk.referenced_entity, fk.join_keys) for fk in obligation.foreign_keys] == [
        ("ex.person", [("person_id", "person_id")])
    ]


def test_projection_compound_key() -> None:
    rows = [
        _row("a", required=True, primary_key=True),
        _row("b", required=True, primary_key=True),
    ]

    entity = _project(_schema("pair", rows), "ex.pair")

    assert entity.obligation is not None
    assert entity.obligation.primary_key.value == "COMPOUND"


def test_entity_description_goes_to_the_context_card() -> None:
    from sema.intake.table_schema_utils import table_schema_from_rows

    schema = table_schema_from_rows("visit", _visit_rows(), description="A visit.")
    entity = _project(schema)

    assert entity.context_card is not None
    assert entity.context_card.description == "A visit."


def test_no_obligation_when_nothing_is_required() -> None:
    with captured_log() as messages:
        entity = _project(_schema("note", [_row("text", "text")]), "ex.note")

    assert entity.obligation is None
    assert [m for m in messages if "ex.note" in m and "no required field" in m]


def test_no_obligation_when_no_primary_key_is_stated() -> None:
    rows = [_row("a", required=True, fk_entity="person", fk_property="person_id")]

    with captured_log() as messages:
        entity = _project(_schema("link", rows), "ex.link")

    assert entity.obligation is None
    assert entity.properties[0].nullable is False
    assert [m for m in messages if "ex.link" in m and "no primary key" in m]


def test_domain_becomes_a_constraint_and_never_a_vocabulary_binding() -> None:
    rows = [
        _row("visit_id", required=True, primary_key=True),
        _row("visit_type_id", required=True, domain="Visit"),
    ]

    entity = _project(_schema("visit", rows))

    assert entity.obligation is not None
    assert [(c.property_name, c.domain_id) for c in entity.obligation.domain_constraints] == [
        ("visit_type_id", "Visit")
    ]
    assert all(p.vocabulary_binding is None for p in entity.properties)
    assert all(not p.synonyms for p in entity.properties)


# Inline code lists: x-sema-values on a field, x-sema-values-extractor on the schema.

BOROUGH_VALUES = {"1": "Manhattan", "2": "Bronx", "3": "Brooklyn", "4": "Queens"}


def _valued_schema() -> Any:
    from sema.intake.lexicon import PATTERNS_VERSION
    from sema.intake.table_schema import Field, TableSchema

    return TableSchema.model_validate(
        {
            "fields": [
                Field(name="borough", type="any", x_sema_values=BOROUGH_VALUES),
                Field(name="lot", type="any"),
            ],
            "x-sema-values-extractor": PATTERNS_VERSION,
        }
    )


def test_inline_values_serialize_under_their_extension_keys_and_validate() -> None:
    from sema.intake.lexicon import PATTERNS_VERSION

    descriptor = _valued_schema().descriptor()

    jsonschema.validate(descriptor, JSON_SCHEMA)
    assert descriptor["x-sema-values-extractor"] == PATTERNS_VERSION
    borough, lot = descriptor["fields"]
    assert borough["x-sema-values"] == BOROUGH_VALUES
    assert list(borough["x-sema-values"]) == ["1", "2", "3", "4"]
    assert "x-sema-values" not in lot


def test_inline_values_parse_back_from_the_descriptor() -> None:
    from sema.intake.table_schema import TableSchema

    schema = _valued_schema()

    assert TableSchema.model_validate(schema.descriptor()) == schema
    assert schema.fields[0].x_sema_values == BOROUGH_VALUES


def test_inline_values_project_to_decoded_values() -> None:
    entity = _project(_valued_schema(), "ex.legals")

    props = {p.name: p for p in entity.properties}
    assert props["borough"].decoded_values == BOROUGH_VALUES
    assert list(props["borough"].decoded_values) == ["1", "2", "3", "4"]
    assert props["lot"].decoded_values == {}

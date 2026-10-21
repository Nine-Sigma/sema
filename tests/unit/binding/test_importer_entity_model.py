"""Importer over an entity_model section: a landed dictionary becomes entities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from sema.binding.spec import ConceptSourceSpec
from sema.ingest.duckdb_staging import Staging
from sema.intake.landing import LOCATOR_COLUMN
from sema.targets.adapters.manifest_parser import parse_manifest_raw

pytestmark = pytest.mark.unit

DICTIONARY = """\
table_name,column_name,datatype,required,pk,fk_table,fk_column,domain,guidance,table_note
VISIT,VISIT_ID,integer,Yes,Yes,,,,Visit key,Visits to a care site
VISIT,PERSON_ID,integer,Yes,No,PERSON,PERSON_ID,,,
VISIT,VISIT_TYPE_ID,integer,No,No,TYPE,TYPE_ID,Visit Type,,
PERSON,PERSON_ID,integer,Yes,Yes,,,,,
PERSON,BIRTH_YEAR,integer,No,No,,,,,
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


def test_entity_description_role_reaches_the_context_card(tmp_path: Path) -> None:
    roles = {**ENTITY_MODEL["roles"], "entity_description": "table_note"}  # type: ignore[dict-item]
    result = _import(tmp_path, _spec(entity_model={**ENTITY_MODEL, "roles": roles}))

    cards = {e["qualified_name"]: e.get("context_card") for e in result.manifest["entities"]}
    assert cards["ex.visit"]["description"] == "Visits to a care site"
    assert cards["ex.person"] is None


def test_type_role_omitted_under_single_relation_gives_any_fields(tmp_path: Path) -> None:
    roles = {k: v for k, v in ENTITY_MODEL["roles"].items() if k != "type"}  # type: ignore[union-attr]

    result = _import(tmp_path, _spec(entity_model={**ENTITY_MODEL, "roles": roles}))

    for entity in result.manifest["entities"]:
        assert {p["type"] for p in entity["properties"]} == {"any"}


# Per-file layout: one dictionary relation per entity, plus a key-value header.

PROMPT_KEY = "How do the rows of this table connect to other tables in the system?"
JOIN_PROSE = "Each order links back to the customers table by customer id."
ORDERS_COLUMNS = (
    ("Order ID", "Order key", None),
    ("Customer ID", "Customer who placed the order", "see the customers table"),
    ("Status", "Order state", None),
)
ORDERS_HEADER = (
    ("Table Description", "One row per order placed."),
    ("Refresh Cadence", None),
    (PROMPT_KEY, JOIN_PROSE),
)
CUSTOMERS_COLUMNS = (
    ("Customer ID", "Customer key", None),
    ("Customer Name", "Full name", None),
)


def _create(staging: Staging, name: str, columns: tuple[str, ...], rows: Any) -> None:
    """Landed like the tabular loader: a trailing locator, data from sheet row 2."""
    quoted = ", ".join(f'"{column}" VARCHAR' for column in (*columns, LOCATOR_COLUMN))
    staging.execute(f'CREATE TABLE land."{name}" ({quoted})')
    for number, row in enumerate(rows, start=2):
        values = [*row, f"{name}.xlsx!{number}"]
        marks = ", ".join("?" for _ in values)
        staging.execute(f'INSERT INTO land."{name}" VALUES ({marks})', values)


def _per_file_staging(staging: Staging) -> None:
    staging.execute("CREATE SCHEMA IF NOT EXISTS land")
    dictionary = ("field", "meaning", "remarks")
    _create(staging, "orders__Columns", dictionary, ORDERS_COLUMNS)
    _create(staging, "orders__Header", ("Table Name", "Customer Orders"), ORDERS_HEADER)
    _create(staging, "customers__Columns", dictionary, CUSTOMERS_COLUMNS)
    _create(staging, "customers", dictionary, CUSTOMERS_COLUMNS)


def _orders_entry(**extra: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "relation": "land.orders__Columns",
        "header": "land.orders__Header",
    }
    entry.update(extra)
    return entry


def _per_file_spec(entries: list[dict[str, Any]], **section: Any) -> ConceptSourceSpec:
    body: dict[str, Any] = {
        "layout": "per_file",
        "entities": entries,
        "roles": {"property": "field", "description": "meaning", "notes": ["remarks"]},
    }
    body.update(section)
    return _spec(entity_model=body)


def _per_file_import(tmp_path: Path, spec: ConceptSourceSpec) -> Any:
    from sema.binding.importer import import_concept_source
    from sema.intake.relations import RelationProvider

    with Staging(str(tmp_path / "per_file.duckdb"), schemas=()) as staging:
        _per_file_staging(staging)
        return import_concept_source(
            spec, RelationProvider(warehouse=None, staging=staging), dataset_map={}
        )


def _read_header(tmp_path: Path, header_keys: Any = None, rows: Any = None) -> Any:
    from sema.binding.importer_entity_utils import read_header
    from sema.binding.spec import HeaderKeys, RelationRef
    from sema.intake.relations import RelationProvider

    keys = HeaderKeys.model_validate(header_keys) if header_keys is not None else None
    ref = RelationRef(provider="staging", name="land.orders__Header")
    with Staging(str(tmp_path / "header.duckdb"), schemas=()) as staging:
        staging.execute("CREATE SCHEMA IF NOT EXISTS land")
        columns, body = rows or (("Table Name", "Customer Orders"), ORDERS_HEADER)
        _create(staging, "orders__Header", columns, body)
        return read_header(RelationProvider(warehouse=None, staging=staging), ref, keys)


def _entities(result: Any) -> dict[str, Any]:
    return {e["qualified_name"]: e for e in result.manifest["entities"]}


def _table_schema(result: Any, qualified_name: str) -> Any:
    return result.table_schemas[qualified_name]


def test_header_column_names_are_the_first_pair(tmp_path: Path) -> None:
    header = _read_header(tmp_path)

    assert header.pairs[0] == ("Table Name", "Customer Orders")
    assert header.pairs[1:] == list(ORDERS_HEADER)


def test_header_key_with_no_value_is_kept_as_null(tmp_path: Path) -> None:
    header = _read_header(tmp_path)

    assert ("Refresh Cadence", None) in header.pairs
    assert header.prose["Refresh Cadence"] is None


def test_header_name_and_description_found_by_lexicon_tokens(tmp_path: Path) -> None:
    header = _read_header(tmp_path)

    assert header.name == "Customer Orders"
    assert header.description == "One row per order placed."


def test_header_prose_is_every_other_key_when_text_is_omitted(tmp_path: Path) -> None:
    header = _read_header(tmp_path)

    assert set(header.prose) == {"Refresh Cadence", PROMPT_KEY}
    assert header.prose[PROMPT_KEY] == JOIN_PROSE


def test_header_prose_is_only_the_listed_text_keys(tmp_path: Path) -> None:
    header = _read_header(tmp_path, {"text": [PROMPT_KEY]})

    assert header.prose == {PROMPT_KEY: JOIN_PROSE}


def test_header_keys_are_matched_after_normalization(tmp_path: Path) -> None:
    header = _read_header(
        tmp_path, {"name": "TABLE  name:", "description": "table descriptions"}
    )

    assert header.name == "Customer Orders"
    assert header.description == "One row per order placed."


def test_header_title_and_summary_are_header_tokens(tmp_path: Path) -> None:
    rows = (("Report Title", "Shipments"), (("Short Summary", "One row per shipment."),))

    header = _read_header(tmp_path, rows=rows)

    assert header.name == "Shipments"
    assert header.description == "One row per shipment."


def test_header_token_tie_takes_the_first_key_in_reading_order(tmp_path: Path) -> None:
    body = (
        ("Owner Name", "Sales team"),
        ("Table Description", "One row per order placed."),
        ("Detailed Description", "Orders are written at checkout."),
    )

    header = _read_header(tmp_path, rows=(("Table Name", "Customer Orders"), body))

    assert header.name == "Customer Orders"
    assert header.description == "One row per order placed."
    assert set(header.prose) == {"Owner Name", "Detailed Description"}


def test_per_file_entity_is_named_and_described_from_its_header(tmp_path: Path) -> None:
    result = _per_file_import(tmp_path, _per_file_spec([_orders_entry()]))

    entity = _entities(result)["ex.Customer Orders"]
    assert entity["kind"] == "TABLE_ROW"
    assert entity["context_card"]["description"] == "One row per order placed."
    props = {p["name"]: p for p in entity["properties"]}
    assert list(props) == ["Order ID", "Customer ID", "Status"]
    assert props["Customer ID"]["definition"] == "Customer who placed the order"


@pytest.mark.parametrize("relation", ["land.customers__Columns", "land.customers"])
def test_per_file_entity_without_header_is_named_after_the_file_stem(
    tmp_path: Path, relation: str
) -> None:
    result = _per_file_import(tmp_path, _per_file_spec([{"relation": relation}]))

    entity = _entities(result)["ex.customers"]
    assert [p["name"] for p in entity["properties"]] == ["Customer ID", "Customer Name"]
    assert "context_card" not in entity


def test_per_file_entry_entity_overrides_the_header_name(tmp_path: Path) -> None:
    result = _per_file_import(tmp_path, _per_file_spec([_orders_entry(entity="Orders")]))

    assert set(_entities(result)) == {"ex.Orders"}


def test_type_omitted_gives_any_fields_and_logs_a_count_per_entity(tmp_path: Path) -> None:
    import re

    from tests.unit.binding.fixtures.concept_library_v2 import captured_log

    spec = _per_file_spec([_orders_entry(), {"relation": "land.customers__Columns"}])
    with captured_log() as messages:
        result = _per_file_import(tmp_path, spec)

    for name in ("ex.Customer Orders", "ex.customers"):
        fields = _table_schema(result, name).fields
        assert {f.type for f in fields} == {"any"}
        assert {f.x_sema_raw_type for f in fields} == {None}
    assert any("Customer Orders" in m and re.search(r"\b3\b", m) for m in messages)
    assert any("customers" in m and re.search(r"\b2\b", m) for m in messages)


def test_declared_foreign_keys_and_domains_reach_the_table_schema(tmp_path: Path) -> None:
    from sema.intake.table_schema import ForeignKey, Reference

    entry = _orders_entry(
        foreign_keys=[
            {
                "columns": ["Customer ID"],
                "entity": "customers",
                "reference_columns": ["Customer ID"],
            }
        ],
        domains={"Status": "Order_Status"},
    )
    spec = _per_file_spec([entry, {"relation": "land.customers__Columns"}])

    result = _per_file_import(tmp_path, spec)

    schema = _table_schema(result, "ex.Customer Orders")
    reference = Reference(resource="customers", fields=["Customer ID"])
    assert schema.foreign_keys == [ForeignKey(fields=["Customer ID"], reference=reference)]
    domains = {f.name: f.x_sema_domain for f in schema.fields}
    assert domains == {"Order ID": None, "Customer ID": None, "Status": "Order_Status"}
    assert "obligation" not in _entities(result)["ex.Customer Orders"]


def test_table_schemas_cover_every_entity_and_validate(tmp_path: Path) -> None:
    import json

    import jsonschema

    schema_path = (
        Path(__file__).parents[2] / "fixtures" / "intake" / "frictionless" / "table-schema.json"
    )
    json_schema = json.loads(schema_path.read_text())
    spec = _per_file_spec([_orders_entry(), {"relation": "land.customers__Columns"}])

    result = _per_file_import(tmp_path, spec)

    assert len(result.table_schemas) == 2
    for schema in result.table_schemas.values():
        jsonschema.validate(schema.descriptor(), json_schema)


def test_single_relation_import_also_returns_table_schemas(tmp_path: Path) -> None:
    result = _import(tmp_path, _spec())

    visit = _table_schema(result, "ex.visit")
    assert {f.name for f in visit.fields} == {"person_id", "visit_id", "visit_type_id"}
    assert visit.primary_key == ["visit_id"]
    assert set(result.table_schemas) == {"ex.person", "ex.visit"}


def test_per_file_manifest_parses(tmp_path: Path) -> None:
    spec = _per_file_spec([_orders_entry(), {"relation": "land.customers__Columns"}])

    result = _per_file_import(tmp_path, spec)

    parsed = parse_manifest_raw(yaml.safe_load(yaml.safe_dump(result.manifest)))
    names = sorted(e.qualified_name for e in parsed.entities)
    assert names == ["ex.Customer Orders", "ex.customers"]

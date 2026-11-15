"""Shared setup for test_importer_entity_model.py and the tests that reuse it."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sema.binding.spec import ConceptSourceSpec
from sema.ingest.duckdb_staging import Staging
from sema.intake.land.landing import LOCATOR_COLUMN

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
    from sema.intake.land.relations import RelationProvider

    with Staging(str(tmp_path / "per_file.duckdb"), schemas=()) as staging:
        _per_file_staging(staging)
        return import_concept_source(
            spec, RelationProvider(warehouse=None, staging=staging), dataset_map={}
        )

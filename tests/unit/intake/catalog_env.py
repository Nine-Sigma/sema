"""A synthetic catalog for the graph tests: items, their coded values, and a status column.

The same rows land two ways: as a workbook (``write_catalog``) and as a warehouse schema
behind a fake connector (``warehouse_catalog``). The workbook can carry a header-less numeric
sheet, which the layout proposal is asked about before Roles runs, so a later re-entry has a
recorded call to replay. Status values use the lexicon's status words, so a settled status
column assembles only once its values are read.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import duckdb
import openpyxl
import yaml

from sema.models.schema_metadata import ColumnMetadata, ObjectMetadata, SchemaMetadata
from tests.unit.binding.fixtures.concept_library import query_fn
from tests.unit.intake.warehouse_env import CATALOG, SCHEMA, FakeConnector

STEM = "catalog"
ITEMS, VALUES, READINGS = "Items", "Values", "Readings"
ITEMS_TABLE, VALUES_TABLE = "items", "item_values"
GROUPS_TABLE, GROUP_VALUES_TABLE = "groups", "group_values"
STATUS = "status"
ACTIVE, RETIRED = ("active",), ("deprecated", "retired")
WAREHOUSE = f"{CATALOG}.{SCHEMA}"

ITEM_ROWS: tuple[tuple[Any, ...], ...] = (
    ("concept_code", "concept_name", "definition"),
    ("KIND", "Kind", "Kind of a thing"),
    ("TONE", "Tone", "Tone of a thing"),
)
VALUE_ROWS: tuple[tuple[Any, ...], ...] = (
    ("concept_code", "code", "label", STATUS),
    ("KIND", "K1", "First", "active"),
    ("KIND", "K2", "Second", "active"),
    ("KIND", "K0", "Old", "retired"),
    ("TONE", "T1", "Light", "active"),
    ("TONE", "T2", "Dark", "deprecated"),
)
GROUP_ROWS: tuple[tuple[Any, ...], ...] = (
    ("group_code", "group_name", "definition"),
    ("SIZE", "Size", "Size of a thing"),
    ("FORM", "Form", "Form of a thing"),
)
GROUP_VALUE_ROWS: tuple[tuple[Any, ...], ...] = (
    ("group_code", "code", "label"),
    ("SIZE", "S1", "Small"),
    ("SIZE", "S2", "Large"),
    ("FORM", "F1", "Round"),
)
DENORMALIZED_HEADER = ("group_key", "group_label", "group_description", "item_code", "item_label")
COLOUR_ROWS: tuple[tuple[Any, ...], ...] = (
    DENORMALIZED_HEADER,
    ("G1", "Colour", "The colour a surface shows", "C01", "Crimson"),
    ("G1", "Colour", "The colour a surface shows", "C02", "Teal"),
    ("G2", "Size", "The extent of an object", "S01", "Petite"),
    ("G2", "Size", "The extent of an object", "S02", "Grand"),
)
SHAPE_ROWS: tuple[tuple[Any, ...], ...] = (
    DENORMALIZED_HEADER,
    ("H1", "Shape", "The outline of an object", "R01", "Round"),
    ("H1", "Shape", "The outline of an object", "R02", "Square"),
    ("H2", "Tone", "The lightness of a colour", "T01", "Light"),
    ("H2", "Tone", "The lightness of a colour", "T02", "Dark"),
)
FLAT_HEADER = ("code", "name", "definition")
FIRST_FLAT_ROWS = (FLAT_HEADER, ("A1", "Alpha", "The first"), ("B2", "Beta", "The second"))
SECOND_FLAT_ROWS = (FLAT_HEADER, ("C3", "Gamma", "The third"), ("D4", "Delta", "The fourth"))
COLOURS_TABLE, SHAPES_TABLE = "colours", "shapes"
# Numbers only: no row passes as a header, so Structure ignores it as ``no_header``.
READING_ROWS: tuple[tuple[Any, ...], ...] = ((1, 0.5), (2, 0.25), (3, 0.125))

DESCRIPTOR = {
    "id": STEM,
    "identity_system": STEM,
    "release": "1",
    "scheme": {"qualified_name": f"{STEM}.codes", "description": "A synthetic catalog"},
}


def without_status(rows: Sequence[tuple[Any, ...]]) -> tuple[tuple[Any, ...], ...]:
    return tuple(row[:-1] for row in rows)


def write_catalog(
    directory: Path, *, status: bool = True, readings: bool = True
) -> tuple[Path, Path]:
    """The workbook and its run descriptor; ``status=False`` drops the status column."""
    directory.mkdir(parents=True, exist_ok=True)
    workbook = openpyxl.Workbook()
    items = workbook.active
    assert items is not None
    items.title = ITEMS
    sheets = [
        (VALUES, VALUE_ROWS if status else without_status(VALUE_ROWS)),
        *([(READINGS, READING_ROWS)] if readings else []),
    ]
    for row in ITEM_ROWS:
        items.append(list(row))
    for title, rows in sheets:
        sheet = workbook.create_sheet(title)
        for row in rows:
            sheet.append(list(row))
    source = directory / f"{STEM}.xlsx"
    workbook.save(source)
    return source, write_descriptor(directory)


def write_descriptor(directory: Path, **fields: Any) -> Path:
    """The catalog's run descriptor, with ``fields`` stated beside its own."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{STEM}.run.yaml"
    path.write_text(yaml.safe_dump(DESCRIPTOR | fields, sort_keys=False), encoding="utf-8")
    return path


def values_relation(name: str) -> bool:
    """True for the landed values relation, whatever schema it landed in."""
    return name.rpartition(".")[2] == f"{STEM}__{VALUES}"


def status_role(spec: dict[str, Any] | None) -> dict[str, Any] | None:
    assert spec is not None, "Review presented no spec"
    found: dict[str, Any] | None = spec["value_sets"]["roles"][STATUS]
    return found


def status_classes(role: dict[str, Any]) -> tuple[set[str], set[str]]:
    return set(role["active_values"]), set(role["retired_values"])


def warehouse_catalog(
    *, status: bool = True
) -> tuple[duckdb.DuckDBPyConnection, SchemaMetadata]:
    """The catalog's rows in an in-memory warehouse, and the metadata its connector reports;
    ``status=False`` drops the status column, so Roles asks for no value read."""
    values = VALUE_ROWS if status else without_status(VALUE_ROWS)
    return warehouse_tables({ITEMS_TABLE: ITEM_ROWS, VALUES_TABLE: values})


def two_catalogs() -> tuple[duckdb.DuckDBPyConnection, SchemaMetadata]:
    """Two independent concept and value set pairs, each binding the same spec sections."""
    return warehouse_tables({
        ITEMS_TABLE: ITEM_ROWS,
        VALUES_TABLE: without_status(VALUE_ROWS),
        GROUPS_TABLE: GROUP_ROWS,
        GROUP_VALUES_TABLE: GROUP_VALUE_ROWS,
    })


def two_grouped_tables() -> tuple[duckdb.DuckDBPyConnection, SchemaMetadata]:
    """Two tables, each one group-by-key concept list denormalized over its value rows."""
    return warehouse_tables({COLOURS_TABLE: COLOUR_ROWS, SHAPES_TABLE: SHAPE_ROWS})


def two_flat_tables() -> tuple[duckdb.DuckDBPyConnection, SchemaMetadata]:
    return warehouse_tables({COLOURS_TABLE: FIRST_FLAT_ROWS, SHAPES_TABLE: SECOND_FLAT_ROWS})


def warehouse_tables(
    tables: Mapping[str, Sequence[tuple[Any, ...]]],
) -> tuple[duckdb.DuckDBPyConnection, SchemaMetadata]:
    """Each table's header row and body in an in-memory warehouse, and its metadata."""
    con = duckdb.connect()
    con.execute(f"CREATE SCHEMA {SCHEMA}")
    objects = []
    for table, rows in tables.items():
        header, *body = rows
        columns = ", ".join(f"{name} VARCHAR" for name in header)
        con.execute(f"CREATE TABLE {SCHEMA}.{table} ({columns})")
        marks = ", ".join("?" for _ in header)
        con.executemany(f"INSERT INTO {SCHEMA}.{table} VALUES ({marks})", body)
        objects.append(_object(table, header))
    return con, SchemaMetadata(catalog=CATALOG, schema_name=SCHEMA, objects=tuple(objects))


def _object(table: str, header: Sequence[str]) -> ObjectMetadata:
    return ObjectMetadata(
        catalog=CATALOG,
        schema_name=SCHEMA,
        name=table,
        object_type="TABLE",
        columns=tuple(
            ColumnMetadata(name=name, data_type="STRING", comment=None) for name in header
        ),
        view_definition=None,
    )


def connector_over(con: duckdb.DuckDBPyConnection, metadata: SchemaMetadata) -> FakeConnector:
    return FakeConnector(metadata, query=query_fn(con))

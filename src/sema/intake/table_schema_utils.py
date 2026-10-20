"""Dictionary rows to Table Schema: raw type mapping and descriptor assembly."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from sema.intake.table_schema import (
    Constraints,
    Field,
    ForeignKey,
    FrictionlessType,
    Reference,
    TableSchema,
)
from sema.log import logger

_SIZE_SUFFIX = re.compile(r"\s*\(.*\)\s*$")
_TYPE_NAMES: tuple[tuple[FrictionlessType, frozenset[str]], ...] = (
    (
        "integer",
        frozenset(
            {"integer", "int", "int2", "int4", "int8", "bigint", "smallint", "tinyint",
             "serial", "bigserial"}
        ),
    ),
    (
        "number",
        frozenset(
            {"numeric", "decimal", "float", "float4", "float8", "double", "real",
             "double precision", "number"}
        ),
    ),
    (
        "string",
        frozenset(
            {"char", "varchar", "text", "string", "character", "character varying",
             "nvarchar", "nchar"}
        ),
    ),
    ("date", frozenset({"date"})),
    (
        "datetime",
        frozenset(
            {"timestamp", "datetime", "timestamptz", "timestamp with time zone",
             "timestamp without time zone"}
        ),
    ),
    ("boolean", frozenset({"boolean", "bool"})),
)
_FLAG_TEXT = {"yes": True, "no": False, "y": True, "n": False, "true": True,
              "false": False, "1": True, "0": False}


@dataclass(frozen=True)
class DictionaryRow:
    property: str
    raw_type: str
    required: bool = False
    primary_key: bool = False
    fk_entity: str | None = None
    fk_property: str | None = None
    domain: str | None = None
    description: str | None = None


def frictionless_type(raw_type: str, column: str) -> FrictionlessType:
    base = _SIZE_SUFFIX.sub("", raw_type.strip().lower())
    for name, raw_names in _TYPE_NAMES:
        if base in raw_names:
            return name
    logger.info(
        "Column {}: raw type {!r} maps to no Frictionless type; using any",
        column,
        raw_type,
    )
    return "any"


def dictionary_flag(value: Any) -> bool:
    """Yes/no, true/false, and 1/0 cells; anything else is false."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return _FLAG_TEXT.get(str(value).strip().lower(), False)


def table_schema_from_rows(
    entity: str, rows: Iterable[DictionaryRow], *, description: str | None = None
) -> TableSchema:
    rows = list(rows)
    return TableSchema(
        fields=[_field(entity, row) for row in rows],
        primary_key=[row.property for row in rows if row.primary_key] or None,
        foreign_keys=_foreign_keys(entity, rows) or None,
        description=description,
    )


def _field(entity: str, row: DictionaryRow) -> Field:
    return Field(
        name=row.property,
        type=frictionless_type(row.raw_type, f"{entity}.{row.property}"),
        description=row.description,
        constraints=Constraints(required=True) if row.required else None,
        x_sema_raw_type=row.raw_type,
        x_sema_domain=row.domain,
    )


def _foreign_keys(entity: str, rows: list[DictionaryRow]) -> list[ForeignKey]:
    keys: list[ForeignKey] = []
    for row in rows:
        if row.fk_entity is None:
            continue
        if row.fk_property is None:
            logger.info(
                "Column {}.{} names entity {} but no referenced property; "
                "no foreign key",
                entity,
                row.property,
                row.fk_entity,
            )
            continue
        reference = Reference(resource=row.fk_entity, fields=[row.fk_property])
        keys.append(ForeignKey(fields=[row.property], reference=reference))
    return keys

"""Synthetic version 2 library in DuckDB: hierarchy forms, lifecycle, lang, mappings.

Concepts: RACE and RACE_DETAIL are active (RACE_DETAIL's parent is RACE);
OLD is retired and replaced by RACE. Values of RACE: RACE_W, RACE_B active,
RACE_X deprecated with replacement RACE_W and valid dates. OLD has one value.
Level and path relations carry their own concept codes.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import duckdb

from sema.binding.spec import ConceptSourceSpec
from sema.intake.relations import RelationProvider
from sema.log import logger
from tests.unit.binding.fixtures.concept_library import warehouse_sql

RETIRED_TERM = "RACE_X"
RETIRED_CONCEPT = "OLD"

_CONCEPTS = (
    ("RACE", "Race", None, "active", None, None, None),
    ("RACE_DETAIL", "Race detail", "RACE", "active", None, None, None),
    ("OLD", "Old element", None, "retired", "2001-01-01", "2019-06-30", "RACE"),
)

_VALUES = (
    ("RACE", "RACE_W", "White", "en", "active", "2000-01-01", None, None, "Caucasian"),
    ("RACE", "RACE_B", "Black", "en", "active", "2000-01-01", None, None, None),
    (
        "RACE",
        RETIRED_TERM,
        "Other (old)",
        "en",
        "deprecated",
        "2000-01-01",
        "2023-12-31",
        "RACE_W",
        None,
    ),
    ("OLD", "OLD_1", "Old value", "en", "active", None, None, None, None),
)

_LEVEL_CONCEPTS = (("A", "Alpha"), ("B", "Beta"), ("C", "Gamma"), ("D", "Delta"))
_LEVELS = (("A", "B", None), ("A", "C", "D"))
_PATHS = (("1", "One"), ("1.1", "One one"), ("1.2.3", "One two three"))

_MAPPINGS = (
    ("syn:RACE_W", "skos:exactMatch", "omop:8527", "semapv:ManualMappingCuration"),
    ("syn:RACE_B", "skos:closeMatch", "omop:8516", "semapv:ManualMappingCuration"),
    ("syn:RACE", "skos:exactMatch", "omop:4013886", "semapv:ManualMappingCuration"),
)

STATUS = {
    "column": "state",
    "active_values": ["active"],
    "retired_values": ["deprecated", "retired"],
}
LIFECYCLE = {
    "status": STATUS,
    "valid_from": "valid_start",
    "valid_to": "valid_end",
    "replaced_by": "successor",
}
MAPPING_ROLES = {
    "subject_id": "subject_id",
    "predicate_id": "predicate_id",
    "object_id": "object_id",
    "mapping_justification": "mapping_justification",
}


def build_library_v2() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    con.execute("CREATE SCHEMA land")
    _table(
        con,
        "concepts",
        "concept_id, label, parent_id, state, valid_start, valid_end, successor",
        _CONCEPTS,
    )
    _table(
        con,
        "vals",
        "concept_id, code, label, label_lang, state, valid_start, valid_end,"
        " successor, synonym",
        _VALUES,
    )
    _table(con, "level_concepts", "code, label", _LEVEL_CONCEPTS)
    _table(con, "levels", "l1, l2, l3", _LEVELS)
    _table(con, "paths", "code, label", _PATHS)
    _table(
        con,
        "sssom",
        "subject_id, predicate_id, object_id, mapping_justification",
        _MAPPINGS,
    )
    return con


def provider_v2(con: duckdb.DuckDBPyConnection) -> RelationProvider:
    def run(sql: str) -> list[tuple[Any, ...]]:
        return con.execute(warehouse_sql(sql)).fetchall()

    return RelationProvider(warehouse=run, staging=None)


def spec_v2(**sections: Any) -> ConceptSourceSpec:
    raw: dict[str, Any] = {
        "spec_version": 2,
        "id": "syn-v2",
        "identity_system": "syn",
        "release": "2026-10",
        "scheme": {"qualified_name": "syn.v2", "description": "Version 2 library"},
        "concepts": {
            "relation": "land.concepts",
            "group_by": "concept_id",
            "roles": {"concept_id": "concept_id", "name": "label"},
        },
    }
    raw.update(sections)
    return ConceptSourceSpec.model_validate(raw)


def value_sets(**roles: Any) -> dict[str, Any]:
    return {
        "relation": "land.vals",
        "roles": {
            "concept_id": "concept_id",
            "code": "code",
            "display": "label",
            "synonyms": "synonym",
            **roles,
        },
    }


@contextmanager
def captured_log() -> Iterator[list[str]]:
    messages: list[str] = []
    sink: Callable[[Any], None] = lambda message: messages.append(str(message))
    sink_id = logger.add(sink, format="{message}", level="DEBUG")
    try:
        yield messages
    finally:
        logger.remove(sink_id)


def _table(
    con: duckdb.DuckDBPyConnection,
    name: str,
    columns: str,
    rows: tuple[tuple[Any, ...], ...],
) -> None:
    names = [c.strip() for c in columns.split(",")]
    con.execute(
        f"CREATE TABLE land.{name} ({', '.join(f'{c} VARCHAR' for c in names)})"
    )
    marks = ", ".join("?" for _ in names)
    con.executemany(f"INSERT INTO land.{name} VALUES ({marks})", rows)

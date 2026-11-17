"""A neutral warehouse of coded columns, with a dictionary that states some of their domains.

One relation per label location and per evidence type, and one column per decoy. Every row
is spelled out in Python, so a test derives its expected numbers from these rows, never
from a run of SEMA. The dictionary is an intake-style concept spec over a staging file:
its code lists are the only place some labels exist.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from sema.ingest.duckdb_staging import Staging
from tests.support.connector_fixture import DeclaredKey, FixtureTable, write_fixture_database

CODES_CATALOG = "fixcodes"
CODES_SCHEMA = "works"
DICTIONARY_SCHEMA = "land"

Row = tuple[Any, ...]

FINISH = {"MAT": "Matte", "GLS": "Gloss", "SAT": "Satin", "TXR": "Textured", "BRU": "Brushed",
          "POL": "Polished"}
TONE_NAMES = {"T-RD": "Crimson", "T-BL": "Cobalt", "T-GN": "Jade", "T-YL": "Saffron",
              "T-BK": "Onyx"}
GRIT = {"G040": ("coarse stock removal", "forty"), "G080": ("shaping", "eighty"),
        "G120": ("smoothing", "one twenty"), "G240": ("pre-finish", "two forty")}
SHIFT_SLOT = {"EARLY": "Dawn rota", "LATE": "Dusk rota", "NIGHT": "Owl rota",
              "SWING": "Relief rota"}
RACK = {101: "north rack", 102: "south rack", 103: "east rack", 104: "west rack", 105: "top rack"}
RACK_SPARE = {**{n: f"spare {n}" for n in RACK}, 106: "spare overflow"}
BAY = {n: f"bay {n}" for n in range(1, 9)}
CREW = {n: f"crew member {n}" for n in range(2001, 3201)}

# Dictionary code lists: vocabulary id -> code -> label. No warehouse relation holds these.
COAT_KINDS = {"CK-A": "Primer", "CK-B": "Enamel", "CK-C": "Lacquer", "CK-D": "Varnish",
              "CK-E": "Powder", "CK-F": "Anodised"}
TRIM_KINDS = {"TR-1": "Bead", "TR-2": "Flat", "TR-3": "Round"}
EDGE_KINDS = {"EK-1": "Sharp", "EK-2": "Eased"}
CODE_LISTS = {"coat_kinds": COAT_KINDS, "trim_kinds": TRIM_KINDS, "edge_kinds": EDGE_KINDS}
STATED_DOMAINS = {
    "panel": {"coat_kind": "coat_kinds", "trim": "trim_kinds", "edge treatment": "edge_kinds"},
    "fitting": {"coat_kind": "coat_kinds"},
}
UNPLACED_PROPERTY = "edge treatment"

PANEL_ROWS = 1500
FITTING_ROWS = 240
_COAT_PATTERN = (("CK-A",) * 6 + ("CK-B",) * 6 + ("CK-C",) * 6 + ("CK-D",) * 5 + ("CK-E",) * 4
                 + ("CK-Z", None, None))
_FINISH_PATTERN = ("MAT", "GLS", "SAT", "TXR", "BRU", None)
_TRIM_PATTERN = ("TR-1", "BEAD", "FLAT", "ROUND", "NONE", "SQ")
_FITTING_COATS = ("CK-A", "CK-B", "CK-C", "CK-D", "CK-E", "CK-F", "CK-P", "CK-Q", "CK-R", "CK-S")


def _cycle(pattern: Sequence[Any], i: int) -> Any:
    return pattern[i % len(pattern)]


def _panel_row(i: int) -> Row:
    return (
        10001 + i,
        _cycle(_FINISH_PATTERN, i),
        _cycle(_COAT_PATTERN, i),
        _cycle(_TRIM_PATTERN, i),
        _cycle(("F1", "F2", "F3", "F4"), i),
        _cycle(("EARLY", "LATE", "NIGHT"), i),
        1 + i % 4,
        _cycle(tuple(GRIT), i),
        _cycle((101, 102, 103), i),
        2001 + i % 1100,
    )


def _fitting_row(i: int) -> Row:
    tone = _cycle(tuple(TONE_NAMES), i)
    return (20001 + i, tone, TONE_NAMES[tone], _cycle(("Z1", "Z2", "Z3"), i),
            _cycle(_FITTING_COATS, i))


ROWS: dict[str, list[Row]] = {
    "finish": list(FINISH.items()),
    "tone": [(code,) for code in TONE_NAMES],
    "grit": [(code, note, alias) for code, (note, alias) in GRIT.items()],
    "shift_slot": list(SHIFT_SLOT.items()),
    "rack": list(RACK.items()),
    "rack_spare": list(RACK_SPARE.items()),
    "bay": list(BAY.items()),
    "crew": list(CREW.items()),
    "panel": [_panel_row(i) for i in range(PANEL_ROWS)],
    "fitting": [_fitting_row(i) for i in range(FITTING_ROWS)],
}


def _literal(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, int):
        return str(value)
    return "'" + str(value).replace("'", "''") + "'"


def _values_sql(rows: list[Row]) -> str:
    return "VALUES " + ", ".join(
        "(" + ", ".join(_literal(value) for value in row) + ")" for row in rows
    )


def _table(
    name: str, columns: tuple[tuple[str, str], ...], keys: tuple[DeclaredKey, ...] = (),
) -> FixtureTable:
    return FixtureTable(name, columns, (columns[0][0],), _values_sql(ROWS[name]), keys)


_TEXT = "VARCHAR"
CODED_TABLES: tuple[FixtureTable, ...] = (
    _table("finish", (("finish_code", _TEXT), ("finish_label", _TEXT))),
    _table("tone", (("tone_code", _TEXT),)),
    _table("grit", (("grit_code", _TEXT), ("grit_note", _TEXT), ("grit_alias", _TEXT))),
    _table("shift_slot", (("slot_key", _TEXT), ("slot_title", _TEXT))),
    _table("rack", (("rack_no", "INTEGER"), ("rack_title", _TEXT))),
    _table("rack_spare", (("rack_no", "INTEGER"), ("spare_title", _TEXT))),
    _table("bay", (("bay_no", "INTEGER"), ("bay_title", _TEXT))),
    _table("crew", (("crew_ref", "INTEGER"), ("crew_name", _TEXT))),
    _table(
        "panel",
        (("panel_no", "INTEGER"), ("finish_code", _TEXT), ("coat_kind", _TEXT), ("trim", _TEXT),
         ("finish_cd", _TEXT), ("shift", _TEXT), ("crew_size", "INTEGER"),
         ("grit_code", _TEXT), ("rack_no", "INTEGER"), ("assigned_crew", "INTEGER")),
        (DeclaredKey(("finish_code",), "finish", ("finish_code",)),
         DeclaredKey(("grit_code",), "grit", ("grit_code",))),
    ),
    _table(
        "fitting",
        (("fitting_no", "INTEGER"), ("tone_code", _TEXT), ("tone_name", _TEXT),
         ("grit_code", _TEXT), ("coat_kind", _TEXT)),
        (DeclaredKey(("tone_code",), "tone", ("tone_code",)),),
    ),
)
CODED_SCHEMA_TABLES = {CODES_SCHEMA: CODED_TABLES}


def column_values(table: str, column: str) -> list[Any]:
    """Every value of one fixture column, in row order, NULLs included."""
    (fixture,) = (t for t in CODED_TABLES if t.name == table)
    index = fixture.column_names.index(column)
    return [row[index] for row in ROWS[table]]


def write_coded_database(directory: Path) -> Path:
    """The coded warehouse as a DuckDB file whose catalog, its file stem, is `CODES_CATALOG`."""
    return write_fixture_database(directory, CODED_SCHEMA_TABLES, CODES_CATALOG)


@dataclass(frozen=True)
class DictionaryFiles:
    spec: Path
    staging: Path


def _field_rows(entity: str) -> list[tuple[str, str]]:
    (table,) = (t for t in CODED_TABLES if t.name == entity)
    names = list(table.column_names)
    names += [prop for prop in STATED_DOMAINS[entity] if prop not in names]
    return [(name, f"the {name.replace('_', ' ')} of a {entity}") for name in names]


def _spec() -> dict[str, Any]:
    return {"concept_source": {
        "spec_version": 2,
        "id": "fixcodes-dictionary",
        "identity_system": "fixcodes",
        "release": "1",
        "default_provider": "staging",
        "entity_model": {
            "layout": "per_file",
            "roles": {"property": "field_name", "description": "field_note"},
            "entities": [
                {"relation": f"{DICTIONARY_SCHEMA}.{entity}_fields", "entity": entity,
                 "domains": domains}
                for entity, domains in STATED_DOMAINS.items()
            ],
        },
        "code_lists": [
            {"vocabulary_id": vocabulary, "relation": f"{DICTIONARY_SCHEMA}.{vocabulary}",
             "roles": {"code": "kind_code", "display": "kind_text"}}
            for vocabulary in CODE_LISTS
        ],
    }}


def write_dictionary(directory: Path) -> DictionaryFiles:
    """The dictionary's staging file and the concept spec that reads it."""
    staging_path = directory / "dictionary.duckdb"
    with Staging(str(staging_path), schemas=()) as staging:
        staging.execute(f"CREATE SCHEMA {DICTIONARY_SCHEMA}")
        for entity in STATED_DOMAINS:
            relation = f"{DICTIONARY_SCHEMA}.{entity}_fields"
            staging.execute(f"CREATE TABLE {relation} (field_name VARCHAR, field_note VARCHAR)")
            staging.execute(f"INSERT INTO {relation} {_values_sql(list(_field_rows(entity)))}")
        for vocabulary, codes in CODE_LISTS.items():
            relation = f"{DICTIONARY_SCHEMA}.{vocabulary}"
            staging.execute(f"CREATE TABLE {relation} (kind_code VARCHAR, kind_text VARCHAR)")
            staging.execute(f"INSERT INTO {relation} {_values_sql(list(codes.items()))}")
    spec_path = directory / "dictionary_spec.yaml"
    spec_path.write_text(yaml.safe_dump(_spec(), sort_keys=False), encoding="utf-8")
    return DictionaryFiles(spec_path, staging_path)

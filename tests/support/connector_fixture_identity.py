"""A neutral warehouse for identity facts: keys, entities coarser than a row, shared
entities, attribute rules and label sources, with a decoy for each predicate.

One relation per case. Every row is spelled out in Python, so a test derives each expected
number from these rows, never from a run of SEMA. The dictionary is an intake-style concept
spec over a staging file: it states two code lists with their code and display roles, two
property domains and one foreign key with reference columns. It states no identity.

The helpers at the end count on the rows the way the contract's count statements do: a
NULL part of a tuple counts as a value, a conflict is a key group holding more than one
distinct non-NULL value of a column, and containment compares distinct non-NULL tuples.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import yaml

from sema.ingest.duckdb_staging import Staging
from tests.support.connector_fixture import DeclaredKey
from tests.support.connector_fixture_codes import DictionaryFiles

IDENTITY_CATALOG = "fixident"
IDENTITY_SCHEMA = "yard"
DICTIONARY_SCHEMA = "land"

Row = tuple[Any, ...]
_INT, _TEXT = "INTEGER", "VARCHAR"


@dataclass(frozen=True)
class IdentityTable:
    """A relation, its rows, and the keys it declares; no primary key when none is named."""

    name: str
    columns: tuple[tuple[str, str], ...]
    rows: tuple[Row, ...]
    primary_key: tuple[str, ...] = ()
    unique: tuple[tuple[str, ...], ...] = ()
    references: tuple[DeclaredKey, ...] = ()

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.columns)


# Code relations. `rig_kind` is also the dictionary's code list `rig_kinds`: its codes are
# the list's codes, its texts are not the list's texts, one text column is constant and
# the dictionary's display role names another. `dock_kind` has two non-constant texts.
RIG_TITLES = {"SL": "sloop", "KT": "ketch", "YW": "yawl", "CT": "cutter"}
RIG_NOTES = {"SL": "fore and aft", "KT": "two masts", "YW": "two masts", "CT": "fore and aft"}
RIG_GROUP = "sail"
RIG_LIST_TEXTS = {"SL": "Sloop rig", "KT": "Ketch rig", "YW": "Yawl rig", "CT": "Cutter rig"}
DOCK = {"D1": ("dry dock", "covered"), "D2": ("wet dock", "uncovered"),
        "D3": ("float dock", "covered")}
# A code list that exists only in the dictionary.
HATCH_LIST_TEXTS = {"H1": "Flush hatch", "H2": "Raised hatch", "H3": "Sliding hatch"}

RIG_KIND = IdentityTable(
    "rig_kind",
    (("rig_code", _TEXT), ("kind_group", _TEXT), ("rig_title", _TEXT), ("rig_note", _TEXT)),
    tuple((code, RIG_GROUP, title, RIG_NOTES[code]) for code, title in RIG_TITLES.items()),
)
DOCK_KIND = IdentityTable(
    "dock_kind", (("dock_code", _TEXT), ("dock_title", _TEXT), ("dock_note", _TEXT)),
    tuple((code, title, note) for code, (title, note) in DOCK.items()), ("dock_code",),
)
# A declared single-column key, and a coded column with a declared key to `dock_kind`.
BERTH = IdentityTable(
    "berth", (("berth_no", _INT), ("berth_side", _TEXT), ("depth_band", _INT),
              ("dock_code", _TEXT)),
    ((1, "E", 1, "D1"), (2, "W", 1, "D2"), (3, "E", 2, "D1"), (4, "W", 2, "D3"),
     (5, "E", 3, "D2"), (6, "W", 3, "D1")),
    ("berth_no",), references=(DeclaredKey(("dock_code",), "dock_kind", ("dock_code",)),),
)
# A two-column key; one entity per hull, several rows per hull, one per version. The paint
# changes between versions of two hulls, the keel and the rig never change. The rig is coded
# by `rig_kinds`, under a name no code relation's column has.
HULL_REV = IdentityTable(
    "hull_rev", (("hull_no", _INT), ("rev_no", _INT), ("paint_band", _TEXT),
                 ("keel_class", _TEXT), ("rig_type", _TEXT)),
    ((1, 1, "red", "deep", "SL"), (1, 2, "blue", "deep", "SL"), (1, 3, "blue", "deep", "SL"),
     (2, 1, "green", "fin", "KT"), (2, 2, "green", "fin", "KT"),
     (3, 1, "red", "deep", "SL"),
     (4, 1, "red", "fin", "YW"), (4, 2, "grey", "fin", "YW")),
    ("hull_no", "rev_no"),
)
# Two child relations whose every hull is a hull of `hull_rev`; a third whose hulls mostly
# are not (a decoy for a shared entity).
HULL_SURVEY = IdentityTable(
    "hull_survey", (("survey_no", _INT), ("hull_no", _INT), ("survey_grade", _TEXT)),
    ((1, 1, "A"), (2, 1, "B"), (3, 2, "A"), (4, 3, "B"), (5, 4, "A"), (6, 4, "B")),
    ("survey_no",),
)
HULL_CARGO = IdentityTable(
    "hull_cargo", (("cargo_no", _INT), ("hull_no", _INT), ("cargo_tons", _INT),
                   ("hatch_kind", _TEXT)),
    ((1, 2, 10, "H1"), (2, 3, 20, "H2"), (3, 3, 10, "H1"), (4, 4, 20, "H3"), (5, 1, 10, "H2")),
    ("cargo_no",),
)
HULL_QUOTE = IdentityTable(
    "hull_quote", (("quote_no", _INT), ("hull_no", _INT), ("quote_band", _INT)),
    ((1, 1, 5), (2, 7, 5), (3, 7, 6), (4, 8, 6), (5, 8, 5)), ("quote_no",),
)
# A column unique by coincidence that nothing corroborates, beside a declared key.
TIDE_LOG = IdentityTable(
    "tide_log", (("log_no", _INT), ("reading_mm", _INT), ("tide_phase", _TEXT)),
    ((1, 412, "ebb"), (2, 388, "flood"), (3, 405, "ebb"), (4, 371, "flood"), (5, 399, "ebb")),
    ("log_no",),
)
# A declared unique column set with a NULL part on three rows.
MOORING = IdentityTable(
    "mooring", (("pier_code", _TEXT), ("slot_no", _INT), ("mooring_fee", _INT)),
    (("P1", 1, 50), ("P1", 2, 50), ("P2", 1, 60), (None, 1, 60), (None, 2, 70), (None, 3, 70)),
    unique=(("pier_code", "slot_no"),),
)
# An ordering column with a tie inside one buoy: no ordering picks one row of that buoy.
BUOY_CHECK = IdentityTable(
    "buoy_check", (("check_no", _INT), ("buoy_no", _INT), ("check_day", _INT),
                   ("lamp_state", _TEXT), ("buoy_zone", _TEXT)),
    ((1, 1, 10, "lit", "north"), (2, 1, 12, "dark", "north"), (3, 1, 12, "lit", "north"),
     (4, 2, 10, "lit", "south"), (5, 2, 11, "lit", "south"), (6, 3, 9, "dark", "north")),
    ("check_no",),
)
# A versioned relation keyed by a unique surrogate; two rows have no crane.
CRANE_REV = IdentityTable(
    "crane_rev", (("crane_rev_id", _INT), ("crane_no", _INT), ("rev_no", _INT),
                  ("lift_tons", _INT)),
    ((101, 1, 1, 5), (102, 1, 2, 8), (103, 2, 1, 10), (104, 2, 2, 10), (105, 2, 3, 10),
     (106, None, 1, 3), (107, None, 2, 4)),
    ("crane_rev_id",),
)
# True duplicate rows and no key.
GATE_TALLY = IdentityTable(
    "gate_tally", (("gate_code", _INT), ("tally", _INT)),
    ((1, 5), (1, 5), (2, 7), (2, 7), (3, 5)),
)
# Two relations keyed by a same-named dense integer column, one inside the other; the
# second also declares a unique superset of its key.
PUMP = IdentityTable(
    "pump", (("unit_no", _INT), ("pump_head", _INT)),
    ((1, 30), (2, 30), (3, 40), (4, 40), (5, 50), (6, 50)), ("unit_no",),
)
VALVE = IdentityTable(
    "valve", (("unit_no", _INT), ("valve_tag", _TEXT)),
    ((1, "VA"), (2, "VB"), (3, "VA"), (4, "VB")), ("unit_no",),
    unique=(("unit_no", "valve_tag"),),
)
# The dictionary's foreign key names `slip_ref`: unique on two relations, repeated on a third.
# The three hold disjoint refs, so no containment corroborates the list on either.
SLIP_A = IdentityTable(
    "slip_a", (("slip_id", _INT), ("slip_ref", _TEXT), ("slip_len", _INT)),
    ((1, "S-1", 10), (2, "S-2", 10), (3, "S-3", 12), (4, "S-4", 12)), ("slip_id",),
)
SLIP_B = IdentityTable(
    "slip_b", (("slip_ref", _TEXT), ("slip_width", _INT)),
    (("S-5", 3), ("S-6", 3), ("S-7", 4)),
)
SLIP_USE = IdentityTable(
    "slip_use", (("use_no", _INT), ("slip_ref", _TEXT)),
    ((1, "S-8"), (2, "S-8"), (3, "S-9"), (4, "S-9")), ("use_no",),
)
# A member relation: several calls per vessel, each with its own day; one call has no vessel.
BERTH_CALL = IdentityTable(
    "berth_call", (("call_no", _INT), ("vessel_no", _INT), ("call_day", _INT),
                   ("call_berth", _INT)),
    ((1, 1, 3, 1), (2, 1, 5, 2), (3, 1, 9, 1), (4, 2, 3, 2), (5, 2, 5, 1), (6, 3, 9, 2),
     (7, None, 2, 1)),
    ("call_no",),
)
# A deck's class follows from its number: a key naming both holds a redundant column.
DECK_PLAN = IdentityTable(
    "deck_plan", (("plan_no", _INT), ("deck_no", _INT), ("deck_class", _TEXT)),
    ((1, 1, "A"), (2, 1, "A"), (3, 2, "B"), (4, 2, "B"), (5, 3, "A"), (6, 3, "A")),
    ("plan_no",),
)

IDENTITY_TABLES: tuple[IdentityTable, ...] = (
    DOCK_KIND, RIG_KIND, BERTH, HULL_REV, HULL_SURVEY, HULL_CARGO, HULL_QUOTE, TIDE_LOG,
    MOORING, BUOY_CHECK, CRANE_REV, GATE_TALLY, PUMP, VALVE, SLIP_A, SLIP_B, SLIP_USE,
    BERTH_CALL, DECK_PLAN,
)
TABLES = {table.name: table for table in IDENTITY_TABLES}


@dataclass(frozen=True)
class Proposal:
    """What the mocked model proposes for one relation (contract F4's fields)."""

    entity: str
    entity_key: tuple[str, ...]
    latest_by: str | None = None
    row_key: tuple[str, ...] | None = None


PROPOSALS: dict[str, Proposal] = {
    "dock_kind": Proposal("dock kind", ("dock_code",)),
    "rig_kind": Proposal("rig kind", ("rig_code",)),
    "berth": Proposal("berth", ("berth_no",)),
    "hull_rev": Proposal("Hull", ("hull_no",), "rev_no"),
    "hull_survey": Proposal("survey", ("survey_no",)),
    "hull_cargo": Proposal("cargo", ("cargo_no",)),
    "hull_quote": Proposal("quote", ("quote_no",)),
    "tide_log": Proposal("tide reading", ("log_no",)),
    "mooring": Proposal("mooring", ("pier_code", "slot_no")),
    "buoy_check": Proposal("Buoy", ("buoy_no",), "check_day"),
    "crane_rev": Proposal("Crane", ("crane_no",), "rev_no"),
    "gate_tally": Proposal("gate", ("gate_code",), None, ("gate_code", "tally")),
    "pump": Proposal("pump", ("unit_no",)),
    "valve": Proposal("valve", ("unit_no",)),
    "slip_a": Proposal("slip berth", ("slip_id",)),
    "slip_b": Proposal("slip", ("slip_ref",)),
    "slip_use": Proposal("slip use", ("use_no",)),
    "berth_call": Proposal("Vessel", ("vessel_no",), "call_day"),
    "deck_plan": Proposal("deck", ("deck_no", "deck_class")),
}
# The entity the mocked L2 stage names for a relation; any other is named after its table.
# `hull_cargo` shares L2's entity with `hull_rev`, which becomes the entity's owner.
L2_ENTITIES = {"hull_rev": "Hull", "hull_cargo": "Hull", "crane_rev": "Crane"}

# The dictionary: per-file entries with their domains and foreign keys, and two code lists.
HULL_ENTRY = ("hull", "Hull", {"Rig Type": "rig_kinds"})
CARGO_ENTRY = ("cargo", "Hull Cargo", {"Hatch Kind": "hatch_kinds"})
SLIP_USE_ENTRY = ("slip_use", "Slip Use", {})
SLIP_REFERENCE = ("slip_ref",)
CODE_LIST_ROLES = {"rig_kinds": ("rig_code", "rig_title"), "hatch_kinds": ("hatch_code", "hatch_text")}
CODE_LIST_TEXTS = {"rig_kinds": RIG_LIST_TEXTS, "hatch_kinds": HATCH_LIST_TEXTS}


def column_values(table: str, column: str) -> list[Any]:
    """Every value of one fixture column, in row order, NULLs included."""
    fixture = TABLES[table]
    index = fixture.column_names.index(column)
    return [row[index] for row in fixture.rows]


def tuples(table: str, columns: Sequence[str]) -> list[Row]:
    return list(zip(*(column_values(table, c) for c in columns), strict=True))


def distinct_count(table: str, columns: Sequence[str]) -> int:
    """Distinct tuples; a NULL part counts as a value."""
    return len(set(tuples(table, columns)))


def null_row_count(table: str, columns: Sequence[str]) -> int:
    """Rows with a NULL in any listed column."""
    return sum(None in t for t in tuples(table, columns))


def conflict_counts(
    table: str, key: Sequence[str], attributes: Sequence[str],
) -> tuple[int, dict[str, int]]:
    """Key groups, and per attribute the groups holding more than one distinct non-NULL value."""
    groups: dict[Row, list[Row]] = {}
    for key_part, values in zip(tuples(table, key), tuples(table, attributes), strict=True):
        groups.setdefault(key_part, []).append(values)
    conflicting = {
        name: sum(len({v[i] for v in held if v[i] is not None}) > 1 for held in groups.values())
        for i, name in enumerate(attributes)
    }
    return len(groups), conflicting


def contained_counts(
    table: str, columns: Sequence[str], owner: str, owner_columns: Sequence[str],
) -> tuple[int, int]:
    """Distinct non-NULL tuples of the relation, and how many of them the owner holds."""
    held = {t for t in tuples(table, columns) if None not in t}
    return len(held), len(held & set(tuples(owner, owner_columns)))


def non_key_columns(table: str, key: Sequence[str]) -> tuple[str, ...]:
    return tuple(c for c in TABLES[table].column_names if c not in key)


def conflict_shares(table: str, key: Sequence[str]) -> dict[str, float]:
    """Each non-key column's share of entity groups holding more than one non-NULL value."""
    groups, conflicting = conflict_counts(table, key, non_key_columns(table, key))
    return {name: count / groups for name, count in conflicting.items()}


def _literal(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, int):
        return str(value)
    return "'" + str(value).replace("'", "''") + "'"


def _values_sql(rows: Sequence[Row]) -> str:
    return "VALUES " + ", ".join(
        "(" + ", ".join(_literal(value) for value in row) + ")" for row in rows
    )


def _create_sql(table: IdentityTable) -> str:
    parts = [f"{name} {data_type}" for name, data_type in table.columns]
    if table.primary_key:
        parts.append(f"PRIMARY KEY ({', '.join(table.primary_key)})")
    parts.extend(f"UNIQUE ({', '.join(columns)})" for columns in table.unique)
    parts.extend(
        f"FOREIGN KEY ({', '.join(key.columns)}) REFERENCES "
        f"{IDENTITY_SCHEMA}.{key.referenced_table} ({', '.join(key.referenced_columns)})"
        for key in table.references
    )
    return f"CREATE TABLE {IDENTITY_SCHEMA}.{table.name} ({', '.join(parts)})"


def write_identity_database(
    directory: Path, tables: tuple[IdentityTable, ...] = IDENTITY_TABLES,
) -> Path:
    """The warehouse as a DuckDB file whose catalog, its file stem, is `IDENTITY_CATALOG`."""
    path = directory / f"{IDENTITY_CATALOG}.duckdb"
    with duckdb.connect(str(path)) as con:
        con.execute(f"CREATE SCHEMA {IDENTITY_SCHEMA}")
        for table in tables:
            con.execute(_create_sql(table))
            con.execute(f"INSERT INTO {IDENTITY_SCHEMA}.{table.name} {_values_sql(table.rows)}")
    return path


def _entries() -> list[tuple[str, str, dict[str, str], list[dict[str, Any]]]]:
    reference = list(SLIP_REFERENCE)
    foreign_keys = [{"columns": reference, "entity": "Slip", "reference_columns": reference}]
    return [(*HULL_ENTRY, []), (*CARGO_ENTRY, []), (*SLIP_USE_ENTRY, foreign_keys)]


def _spec() -> dict[str, Any]:
    entities = [
        {"relation": f"{DICTIONARY_SCHEMA}.{stem}_fields", "entity": entity, "domains": domains,
         **({"foreign_keys": keys} if keys else {})}
        for stem, entity, domains, keys in _entries()
    ]
    code_lists = [
        {"vocabulary_id": name, "relation": f"{DICTIONARY_SCHEMA}.{name}",
         "roles": {"code": code, "display": display}}
        for name, (code, display) in CODE_LIST_ROLES.items()
    ]
    return {"concept_source": {
        "spec_version": 2, "id": "fixident-dictionary", "identity_system": "fixident",
        "release": "1", "default_provider": "staging",
        "entity_model": {"layout": "per_file",
                         "roles": {"property": "field_name", "description": "field_note"},
                         "entities": entities},
        "code_lists": code_lists,
    }}


def write_identity_dictionary(directory: Path) -> DictionaryFiles:
    """The dictionary's staging file and the concept spec that reads it."""
    staging_path = directory / "identity_dictionary.duckdb"
    with Staging(str(staging_path), schemas=()) as staging:
        staging.execute(f"CREATE SCHEMA {DICTIONARY_SCHEMA}")
        for stem, entity, domains, keys in _entries():
            relation = f"{DICTIONARY_SCHEMA}.{stem}_fields"
            fields = [*domains, *(c for key in keys for c in key["columns"])]
            staging.execute(f"CREATE TABLE {relation} (field_name VARCHAR, field_note VARCHAR)")
            staging.execute(f"INSERT INTO {relation} "
                            + _values_sql([(f, f"the {f} of a {entity}") for f in fields]))
        for name, (code, display) in CODE_LIST_ROLES.items():
            relation = f"{DICTIONARY_SCHEMA}.{name}"
            staging.execute(f"CREATE TABLE {relation} ({code} VARCHAR, {display} VARCHAR)")
            staging.execute(f"INSERT INTO {relation} "
                            + _values_sql(list(CODE_LIST_TEXTS[name].items())))
    spec_path = directory / "identity_dictionary_spec.yaml"
    spec_path.write_text(yaml.safe_dump(_spec(), sort_keys=False), encoding="utf-8")
    return DictionaryFiles(spec_path, staging_path)

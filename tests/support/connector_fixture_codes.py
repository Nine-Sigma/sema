"""A neutral warehouse of coded columns, with a dictionary that states some of their domains.

One relation per label location and per evidence type, and one column per decoy. Every row
is spelled out in Python, so a test derives its expected numbers from these rows, never
from a run of SEMA. The dictionary is an intake-style concept spec over a staging file:
its code lists are the only place some labels exist, and it spells properties the way a
document does, not the way a column is named.

Each column of a fact relation cycles with its own period, and no period divides another,
so no text column is a function of a coded column unless a case means it (the sibling
labels). Wide columns run in consecutive blocks for the same reason.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from sema.ingest.duckdb_staging import Staging
from sema.models.config import ProfilingConfig
from tests.support.connector_fixture import DeclaredKey, FixtureTable, write_fixture_database

CODES_CATALOG = "fixcodes"
CODES_SCHEMA = "works"
DICTIONARY_SCHEMA = "land"
LOW_DISTINCT_BOUND: int = ProfilingConfig.model_fields["categorical_threshold"].default

Row = tuple[Any, ...]

FINISH = {"MAT": "Matte", "GLS": "Gloss", "SAT": "Satin", "TXR": "Textured", "BRU": "Brushed",
          "POL": "Polished"}
TONE_NAMES = {"T-RD": "Crimson", "T-BL": "Cobalt", "T-GN": "Jade", "T-YL": "Saffron",
              "T-BK": "Onyx"}
SHEEN = {"SH-1": ("Flat sheen", "least light"), "SH-2": ("Eggshell sheen", "some light"),
         "SH-3": ("Pearl sheen", "most light")}
GLAZE = {"GZ-1": "Clear glaze", "GZ-2": "Amber glaze", "GZ-3": "Smoke glaze",
         "GZ-4": "Frost glaze"}
GRIT = {"G040": ("coarse stock removal", "forty"), "G080": ("shaping", "eighty"),
        "G120": ("smoothing", "one twenty"), "G240": ("pre-finish", "two forty")}
SHIFT_SLOT = {"EARLY": "Dawn rota", "LATE": "Dusk rota", "NIGHT": "Owl rota",
              "SWING": "Relief rota"}
RACK = {101: "north rack", 102: "south rack", 103: "east rack", 104: "west rack", 105: "top rack"}
RACK_SPARE = {**{n: f"spare {n}" for n in RACK}, 106: "spare overflow"}
BAY = {n: f"bay {n}" for n in range(1, 9)}
CREW = {n: f"crew member {n}" for n in range(2001, 2001 + LOW_DISTINCT_BOUND)}
PALLET = {n: f"pallet {n}" for n in range(1, LOW_DISTINCT_BOUND + 2)}

# Dictionary code lists: vocabulary id -> code -> label. No warehouse relation holds these.
COAT_KINDS = {"CK-A": "Primer", "CK-B": "Enamel", "CK-C": "Lacquer", "CK-D": "Varnish",
              "CK-E": "Powder", "CK-F": "Anodised"}
TRIM_KINDS = {"TR-1": "Bead", "TR-2": "Flat", "TR-3": "Round"}
EDGE_KINDS = {"EK-1": "Sharp", "EK-2": "Eased"}
SEAL_KINDS = {"SL-1": "Wax", "SL-2": "Oil", "SL-3": "Resin"}
BEVEL_KINDS = {"BV-1": "Chamfer", "BV-2": "Ogee"}
CODE_LISTS = {"coat_kinds": COAT_KINDS, "trim_kinds": TRIM_KINDS, "edge_kinds": EDGE_KINDS,
              "seal_kinds": SEAL_KINDS, "bevel_kinds": BEVEL_KINDS}


@dataclass(frozen=True)
class DictionaryEntry:
    """One dictionary file: the entity it names, if any, and its stated property domains."""

    stem: str
    entity: str | None
    domains: dict[str, str]


DICTIONARY_ENTRIES = (
    DictionaryEntry("panel", "Panel", {"Coat Kind": "coat_kinds", "TRIM": "trim_kinds",
                                       "Edge Treatment": "edge_kinds"}),
    DictionaryEntry("fitting", "fitting", {"coat-kind": "coat_kinds"}),
    DictionaryEntry("shared", None, {"Seal / Kind": "seal_kinds"}),
    DictionaryEntry("wall_fitting", "Wall Fitting", {"Bevel Kind.": "bevel_kinds"}),
)
UNPLACED_PROPERTY = "Edge Treatment"
# A token and a substring of column names that equal none: names only nominate on equality.
CASE_ENTRIES = (DictionaryEntry("loose", None, {"Kind": "trim_kinds", "Coat": "coat_kinds"}),)


def normalized_name(name: str) -> str:
    """Lowercase, each run of other characters one `_`, no `_` at either end."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


PANEL_ROWS = 1500
FITTING_ROWS = 240
_FINISH_PATTERN = ("MAT", "GLS", "SAT", "TXR", "BRU", None)
_COAT_PATTERN = (("CK-A", "CK-B", "CK-C", "CK-D", "CK-E") * 5
                 + ("CK-A", None, "CK-B", "CK-Z", "CK-C", None))
_TRIM_PATTERN = ("TR-1", "BEAD", "FLAT", "ROUND", "NONE", "SQ", "OVAL")
_FINISH_CD_PATTERN = ("F1", "F2", "F3", "F4", "F5")
_SHIFT_PATTERN = ("EARLY", "LATE", "NIGHT") * 3 + ("EARLY", "LATE")
_PANEL_SEALS = ("SL-1", "SL-2", "SL-3") * 4 + ("SL-1",)
_BEVELS = ("BV-1", "BV-2") * 8 + ("BV-1",)
_TONES = (*TONE_NAMES, "T-RD", "T-BL")
_SHEENS = (*SHEEN, *SHEEN, *SHEEN, "SH-1", "SH-2")
_FITTING_COATS = ("CK-A", "CK-B", "CK-C", "CK-D", "CK-E", "CK-F", "CK-P", "CK-Q", "CK-R", "CK-S")
_FITTING_SEALS = ("SL-1", "SL-2", "SL-3") * 5 + ("SL-1", "SL-2")
_FITTING_TRIMS = ("TR-1", "TR-2", "TR-3") * 6 + ("TR-1",)


def _cycle(pattern: Sequence[Any], i: int) -> Any:
    return pattern[i % len(pattern)]


def _panel_row(i: int) -> Row:
    return (
        10001 + i,
        _cycle(_FINISH_PATTERN, i),
        _cycle(_COAT_PATTERN, i),
        _cycle(_TRIM_PATTERN, i),
        _cycle(_FINISH_CD_PATTERN, i),
        _cycle(_SHIFT_PATTERN, i),
        1 + i % 3,
        _cycle(tuple(GRIT), i),
        _cycle((101, 102, 103), i),
        2001 + i * (LOW_DISTINCT_BOUND + 1) // PANEL_ROWS,
        2001 + i * LOW_DISTINCT_BOUND // PANEL_ROWS,
        201 + i * 4 // PANEL_ROWS,
        _cycle(_PANEL_SEALS, i),
        _cycle(_BEVELS, i),
    )


def _fitting_row(i: int) -> Row:
    tone, sheen = _cycle(_TONES, i), _cycle(_SHEENS, i)
    return (20001 + i, tone, TONE_NAMES[tone], _cycle(("Z1", "Z2", "Z3"), i),
            _cycle(_FITTING_COATS, i), sheen, *SHEEN[sheen], _cycle(tuple(GLAZE), i),
            _cycle(_FITTING_SEALS, i), _cycle(_FITTING_TRIMS, i))


# The second review's cases, in relations of their own so the tables above keep their numbers.
STRIP_ROWS = 200
HUE = {"H-1": "Ash", "H-2": "Birch", "H-3": "Cedar"}
DYE = {"D-1": "Ochre", "D-2": "Umber", "D-3": "Sienna", "D-4": "Indigo", "D-5": "Madder"}
GLINT = {"GL-1": "Faint glint", "GL-2": "Bright glint", "GL-3": "Blinding glint"}
SPARK = {"GL-1": "spark one", "GL-2": "spark two", "SP-3": "spark three", "SP-4": "spark four"}
HUE_NOISE_PERIOD = 20
DYE_NOISE = (0, 1)
DYE_NOISE_PERIOD = 7


def _strip_row(i: int) -> Row:
    """`hue_name` names another hue on one row in 20; `dye_name` another dye on two in 7."""
    hues, dyes = tuple(HUE), tuple(DYE)
    hue, dye = _cycle(hues, i), _cycle(dyes, i)
    hue_name = HUE[_cycle(hues, i + 1)] if i % HUE_NOISE_PERIOD == 0 else HUE[hue]
    dye_name = DYE[_cycle(dyes, i + 1)] if i % DYE_NOISE_PERIOD in DYE_NOISE else DYE[dye]
    return (30001 + i, hue, hue_name, dye, dye_name, _cycle(("GL-1", "GL-2"), i))


ROWS: dict[str, list[Row]] = {
    "hue": [(code,) for code in HUE],
    "dye": [(code,) for code in DYE],
    "glint": list(GLINT.items()),
    "spark": list(SPARK.items()),
    "strip": [_strip_row(i) for i in range(STRIP_ROWS)],
    "finish": list(FINISH.items()),
    "tone": [(code,) for code in TONE_NAMES],
    "sheen": [(code,) for code in SHEEN],
    "glaze": list(GLAZE.items()),
    "grit": [(code, note, alias) for code, (note, alias) in GRIT.items()],
    "shift_slot": list(SHIFT_SLOT.items()),
    "rack": list(RACK.items()),
    "rack_spare": list(RACK_SPARE.items()),
    "bay": list(BAY.items()),
    "crew": list(CREW.items()),
    "pallet": list(PALLET.items()),
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


def _key(column: str, table: str) -> DeclaredKey:
    return DeclaredKey((column,), table, (column,))


_TEXT = "VARCHAR"
_INT = "INTEGER"
CODED_TABLES: tuple[FixtureTable, ...] = (
    _table("finish", (("finish_code", _TEXT), ("finish_label", _TEXT))),
    _table("tone", (("tone_code", _TEXT),)),
    _table("sheen", (("sheen_code", _TEXT),)),
    _table("glaze", (("glaze_code", _TEXT), ("glaze_title", _TEXT))),
    _table("grit", (("grit_code", _TEXT), ("grit_note", _TEXT), ("grit_alias", _TEXT))),
    _table("shift_slot", (("slot_key", _TEXT), ("slot_title", _TEXT))),
    _table("rack", (("rack_no", _INT), ("rack_title", _TEXT))),
    _table("rack_spare", (("rack_no", _INT), ("spare_title", _TEXT))),
    _table("bay", (("bay_no", _INT), ("bay_title", _TEXT))),
    _table("crew", (("crew_ref", _INT), ("crew_name", _TEXT))),
    _table("pallet", (("pallet_no", _INT), ("pallet_title", _TEXT))),
    _table(
        "panel",
        (("panel_no", _INT), ("finish_code", _TEXT), ("coat_kind", _TEXT), ("trim", _TEXT),
         ("finish_cd", _TEXT), ("shift", _TEXT), ("crew_size", _INT), ("grit_code", _TEXT),
         ("rack_no", _INT), ("assigned_crew", _INT), ("crew_ref", _INT), ("pallet_no", _INT),
         ("seal_kind", _TEXT), ("bevel_kind", _TEXT)),
        (_key("finish_code", "finish"), _key("grit_code", "grit"), _key("pallet_no", "pallet")),
    ),
    _table(
        "fitting",
        (("fitting_no", _INT), ("tone_code", _TEXT), ("tone_name", _TEXT), ("grit_code", _TEXT),
         ("coat_kind", _TEXT), ("sheen_code", _TEXT), ("sheen_name", _TEXT),
         ("sheen_hint", _TEXT), ("glaze_code", _TEXT), ("seal_kind", _TEXT), ("trim", _TEXT)),
        (_key("tone_code", "tone"), _key("sheen_code", "sheen")),
    ),
)
CODED_SCHEMA_TABLES = {CODES_SCHEMA: CODED_TABLES}
FACT_TABLES = ("panel", "fitting")
CASE_TABLES: tuple[FixtureTable, ...] = (
    _table("hue", (("hue_code", _TEXT),)),
    _table("dye", (("dye_code", _TEXT),)),
    _table("glint", (("glint_code", _TEXT), ("glint_title", _TEXT))),
    _table("spark", (("spark_code", _TEXT), ("spark_title", _TEXT))),
    _table(
        "strip",
        (("strip_no", _INT), ("hue_code", _TEXT), ("hue_name", _TEXT), ("dye_code", _TEXT),
         ("dye_name", _TEXT), ("glint_code", _TEXT)),
        (_key("hue_code", "hue"), _key("dye_code", "dye")),
    ),
)


def column_values(table: str, column: str) -> list[Any]:
    """Every value of one fixture column, in row order, NULLs included."""
    (fixture,) = (t for t in (*CODED_TABLES, *CASE_TABLES) if t.name == table)
    index = fixture.column_names.index(column)
    return [row[index] for row in ROWS[table]]


def write_coded_database(
    directory: Path, tables: tuple[FixtureTable, ...] = CODED_TABLES,
) -> Path:
    """The coded warehouse as a DuckDB file whose catalog, its file stem, is `CODES_CATALOG`."""
    return write_fixture_database(directory, {CODES_SCHEMA: tables}, CODES_CATALOG)


@dataclass(frozen=True)
class DictionaryFiles:
    spec: Path
    staging: Path


def named_table(entry: DictionaryEntry) -> FixtureTable | None:
    """The warehouse relation whose normalized name equals the entry's entity, if any."""
    if entry.entity is None:
        return None
    wanted = normalized_name(entry.entity)
    return next((t for t in CODED_TABLES if normalized_name(t.name) == wanted), None)


def _field_rows(entry: DictionaryEntry) -> list[tuple[str, str]]:
    table = named_table(entry)
    stated = {normalized_name(prop) for prop in entry.domains}
    columns = [c for c in table.column_names if c not in stated] if table else []
    subject = entry.entity or entry.stem
    return [(name, f"the {name} of a {subject}") for name in [*columns, *entry.domains]]


def _entity_entry(entry: DictionaryEntry) -> dict[str, Any]:
    named = {"entity": entry.entity} if entry.entity else {}
    return {"relation": f"{DICTIONARY_SCHEMA}.{entry.stem}_fields", **named,
            "domains": entry.domains}


def _spec(entries: tuple[DictionaryEntry, ...]) -> dict[str, Any]:
    return {"concept_source": {
        "spec_version": 2,
        "id": "fixcodes-dictionary",
        "identity_system": "fixcodes",
        "release": "1",
        "default_provider": "staging",
        "entity_model": {
            "layout": "per_file",
            "roles": {"property": "field_name", "description": "field_note"},
            "entities": [_entity_entry(entry) for entry in entries],
        },
        "code_lists": [
            {"vocabulary_id": vocabulary, "relation": f"{DICTIONARY_SCHEMA}.{vocabulary}",
             "roles": {"code": "kind_code", "display": "kind_text"}}
            for vocabulary in CODE_LISTS
        ],
    }}


def write_dictionary(
    directory: Path, entries: tuple[DictionaryEntry, ...] = DICTIONARY_ENTRIES,
) -> DictionaryFiles:
    """The dictionary's staging file and the concept spec that reads it."""
    staging_path = directory / "dictionary.duckdb"
    with Staging(str(staging_path), schemas=()) as staging:
        staging.execute(f"CREATE SCHEMA {DICTIONARY_SCHEMA}")
        for entry in entries:
            relation = f"{DICTIONARY_SCHEMA}.{entry.stem}_fields"
            staging.execute(f"CREATE TABLE {relation} (field_name VARCHAR, field_note VARCHAR)")
            staging.execute(f"INSERT INTO {relation} {_values_sql(list(_field_rows(entry)))}")
        for vocabulary, codes in CODE_LISTS.items():
            relation = f"{DICTIONARY_SCHEMA}.{vocabulary}"
            staging.execute(f"CREATE TABLE {relation} (kind_code VARCHAR, kind_text VARCHAR)")
            staging.execute(f"INSERT INTO {relation} {_values_sql(list(codes.items()))}")
    spec_path = directory / "dictionary_spec.yaml"
    spec_path.write_text(yaml.safe_dump(_spec(entries), sort_keys=False), encoding="utf-8")
    return DictionaryFiles(spec_path, staging_path)

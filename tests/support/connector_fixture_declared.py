"""A neutral warehouse for the joins and value lists a dictionary declares, with a decoy
for each rule that judges them (unit 10a, contract F1 to F8).

Each case has its own column family, so no case enters another's candidate set:
`crate_*` holds the accepted two-column join, its tie decoy, the NULL part, the misspelled
property and the inline lists; `bale_*` a referenced side whose tuples repeat; `drum_*` two
referenced relations that both hold; `pallet_*` one entity that two accepted joins bind to two
relations; `cask_*` an entity under nine properties with one misspelling; `tray_*` an entity in
no join with a stated domain. No relation declares a key: every join is the dictionary's.

The dictionary is an intake-style concept spec over a staging file with per-file entries. A
referenced entry names its entity; a referencing entry leaves its name to a header and states
the join. Every entry lists every property of its relation, as it spells them. No entity name
is a table name, so a stated list today is placed on every relation holding its column.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from sema.binding.spec import ConceptSourceSpec
from sema.ingest.duckdb_staging import Staging
from sema.intake.calls.registry import Registry
from sema.models.config import BuildConfig
from sema.pipeline.normalizer import AssertionNormalizer
from tests.support.connector_fixture_codes import DictionaryFiles
from tests.support.connector_fixture_identity import (
    IDENTITY_CATALOG,
    IDENTITY_SCHEMA,
    IdentityTable,
    Row,
    write_identity_database,
)
from tests.support.connector_fixture_identity_runs import (
    STEP_RUN_ID,
    IdentityBuild,
    StepGraph,
    StepRun,
    fixture_connector,
    identity_build_config,
    identity_model_environment,
    neo4j_config,
)

DECLARED_DICTIONARY_SCHEMA = "land"


def _relation(name: str, columns: tuple[str, ...], rows: tuple[Row, ...]) -> IdentityTable:
    """No declared key; a column is an integer when any of its values is one."""
    kinds = tuple(
        "INTEGER" if any(isinstance(row[i], int) for row in rows) else "VARCHAR"
        for i in range(len(columns))
    )
    return IdentityTable(name, tuple(zip(columns, kinds, strict=True)), rows)


LID_LIST = {"LK1": "open lid", "LK2": "shut lid"}
STRAP_LIST = {"SK1": "flat strap", "SK2": "cross strap"}
TRAY_LIST = {"TK1": "flat tray", "TK2": "deep tray", "TK3": "wide tray"}

CRATE_COLUMNS = ("crate_no", "seal_day", "crate_class", "lid_kind", "strap_kind", "wrap_kind",
                 "crate_mass", "crate_tier", "crate_bay", "crate_grade")
# The referenced side of the accepted join: one row has a NULL second part, its first unique.
# `lid_kind` and `wrap_kind` hold only listed codes; `strap_kind` holds a third code.
CRATE_HEAD = _relation("crate_head", CRATE_COLUMNS, (
    (1, 10, "A", "LK1", "SK1", "LK1", 40, 1, "B1", 70),
    (2, 10, "B", "LK2", "SK2", "LK2", 45, 1, "B1", 70),
    (2, 11, "A", "LK1", "SK3", "LK2", 50, 2, "B2", 80),
    (3, 12, "B", "LK2", "SK1", "LK1", 55, 2, "B2", 80),
    (4, 12, "A", "LK1", "SK2", "LK1", 60, 3, "B3", 90),
    (5, None, "B", "LK2", "SK3", "LK2", 65, 3, "B3", 90),
))
# The tie decoy and the placement decoy: crate_head's column set, unique pairs the child does
# not hold, and listed codes in `lid_kind`.
CRATE_HEAD_SPARE = _relation("crate_head_spare", CRATE_COLUMNS, (
    (6, 20, "A", "LK1", "SK1", "LK2", 40, 1, "B1", 70),
    (6, 21, "B", "LK2", "SK2", "LK1", 45, 1, "B1", 70),
    (7, 20, "A", "LK1", "SK1", "LK2", 50, 2, "B2", 80),
    (8, 21, "B", "LK2", "SK2", "LK1", 55, 2, "B2", 80),
))
CRATE_LINE = _relation("crate_line", ("line_no", "crate_no", "seal_day", "line_qty"), (
    (101, 1, 10, 3), (102, 1, 10, 4), (103, 2, 10, 1), (104, 2, 11, 2), (105, 3, 12, 5),
    (106, 4, 12, 6), (107, 5, None, 2),
))
# A referenced side whose pair (1, 5) repeats; the child's pairs are all held.
BALE_HEAD = _relation("bale_head", ("bale_no", "weigh_day", "bale_grade", "bale_mass"), (
    (1, 5, "A", 10), (1, 5, "B", 11), (2, 6, "A", 12), (3, 6, "B", 13),
))
BALE_LINE = _relation("bale_line", ("bale_row", "bale_no", "weigh_day", "bale_count"), (
    (201, 1, 5, 2), (202, 2, 6, 3), (203, 3, 6, 1), (204, 1, 5, 1),
))
# Two referenced relations, both unique on the pair, both holding every pair of the child.
DRUM_COLUMNS = ("drum_no", "fill_day", "drum_tone", "drum_cap")
DRUM_ROWS: tuple[Row, ...] = ((1, 3, "R", 5), (1, 4, "G", 6), (2, 3, "R", 7), (3, 4, "G", 8))
DRUM_HEAD = _relation("drum_head", DRUM_COLUMNS, DRUM_ROWS)
DRUM_HEAD_COPY = _relation("drum_head_copy", DRUM_COLUMNS, (*DRUM_ROWS, (4, 5, "R", 9)))
DRUM_LINE = _relation("drum_line", ("drum_row", "drum_no", "fill_day", "drum_litres"), (
    (301, 1, 3, 9), (302, 1, 4, 8), (303, 2, 3, 7), (304, 3, 4, 6), (305, 1, 3, 5),
))
# One entity's column set on two relations with disjoint pairs; each child holds one's pairs.
PALLET_COLUMNS = ("pallet_no", "pack_day", "pallet_wrap", "pallet_load")
PALLET_EAST = _relation("pallet_east", PALLET_COLUMNS, (
    (1, 1, "F", 10), (1, 2, "S", 11), (2, 1, "F", 12),
))
PALLET_WEST = _relation("pallet_west", PALLET_COLUMNS, (
    (3, 1, "F", 13), (3, 2, "S", 14), (4, 1, "F", 15),
))
EAST_LOAD = _relation("east_load", ("east_row", "pallet_no", "pack_day", "east_qty"), (
    (401, 1, 1, 2), (402, 1, 2, 3), (403, 2, 1, 4), (404, 1, 1, 5),
))
WEST_LOAD = _relation("west_load", ("west_row", "pallet_no", "pack_day", "west_qty"), (
    (501, 3, 1, 2), (502, 3, 2, 3), (503, 4, 1, 4), (504, 3, 1, 5),
))
CASK_HEAD = _relation("cask_head", ("cask_no", "cask_day", "cask_wood", "cask_size"), (
    (1, 1, "oak", 20), (1, 2, "ash", 30), (2, 1, "oak", 20),
))
CASK_LINE = _relation("cask_line", ("cask_row", "cask_no", "cask_day", "cask_qty"), (
    (601, 1, 1, 5), (602, 1, 2, 6), (603, 2, 1, 7),
))
# A stated domain of an entity in no join, on a column two relations hold.
TRAY_STOCK = _relation("tray_stock", ("tray_no", "tray_kind", "tray_load"), (
    (701, "TK1", 5), (702, "TK2", 6), (703, "TK1", 7), (704, "TK2", 8),
))
TRAY_RETURN = _relation("tray_return", ("return_no", "tray_kind"), (
    (801, "TK1"), (802, "TK1"), (803, "TK2"),
))

DECLARED_TABLES: tuple[IdentityTable, ...] = (
    CRATE_HEAD, CRATE_HEAD_SPARE, CRATE_LINE, BALE_HEAD, BALE_LINE, DRUM_HEAD, DRUM_HEAD_COPY,
    DRUM_LINE, PALLET_EAST, PALLET_WEST, EAST_LOAD, WEST_LOAD, CASK_HEAD, CASK_LINE,
    TRAY_STOCK, TRAY_RETURN,
)
DECLARED_BY_NAME = {table.name: table for table in DECLARED_TABLES}


@dataclass(frozen=True)
class DictionaryEntry:
    """One per-file entry: the relation it describes, its properties as the dictionary spells
    them, the entity it names (None leaves the name to its header), the entity its join refers
    to with the (local, remote) column pairs, its domains, and each property's list note."""

    relation: str
    entity: str | None
    header: str | None
    properties: tuple[str, ...]
    joins_to: str | None = None
    pairs: tuple[tuple[str, str], ...] = ()
    domains: dict[str, str] = field(default_factory=dict)
    lists: dict[str, dict[str, str]] = field(default_factory=dict)


def _entry(table: IdentityTable, entity: str, **spelled: str) -> DictionaryEntry:
    """A referenced entry naming its entity; `spelled` respells a column's property."""
    names = tuple(spelled.get(c, c) for c in table.column_names)
    return DictionaryEntry(table.name, entity, None, names)


def _child(table: IdentityTable, header: str, joins_to: str, *pairs: str) -> DictionaryEntry:
    return DictionaryEntry(table.name, None, header, table.column_names, joins_to,
                           tuple((c, c) for c in pairs))


# The misspelled property: one of ten, so the name sets overlap at 9 of 11.
MISSPELLED_PROPERTY = "Crate Clas"
# The one below the floor: one of four, so the name sets overlap at 3 of 5.
LOST_PROPERTY = "Cask Wod"
CRATE_ENTRY = replace(
    _entry(CRATE_HEAD, "Crate Record", crate_class=MISSPELLED_PROPERTY),
    lists={"lid_kind": LID_LIST, "strap_kind": STRAP_LIST, "wrap_kind": LID_LIST},
)
TRAY_ENTRY = replace(_entry(TRAY_STOCK, "Tray Holding"), domains={"tray_kind": "tray_kinds"})
DICTIONARY_ENTRIES: tuple[DictionaryEntry, ...] = (
    CRATE_ENTRY,
    _child(CRATE_LINE, "Crate Entry", "Crate Record", "crate_no", "seal_day"),
    _entry(BALE_HEAD, "Bale Record"),
    _child(BALE_LINE, "Bale Entry", "Bale Record", "bale_no", "weigh_day"),
    _entry(DRUM_HEAD, "Drum Record"),
    _child(DRUM_LINE, "Drum Entry", "Drum Record", "drum_no", "fill_day"),
    _entry(PALLET_EAST, "Pallet Record"),
    _child(EAST_LOAD, "Pallet East Entry", "Pallet Record", "pallet_no", "pack_day"),
    _child(WEST_LOAD, "Pallet West Entry", "Pallet Record", "pallet_no", "pack_day"),
    _entry(CASK_HEAD, "Cask Record", cask_wood=LOST_PROPERTY),
    _child(CASK_LINE, "Cask Entry", "Cask Record", "cask_no", "cask_day"),
    TRAY_ENTRY,
)
CODE_LIST_ROLES = {"tray_kinds": ("tray_code", "tray_text")}

# What the rules decide, by the referencing relation of each declared join.
ACCEPTED = {"crate_line": "crate_head", "east_load": "pallet_east", "west_load": "pallet_west"}
REJECTED = ("bale_line", "cask_line")
OPEN = ("drum_line",)


def entry_with(properties: Iterable[str]) -> DictionaryEntry:
    """The entry whose properties are exactly these."""
    wanted = set(properties)
    (found,) = [e for e in DICTIONARY_ENTRIES if set(e.properties) == wanted]
    return found


def list_note(codes: dict[str, str]) -> str:
    return " ".join(f"{code} = {label}" for code, label in codes.items())


def write_declared_database(directory: Path) -> Path:
    return write_identity_database(directory, DECLARED_TABLES)


def _stem(entry: DictionaryEntry) -> str:
    return f"{DECLARED_DICTIONARY_SCHEMA}.{entry.relation}"


def _spec_entry(entry: DictionaryEntry) -> dict[str, Any]:
    named = {"entity": entry.entity} if entry.entity else {"header": f"{_stem(entry)}_header"}
    joins = [{"columns": [local for local, _ in entry.pairs], "entity": entry.joins_to,
              "reference_columns": [remote for _, remote in entry.pairs]}]
    return {"relation": f"{_stem(entry)}_fields", **named, "domains": dict(entry.domains),
            **({"foreign_keys": joins} if entry.joins_to else {})}


def declared_spec() -> dict[str, Any]:
    """The concept spec in its `concept_source` form."""
    return {"concept_source": {
        "spec_version": 2, "id": "fixdecl-dictionary", "identity_system": "fixdecl",
        "release": "1", "default_provider": "staging",
        "entity_model": {"layout": "per_file",
                         "roles": {"property": "field_name", "description": "field_note"},
                         "entities": [_spec_entry(e) for e in DICTIONARY_ENTRIES]},
        "code_lists": [
            {"vocabulary_id": name, "relation": f"{DECLARED_DICTIONARY_SCHEMA}.{name}",
             "roles": {"code": code, "display": display}}
            for name, (code, display) in CODE_LIST_ROLES.items()
        ],
    }}


def _note(entry: DictionaryEntry, prop: str) -> str:
    codes = entry.lists.get(prop)
    return list_note(codes) if codes else f"the {prop} of the record"


def _values(rows: Iterable[tuple[str, str]]) -> str:
    return "VALUES " + ", ".join(
        "(" + ", ".join("'" + v.replace("'", "''") + "'" for v in row) + ")" for row in rows)


def write_declared_staging(path: Path) -> Path:
    """Each entry's field rows, each header's name row, and the code list's rows."""
    with Staging(str(path), schemas=()) as staging:
        staging.execute(f"CREATE SCHEMA {DECLARED_DICTIONARY_SCHEMA}")
        for entry in DICTIONARY_ENTRIES:
            fields = f"{_stem(entry)}_fields"
            staging.execute(f"CREATE TABLE {fields} (field_name VARCHAR, field_note VARCHAR)")
            staging.execute(f"INSERT INTO {fields} "
                            + _values((p, _note(entry, p)) for p in entry.properties))
            if entry.header:
                staging.execute(f"CREATE TABLE {_stem(entry)}_header (item VARCHAR, text VARCHAR)")
                staging.execute(f"INSERT INTO {_stem(entry)}_header "
                                + _values([("name", entry.header)]))
        for name, (code, display) in CODE_LIST_ROLES.items():
            relation = f"{DECLARED_DICTIONARY_SCHEMA}.{name}"
            staging.execute(f"CREATE TABLE {relation} ({code} VARCHAR, {display} VARCHAR)")
            staging.execute(f"INSERT INTO {relation} " + _values(TRAY_LIST.items()))
    return path


def write_declared_dictionary(directory: Path) -> DictionaryFiles:
    """The staging file and the spec that reads it, in its `concept_source` YAML form."""
    staging = write_declared_staging(directory / "declared_dictionary.duckdb")
    spec = directory / "declared_dictionary_spec.yaml"
    spec.write_text(yaml.safe_dump(declared_spec(), sort_keys=False), encoding="utf-8")
    return DictionaryFiles(spec, staging)


def write_intake_run(root: Path, *, stage: str, status: str) -> Path:
    """An intake run folder under `root`, as intake leaves one: its registry row at `stage` and
    `status`, the spec as bare JSON, and the staging file. Returns the run folder."""
    run_id = f"fixdecl-{stage}-{status}"
    folder = root / run_id
    (folder / "artifacts").mkdir(parents=True)
    write_declared_staging(folder / "staging.duckdb")
    spec = ConceptSourceSpec.model_validate(declared_spec()["concept_source"])
    (folder / "artifacts" / "spec.json").write_text(spec.model_dump_json(), encoding="utf-8")
    registry = Registry(root / "registry.sqlite")
    registry.register(run_id, source="fixdecl", stage=stage)
    registry.advance(run_id, stage, status)
    return folder


def prepare_declared_step(directory: Path) -> StepRun:
    """The fixture in DuckDB, its columns in a stand-in graph (no declared joins), and the
    build config with the dictionary, ready for one build step."""
    database = write_declared_database(directory)
    dictionary = write_declared_dictionary(directory)
    reader = fixture_connector(database)
    try:
        _, platform, workspace = reader.get_datasource_ref()
        normalizer = AssertionNormalizer(workspace, STEP_RUN_ID, platform)
        columns = [
            replace(c, ref=normalizer.normalize_column(c)[0].subject_ref)
            for table in DECLARED_TABLES
            for c in reader.extract_columns_dto(IDENTITY_CATALOG, IDENTITY_SCHEMA, table.name)
        ]
    finally:
        reader.close()
    refs = {(c.table_name, c.name): c.ref or "" for c in columns}
    return StepRun(StepGraph(columns, []), fixture_connector(database),
                   identity_build_config(database, dictionary), refs, dictionary)


def build_declared_graph(driver: Any, directory: Path) -> IdentityBuild:
    """Clear the graph, then build the fixture with its dictionary and the mocked model."""
    from sema.pipeline.orchestrate import run_build

    database = write_declared_database(directory)
    config = identity_build_config(database, write_declared_dictionary(directory), neo4j_config())
    build = IdentityBuild(directory / "declared_build.yaml", database)
    settings = config.model_dump(mode="json", include={
        "source", "catalog", "schemas", "duckdb", "skip_embeddings", "table_workers",
        "concept_spec", "concept_staging"})
    settings["neo4j"] = {"uri": config.neo4j.uri, "user": config.neo4j.user,
                         "password": config.neo4j.password.get_secret_value()}
    build.config_path.write_text(yaml.safe_dump(settings), encoding="utf-8")
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    try:
        with identity_model_environment():
            report = run_build(BuildConfig.from_file(str(build.config_path)))
    except Exception as exc:  # a build crash fails every test with its cause, never an error
        build.error = f"{type(exc).__name__}: {exc}"
        return build
    if report["tables_failed"]:
        build.error = f"tables failed: {report['failed_tables']}"
    return build

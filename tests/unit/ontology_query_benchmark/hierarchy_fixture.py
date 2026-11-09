"""A synthetic two-vocabulary warehouse in DuckDB: source members, a typed mapping
relation with one replacement chain, a standard flag and a closure with self rows."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import duckdb

from showcase.ontology_query_benchmark.profile import ConceptHierarchy, Extract, load_profile
from tests.unit.ontology_query_benchmark.conftest import PROFILE_PATH

BASE: Extract = load_profile(PROFILE_PATH).extract
ANCHOR = "C1"
HIERARCHY: dict[str, Any] = {
    "attribution": "assisted",
    "mapping": {
        "relation": "wh.voc.relationship", "source_column": "id_1",
        "target_column": "id_2", "type_column": "rel",
        "mapping_types": ["Maps to"], "replacement_types": ["Replaced by"],
        "target_vocabulary": "std",
    },
    "closure": {
        "relation": "wh.voc.ancestor", "ancestor_column": "anc",
        "descendant_column": "des", "level_column": "lvl", "self_rows": True,
    },
    "concepts": {
        "relation": "wh.voc.concept", "id_column": "cid", "name_column": "cname",
        "standard_column": "flag", "standard_values": ["S"],
    },
}
MEMBERS = ("L10", "L11", "L12", "L13", "L14", "L15")
CONCEPTS = [
    ("C1", "Neoplasm", "S"), ("C2", "Lung neoplasm", "S"),
    ("C3", "Small cell lung neoplasm", "S"), ("C4", "Old lung code", None),
    ("C5", "Lung adenocarcinoma", "S"), ("C6", "Fracture", "S"),
    ("C7", "Bronchial lesion", None), ("C8", "Unrelated", "S"), ("C9", "Unrelated child", "S"),
]
PARENT_EDGES = [("C1", "C2"), ("C2", "C3"), ("C2", "C5"), ("C2", "C7"), ("C8", "C9")]
MAPPING = [
    ("L10", "C3", "Maps to"), ("L11", "C4", "Maps to"), ("C4", "C5", "Replaced by"),
    ("L12", "C6", "Maps to"), ("L13", "C7", "Maps to"), ("L14", "C2", "Mapped from"),
    ("L99", "C8", "Maps to"),
]
WAREHOUSE_TYPES = {"varchar": "string", "integer": "int"}
EVENT_VALUES = [*MEMBERS, "L10", "L12"]


def hierarchy(**changes: Any) -> ConceptHierarchy:
    return ConceptHierarchy.model_validate({**HIERARCHY, **changes})


def spec_with(concept_hierarchy: ConceptHierarchy | None = None) -> Extract:
    return BASE.model_copy(update={"concept_hierarchy": concept_hierarchy})


def closure_rows(edges: list[tuple[str, str]], *, self_rows: bool) -> set[tuple[str, str, int]]:
    """(ancestor, descendant, shortest distance) for every path through parent edges."""
    nodes = {code for edge in edges for code in edge}
    rows = {(code, code, 0) for code in nodes} if self_rows else set()
    frontier = {(parent, child, 1) for parent, child in edges}
    while frontier:
        rows |= frontier
        seen = {(a, d) for a, d, _ in rows}
        frontier = {
            (a, child, level + 1) for a, d, level in frontier for parent, child in edges
            if parent == d and (a, child) not in seen
        }
    return rows


def warehouse(members: tuple[str, ...] = MEMBERS) -> duckdb.DuckDBPyConnection:
    """Base library relations under the neutral profile's names plus the vocabulary."""
    con = duckdb.connect()
    for catalog, schema in ((BASE.catalog, BASE.schema_name), ("wh", "voc"), ("wh", "data")):
        if catalog not in {row[0] for row in con.execute("SHOW DATABASES").fetchall()}:
            con.execute(f"ATTACH ':memory:' AS {catalog}")
        con.execute(f"CREATE SCHEMA {catalog}.{schema}")
    base = f"{BASE.catalog}.{BASE.schema_name}"
    con.execute(f"CREATE TABLE {base}.{BASE.elements.relation} "
                "(element_row_id VARCHAR, element_id VARCHAR, member_id VARCHAR)")  # fmt: skip
    con.executemany(f"INSERT INTO {base}.{BASE.elements.relation} VALUES (?, 'G1', ?)",
                    [(f"r{i}", m) for i, m in enumerate(members)])  # fmt: skip
    con.execute(f"CREATE TABLE {base}.{BASE.members.relation} (concept_id VARCHAR)")
    con.executemany(f"INSERT INTO {base}.{BASE.members.relation} VALUES (?)",
                    [(m,) for m in members])  # fmt: skip
    table(con, "wh.voc.concept", "cid VARCHAR, cname VARCHAR, flag VARCHAR", CONCEPTS)
    table(con, "wh.voc.relationship", "id_1 VARCHAR, id_2 VARCHAR, rel VARCHAR", MAPPING)
    standard_self = {(cid, cid, 0) for cid, _, flag in CONCEPTS if flag == "S"}
    closure = sorted(closure_rows(PARENT_EDGES, self_rows=True) | standard_self)
    table(con, "wh.voc.ancestor", "anc VARCHAR, des VARCHAR, lvl INTEGER", closure)
    table(con, "wh.data.events", "event_id INTEGER, dx VARCHAR", list(enumerate(EVENT_VALUES)))
    return con


def table(con: duckdb.DuckDBPyConnection, name: str, columns: str, rows: list[Any]) -> None:
    con.execute(f"CREATE TABLE {name} ({columns})")
    marks = ", ".join("?" for _ in columns.split(","))
    con.executemany(f"INSERT INTO {name} VALUES ({marks})", rows)


class WarehouseCursor:
    """Runs data statements on DuckDB; answers the warehouse catalog statements itself."""

    def __init__(self, con: duckdb.DuckDBPyConnection, *, moved: str | None = None) -> None:
        self.con = con
        self.moved = moved
        self.statements: list[str] = []
        self.table_reads: Counter[str | None] = Counter()
        self._rows: list[Any] = []

    def execute(self, sql: str, parameters: Any = None) -> None:
        self.statements.append(sql)
        self._rows = self._catalog(sql) if "information_schema" in sql else (
            self.con.execute(sql.replace("`", '"')).fetchall()
        )

    def fetchall(self) -> list[Any]:
        return self._rows

    def close(self) -> None:
        pass

    def _catalog(self, sql: str) -> list[Any]:
        catalog, schema, table = _relation_of(sql)
        if "information_schema.columns" in sql:
            described = self.con.execute(
                "SELECT table_name, column_name, lower(data_type)"
                " FROM information_schema.columns WHERE table_catalog = ? AND table_schema = ?"
                " ORDER BY table_name, ordinal_position",
                [catalog, schema],
            ).fetchall()
            return [(t, c, WAREHOUSE_TYPES.get(k, k)) for t, c, k in described
                    if table is None or t == table]  # fmt: skip
        tables = [table] if table else [BASE.elements.relation, BASE.members.relation]
        self.table_reads[table] += 1
        later = table is not None and table == self.moved and self.table_reads[table] > 1
        when = "2026-01-02T00:00:00" if later else "2026-01-01T00:00:00"
        return [(t, when, None) for t in sorted(tables)]


def _relation_of(sql: str) -> tuple[str, str, str | None]:
    catalog = sql.split(" FROM ")[1].split(".")[0]
    schema = sql.split("table_schema = '")[1].split("'")[0]
    table = sql.split("table_name = '")[1].split("'")[0] if "table_name = '" in sql else None
    return catalog, schema, table


def write_product_spec(path: Path) -> Path:
    path.write_text(json.dumps({"product_id": "events", "fields": [
        {"column_id": "dx-id", "column_name": "dx", "physical_type": "STRING",
         "strategy": "pass_through", "de_id": "G1"},
    ]}), encoding="utf-8")  # fmt: skip
    return path


SLICE_MAPPING = [r for r in MAPPING if r[0] in MEMBERS and r[2] == "Maps to"] + [MAPPING[2]]
SLICE_CONCEPTS = [c for c in CONCEPTS if c[0] not in ("C8", "C9")]
SLICE_EDGES = [e for e in PARENT_EDGES if e[0] != "C8"]


def add_hierarchy_slice(projection: Path, receipt: Path) -> None:
    """Write the synthetic slice into a projection and re-pin its receipt."""
    with duckdb.connect(str(projection)) as con:
        con.execute("CREATE SCHEMA concept_hierarchy")
        table(con, "concept_hierarchy.mapping", "id_1 VARCHAR, id_2 VARCHAR, rel VARCHAR",
               SLICE_MAPPING)  # fmt: skip
        table(con, "concept_hierarchy.concept", "cid VARCHAR, cname VARCHAR, flag VARCHAR",
               SLICE_CONCEPTS)  # fmt: skip
        table(con, "concept_hierarchy.edge", "anc VARCHAR, des VARCHAR", SLICE_EDGES)
    raw = json.loads(receipt.read_text())
    raw["duckdb"]["sha256"] = hashlib.sha256(projection.read_bytes()).hexdigest()
    receipt.write_text(json.dumps(raw))


def copy_hierarchy_slice(extracted: Path, projection: Path, receipt: Path) -> None:
    """Copy an extract's hierarchy slice into a projection and re-pin its receipt."""
    with duckdb.connect(str(projection)) as con:
        con.execute(f"ATTACH '{extracted}' AS extracted (READ_ONLY)")
        con.execute("CREATE SCHEMA concept_hierarchy")
        names = con.execute(
            "SELECT table_name FROM duckdb_tables()"
            " WHERE database_name = 'extracted' AND schema_name = 'concept_hierarchy'"
        ).fetchall()
        for (name,) in names:
            con.execute(f"CREATE TABLE concept_hierarchy.{name} AS"
                        f" SELECT * FROM extracted.concept_hierarchy.{name}")  # fmt: skip
    raw = json.loads(receipt.read_text())
    raw["duckdb"]["sha256"] = hashlib.sha256(projection.read_bytes()).hexdigest()
    receipt.write_text(json.dumps(raw))


def typed_hierarchy() -> ConceptHierarchy:
    """Over ``add_typed_vocabulary``: integer concept ids, a text mapping target column."""
    return hierarchy(
        mapping={**HIERARCHY["mapping"], "relation": "wh.num.relationship",
                 "replacement_types": [], "target_type": "INTEGER"},
        closure={**HIERARCHY["closure"], "relation": "wh.num.ancestor"},
        concepts={**HIERARCHY["concepts"], "relation": "wh.num.concept"},
    )  # fmt: skip


def add_typed_vocabulary(con: duckdb.DuckDBPyConnection) -> duckdb.DuckDBPyConnection:
    """One mapping target is not an id, as a text column can hold."""
    con.execute("CREATE SCHEMA wh.num")
    table(con, "wh.num.relationship", "id_1 VARCHAR, id_2 VARCHAR, rel VARCHAR",
          [("L10", "3", "Maps to"), ("L11", "not-an-id", "Maps to"), ("L12", "6", "Maps to")])
    table(con, "wh.num.ancestor", "anc INTEGER, des INTEGER, lvl INTEGER",
          [(1, 1, 0), (1, 3, 1), (3, 3, 0), (6, 6, 0)])
    table(con, "wh.num.concept", "cid INTEGER, cname VARCHAR, flag VARCHAR",
          [(1, "Neoplasm", "S"), (3, "Lung neoplasm", "S"), (6, "Fracture", "S")])
    return con

"""Shared setup for test_sema_hierarchy_anchor.py and the tests that reuse it."""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import duckdb

from showcase.ontology_query_benchmark.profile import Attribution
from showcase.ontology_query_benchmark.sema_bindings import (
    load_products,
    load_sidecar,
    resolve_bindings,
)
from showcase.ontology_query_benchmark.sema_build_utils import LibraryFacts, import_library_facts
from showcase.ontology_query_benchmark.sema_context import SContext, SContextProvider
from showcase.ontology_query_benchmark.sema_context_utils import nothing_denied
from tests.unit.ontology_query_benchmark.hierarchy_fixture import (
    add_hierarchy_slice,
    hierarchy,
    repin,
    table,
    warehouse,
)
from tests.unit.ontology_query_benchmark.sema_bindings_env import (
    CONCEPT_SPEC,
    CONVENTIONS,
    ELEMENTS,
    EVENTS,
    catalog_fixture,
    write_fixture,
    write_projection,
)
from tests.unit.ontology_query_benchmark.sema_context_env import EVENTS_ENTITY, LIMITS, _Retriever
from tests.unit.ontology_query_benchmark.sema_hierarchy_env import RELATIONS, SITE

READABLE = dict.fromkeys(RELATIONS, True)
DEEPER_CONCEPTS = [("C0", "Disorder", "S"), ("R", "Clinical finding", "S")]
DEEPER_EDGES = [("C0", "C1"), ("R", "C0")]


def anchor(code: str, label: str, vocabulary: str, count: int,
           column: str = "site_S1") -> dict[str, Any]:  # fmt: skip
    return {"type": "hierarchy_anchor", "code": code, "label": label,
            "vocabulary": vocabulary, "table": EVENTS, "column": column, "schema": "sch",
            "member_count": count, "member_codes": [], "status": "auto",
            "confidence": 1.0, "confidence_policy": "semantic"}  # fmt: skip


def context(tmp_path: Path, facts: LibraryFacts, readable: dict[str, bool] | None,
            anchors: list[dict[str, Any]],
            is_denied: Callable[[str], bool] = nothing_denied) -> SContext:  # fmt: skip
    sidecar = load_sidecar(write_fixture(tmp_path), tmp_path)
    products = load_products(sidecar)
    provider = SContextProvider(
        _Retriever([EVENTS_ENTITY, *anchors]), sidecar, products,
        resolve_bindings(sidecar, products, catalog_fixture()), facts,
        graph_identity="graph-1", conventions=CONVENTIONS, limits=LIMITS,
        hierarchy_relations=readable,
    )
    return provider("Which events sit under a broader concept?", is_denied)


def fragments(text: str, column: str, *, filled: bool = True) -> list[str]:
    """The template on ``column``, or with ``filled`` that template filled with each anchor
    listed under the column, in listed order."""
    lines = [x.strip() for x in text.splitlines()]
    template = next((x.removeprefix("WHERE ") for x in lines
                     if x.startswith("WHERE ") and "{column}" in x), None)  # fmt: skip
    if template is None:
        return []
    template = template.replace("{column}", column)
    if not filled:
        return [template]
    return [template.replace("{anchor_name}", name) for name in listed(lines, column)]


QUOTED = r"'(?:[^'\\]|\\.|'')*'"
LISTED = rf"anchor name ({QUOTED})(?:, vocabulary {QUOTED})?; \d+ members"


def listed(lines: list[str], column: str) -> list[str]:
    """Anchor name literals on the lines under the column's ``holds`` line."""
    start = next(i for i, x in enumerate(lines) if f".{column} holds " in x and x.endswith(":"))
    codes = []
    for x in lines[start + 1:]:
        found = re.fullmatch(LISTED, x)
        if found is None:
            break
        codes.append(found.group(1))
    return codes


def members(con: duckdb.DuckDBPyConnection, source: str, fragment: str) -> set[str]:
    found = con.execute(f"SELECT DISTINCT site_S1 FROM ({source}) WHERE {fragment}")
    return {row[0] for row in found.fetchall()}


EVENTS_SQL = "SELECT dx AS site_S1 FROM wh.data.events"


def deeper_facts(tmp_path: Path) -> LibraryFacts:
    """The synthetic slice with two more levels above its top concept: depth 4."""
    projection, receipt = write_projection(tmp_path, {**ELEMENTS, "S1": SITE})
    add_hierarchy_slice(projection, receipt)
    with duckdb.connect(str(projection)) as con:
        con.executemany("INSERT INTO concept_hierarchy.concept VALUES (?, ?, ?)", DEEPER_CONCEPTS)
        con.executemany("INSERT INTO concept_hierarchy.edge VALUES (?, ?)", DEEPER_EDGES)
    repin(projection, receipt)
    return import_library_facts(
        CONCEPT_SPEC, projection, receipt,
        spec_attribution=Attribution.ASSISTED, hierarchy=hierarchy(),
    )


def deeper_warehouse() -> duckdb.DuckDBPyConnection:
    con = warehouse()
    table(con, "wh.voc.extra", "cid VARCHAR, cname VARCHAR, flag VARCHAR", DEEPER_CONCEPTS)
    con.execute("INSERT INTO wh.voc.concept SELECT * FROM wh.voc.extra")
    below_c1 = [r[0] for r in con.execute(
        "SELECT des FROM wh.voc.ancestor WHERE anc = 'C1'").fetchall()]  # fmt: skip
    rows = [("C0", "C0", 0), ("R", "R", 0), ("R", "C0", 1)]
    rows += [(a, d, 0) for a in ("C0", "R") for d in below_c1]
    con.executemany("INSERT INTO wh.voc.ancestor VALUES (?, ?, ?)", rows)
    return con

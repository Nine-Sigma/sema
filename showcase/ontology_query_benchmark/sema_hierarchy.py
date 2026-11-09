"""A concept hierarchy for governed coded columns, from the projection's bounded slice.

The slice gives the graph its parent edges and member-to-standard bridges; executed SQL
reads the declared warehouse relations, each checked readable once at build.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import duckdb

from sema.graph.cross_layer_bridge import write_value_bridge
from sema.graph.loader import GraphLoader
from sema.log import logger
from showcase.ontology_query_benchmark.intake_extract_hierarchy import SCHEMA
from showcase.ontology_query_benchmark.profile import ConceptHierarchy

TERM_SOURCE = "target-load"
Edge = tuple[str, str]


@dataclass(frozen=True)
class HierarchyFacts:
    """``edges`` are (parent, child); ``bridges`` are (member, standard concept)."""

    spec: ConceptHierarchy
    names: Mapping[str, str]
    edges: tuple[Edge, ...]
    bridges: tuple[Edge, ...]

    def reaches(self, member: str) -> bool:
        if self.spec.mapping is None:
            return member in self.names or any(member in edge for edge in self.edges)
        return any(source == member for source, _ in self.bridges)


def load_hierarchy(projection: Path, spec: ConceptHierarchy) -> HierarchyFacts:
    with duckdb.connect(str(projection), read_only=True) as con:
        held = con.execute("SELECT 1 FROM duckdb_schemas() WHERE schema_name = ?", [SCHEMA])
        if not held.fetchall():
            raise ValueError(
                f"projection {projection} holds no {SCHEMA} slice for the profile's "
                "concept hierarchy; rerun the extract with this profile"
            )
        concepts = _rows(con, "concept")
        edges = tuple(sorted((str(a), str(d)) for a, d in _rows(con, "edge")))
        mapping = _rows(con, "mapping") if spec.mapping else []
    standard = {
        str(row[0]) for row in concepts
        if spec.concepts.standard_column is None or row[2] in spec.concepts.standard_values
    }
    names = {str(row[0]): str(row[1]) for row in concepts}
    return HierarchyFacts(spec, names, edges, _bridges(spec, mapping, standard))


def _rows(con: duckdb.DuckDBPyConnection, table: str) -> list[tuple[object, ...]]:
    return con.execute(f"SELECT * FROM {SCHEMA}.{table}").fetchall()


def _bridges(
    spec: ConceptHierarchy, mapping: list[tuple[object, ...]], standard: set[str]
) -> tuple[Edge, ...]:
    """Each member's mapping targets, then their replacements, kept when standard."""
    if spec.mapping is None:
        return ()
    typed = [(str(s), str(t), str(kind)) for s, t, kind in mapping]
    replaced: dict[str, set[str]] = {}
    for source, target, kind in typed:
        if kind in spec.mapping.replacement_types:
            replaced.setdefault(source, set()).add(target)
    return tuple(sorted({
        (source, concept)
        for source, target, kind in typed if kind in spec.mapping.mapping_types
        for concept in {target, *replaced.get(target, ())} if concept in standard
    }))  # fmt: skip


def write_hierarchy(
    loader: GraphLoader,
    facts: HierarchyFacts,
    *,
    member_vocabulary: str,
    library_codes: set[str],
    source_schema: str,
) -> dict[str, int]:
    """Terms, PARENT_OF edges and MAPS_TO_CONCEPT bridges through SEMA's graph writers.

    Without a mapping the hierarchy shares the members' vocabulary, so library member
    terms keep their library labels.
    """
    mapping = facts.spec.mapping
    vocabulary = mapping.target_vocabulary if mapping else member_vocabulary
    codes = {*facts.names, *(code for edge in facts.edges for code in edge)}
    terms = sorted(codes if mapping else codes - library_codes)
    for code in terms:
        loader.upsert_term(code, facts.names.get(code, code), source=TERM_SOURCE,
                           confidence=1.0, source_schema=source_schema,
                           vocabulary_name=vocabulary)  # fmt: skip
    for parent, child in facts.edges:
        loader.add_term_hierarchy(
            parent_code=parent, child_code=child,
            source_schema=source_schema, vocabulary_name=vocabulary,
        )
    for member, concept in facts.bridges:
        write_value_bridge(loader, source_vocabulary=member_vocabulary, source_code=member,
                           concept_vocabulary=vocabulary, concept_code=concept,
                           source_schema=source_schema)  # fmt: skip
    return {"hierarchy_terms": len(terms), "hierarchy_edges": len(facts.edges),
            "value_bridges": len(facts.bridges)}  # fmt: skip


def release_findings(spec: ConceptHierarchy) -> list[str]:
    if spec.release is None:
        return ["concept hierarchy declares no vocabulary release; its relations may mix "
                "releases and drop join rows silently"]  # fmt: skip
    return [
        f"hierarchy relation {r} is outside vocabulary release {spec.release}; "
        "joins across it are unverified"
        for r in spec.release_exempt
    ]


def check_relations(
    spec: ConceptHierarchy, read: Callable[[str], object] | None
) -> tuple[dict[str, bool], list[str]]:
    """One ``LIMIT 1`` read per declared relation; without a reader none is verified."""
    readable: dict[str, bool] = {}
    findings: list[str] = []
    for relation in spec.relations:
        try:
            if read is None:
                raise LookupError("no warehouse reader at build")
            read(f"SELECT 1 FROM {relation} LIMIT 1")
            readable[relation] = True
        except Exception as exc:  # noqa: BLE001 - any reader failure means unreadable
            logger.warning("hierarchy relation {} unreadable at build: {}", relation, exc)
            readable[relation] = False
            findings.append(
                f"insufficient evidence: hierarchy relation {relation} unreadable at build: {exc}"
            )
    return readable, findings

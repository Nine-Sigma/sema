"""A term hit that is a hierarchy anchor leads down to the columns it governs.

The fixture graph answers the anchor query by traversing its own edges, so the
tests check candidate shape and threading. The Cypher itself is checked
against Neo4j in tests/integration/test_hierarchy_anchor_integration.py.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

import pytest

from sema.graph.queries import CypherQueries
from sema.models.config import HierarchyAnchorConfig
from sema.models.context import SemanticCandidateSet
from sema.pipeline.context import prune_to_sco
from sema.pipeline.retrieval import RetrievalEngine

pytestmark = pytest.mark.unit

TermKey = tuple[str, str]
ColumnKey = tuple[str, ...]  # (table, column) or (table, column, schema)


@dataclass
class FixtureGraph:
    labels: dict[TermKey, str] = field(default_factory=dict)
    parent_of: set[tuple[TermKey, TermKey]] = field(default_factory=set)
    maps_to: set[tuple[TermKey, TermKey]] = field(default_factory=set)
    value_sets: dict[ColumnKey, set[TermKey]] = field(default_factory=dict)

    def term(self, vocab: str, code: str, label: str) -> TermKey:
        self.labels[(vocab, code)] = label
        return (vocab, code)

    def descendants(self, anchor: TermKey, depth: int) -> set[TermKey]:
        frontier, seen = {anchor}, set()
        for _ in range(depth):
            frontier = {c for p, c in self.parent_of if p in frontier} - seen
            seen |= frontier
        return seen

    def anchor_rows(self, query: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        depth = int(re.search(r"PARENT_OF\*1\.\.(\d+)", query).group(1))  # type: ignore[union-attr]
        rows = []
        for anchor in self.labels:
            vocab_ok = params["vocabulary_name"] in (None, anchor[0])
            if anchor[1] != params["code"] or not vocab_ok:
                continue
            under = self.descendants(anchor, depth)
            bridged = {s for s, t in self.maps_to if t in under}
            for (table, column, *schema), members in sorted(self.value_sets.items()):
                hit = sorted(code for _, code in members & (under | bridged))
                if hit:
                    rows.append({
                        "anchor_code": anchor[1], "anchor_label": self.labels[anchor],
                        "anchor_vocabulary": anchor[0], "column_name": column,
                        "table_name": table, "schema_name": schema[0] if schema else "s",
                        "member_count": len(hit),
                        "member_codes": hit[: params["member_cap"]],
                    })
        return rows

    def lexical_rows(self, token: str) -> list[dict[str, Any]]:
        return [
            {"node": {"code": code, "label": label, "vocabulary_name": vocab, "confidence": 0.9}}
            for (vocab, code), label in self.labels.items()
            if token in label.lower()
        ]

    def run_query(self, query: str, **params: Any) -> list[dict[str, Any]]:
        if "MATCH (t:Term) WHERE toLower(t.label) CONTAINS" in query:
            return self.lexical_rows(params["token"])
        if "(anchor:Term)-[:PARENT_OF" in query:
            return self.anchor_rows(query, params)
        return []


def engine_over(graph: FixtureGraph, config: HierarchyAnchorConfig | None = None) -> RetrievalEngine:
    engine = RetrievalEngine(driver=MagicMock(), hierarchy_anchor=config)
    engine._run_query = graph.run_query  # type: ignore[method-assign]
    return engine


def anchors(engine: RetrievalEngine, question: str) -> list[dict[str, Any]]:
    candidates = engine.retrieve(question).candidates
    return [c for c in candidates if c["type"] == "hierarchy_anchor"]


def two_vocabulary_graph() -> FixtureGraph:
    """Source codes in a column, bridged to standard concepts under one anchor."""
    g = FixtureGraph()
    anchor = g.term("STD", "S100", "Vehicle")
    car, truck = g.term("STD", "S110", "Car"), g.term("STD", "S120", "Truck")
    g.term("STD", "S900", "Bicycle")
    g.parent_of |= {(anchor, car), (anchor, truck)}
    src_car, src_van = g.term("SRC", "c1", "car"), g.term("SRC", "c2", "pickup")
    src_bike = g.term("SRC", "c3", "bike")
    g.maps_to |= {(src_car, car), (src_van, truck), (src_bike, ("STD", "S900"))}
    g.value_sets[("fleet", "kind_code")] = {src_car, src_van, src_bike}
    return g


def acris_shaped_graph() -> FixtureGraph:
    """Document types under class-code parents, as in a parent-column code table."""
    g = FixtureGraph()
    rows = (
        ("MTGE", "MORTGAGE", "MORTGAGES & INSTRUMENTS"),
        ("SAT", "SATISFACTION OF MORTGAGE", "MORTGAGES & INSTRUMENTS"),
        ("AGMT", "AGREEMENT", "MORTGAGES & INSTRUMENTS"),
        ("DEED", "DEED", "DEEDS AND OTHER CONVEYANCES"),
        ("CNFL", "CONTINUATION OF FEDERAL LIEN", "UCC AND FEDERAL LIENS"),
    )
    members = set()
    for doc_type, description, class_description in rows:
        parent = g.term("DOC_CONTROL", class_description, class_description)
        child = g.term("DOC_CONTROL", doc_type, description)
        g.parent_of.add((parent, child))
        members.add(child)
    g.value_sets[("master", "doc_type")] = members
    return g


def test_anchor_over_bridged_members_names_the_column_and_count() -> None:
    found = anchors(engine_over(two_vocabulary_graph()), "every vehicle")

    assert [(a["code"], a["vocabulary"], a["table"], a["column"]) for a in found] == [
        ("S100", "STD", "fleet", "kind_code"),
    ]
    assert found[0]["member_count"] == 2
    assert found[0]["member_codes"] == ["c1", "c2"]


def test_acris_shaped_class_anchor_resolves_to_the_document_type_column() -> None:
    found = anchors(engine_over(acris_shaped_graph()), "documents that are a kind of mortgage")

    by_anchor = {a["code"]: a for a in found}
    mortgages = by_anchor["MORTGAGES & INSTRUMENTS"]
    assert (mortgages["vocabulary"], mortgages["table"], mortgages["column"]) == (
        "DOC_CONTROL", "master", "doc_type",
    )
    assert mortgages["member_count"] == 3


def test_anchor_reaches_the_structured_context() -> None:
    candidates = engine_over(two_vocabulary_graph()).retrieve("every vehicle")
    sco = prune_to_sco(candidates)

    assert [(a.code, a.table, a.column, a.member_count) for a in sco.hierarchy_anchors] == [
        ("S100", "fleet", "kind_code", 2),
    ]


def test_member_cap_bounds_the_listed_codes_but_not_the_count() -> None:
    config = HierarchyAnchorConfig(member_cap=1)
    found = anchors(engine_over(two_vocabulary_graph(), config), "every vehicle")

    assert (found[0]["member_count"], found[0]["member_codes"]) == (2, ["c1"])


def test_depth_cap_comes_from_config() -> None:
    g = two_vocabulary_graph()
    top = g.term("STD", "S000", "Conveyance")
    g.parent_of.add((top, ("STD", "S100")))

    assert anchors(engine_over(g, HierarchyAnchorConfig(max_depth=1)), "conveyance") == []
    deep = anchors(engine_over(g, HierarchyAnchorConfig(max_depth=2)), "conveyance")
    assert [(a["code"], a["member_count"]) for a in deep] == [("S000", 2)]


def test_graph_without_hierarchy_edges_keeps_the_context_unchanged() -> None:
    g = two_vocabulary_graph()
    g.parent_of.clear()
    g.maps_to.clear()
    candidates = engine_over(g).retrieve("every vehicle")

    assert not [c for c in candidates.candidates if c["type"] == "hierarchy_anchor"]
    dumped = prune_to_sco(candidates).model_dump(mode="json")
    assert "hierarchy_anchors" not in dumped


def test_anchor_query_takes_bounds_as_parameters() -> None:
    query = CypherQueries.expand_hierarchy_anchor(max_depth=4)

    assert "PARENT_OF*1..4" in query
    assert "$member_cap" in query
    assert "MAPS_TO_CONCEPT" in query


def test_one_anchor_over_two_columns_keeps_both() -> None:
    g = two_vocabulary_graph()
    g.value_sets[("fleet_archive", "kind_code")] = {("SRC", "c1")}
    g.value_sets[("fleet", "kind_code", "t")] = {("SRC", "c2")}

    found = anchors(engine_over(g), "every vehicle")

    assert sorted((a["schema"], a["table"], a["column"], a["member_count"]) for a in found) == [
        ("s", "fleet", "kind_code", 2), ("s", "fleet_archive", "kind_code", 1),
        ("t", "fleet", "kind_code", 1),
    ]


def test_an_anchor_carries_the_score_of_the_hit_that_found_it() -> None:
    candidates = engine_over(two_vocabulary_graph()).retrieve("every vehicle").candidates

    [term] = [c for c in candidates if c["type"] == "term" and c["code"] == "S100"]
    [found] = [c for c in candidates if c["type"] == "hierarchy_anchor"]
    assert term["score"] > 0 and found["score"] == term["score"]
    [anchor] = prune_to_sco(SemanticCandidateSet(query="q", candidates=candidates)).hierarchy_anchors
    assert anchor.score == term["score"]

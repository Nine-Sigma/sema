"""Done-when 5 (N7): the graph's labels and shared entities reach the core query context.

The stand-in graph's mapping rows hold what `resolve_physical_mapping` returns: each
column's `LABELED_BY` targets with the predicates of the joins their edge names (each a
`JoinPath.join_predicates` value), and the other entities' edges that share the relation.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from sema.consumers.nl2sql.prompting import render_sections
from sema.models.context import SemanticContextObject
from sema.models.target.identity import EntityIdentity, ResolutionRule, ResolvedAttribute
from sema.pipeline.context import prune_to_sco
from sema.pipeline.retrieval import RetrievalEngine
from showcase.ontology_query_benchmark.core_s_provider import without_denied
from tests.support.connector_fixture_context import CaseEmbedder
from tests.support.connector_fixture_identity import IDENTITY_CATALOG, IDENTITY_SCHEMA
from tests.unit.pipeline.test_identity_context import (
    DIALECT,
    EXPECTED_SECTION,
    HULL_IDENTITY,
    HULL_KEY,
    QUESTION,
    SECTION,
    MappingGraph,
    context,
    fqn,
    mapping_row,
    owner_edge,
    section_lines,
    table_line,
)

pytestmark = pytest.mark.unit

RIG_JOIN = {"left_table": "rig_kind", "left_column": "rig_code", "right_table": "hull_rev",
            "right_column": "rig_type", "operator": "="}
HULL_OWNER = {"catalog": IDENTITY_CATALOG, "schema": IDENTITY_SCHEMA, "table": "hull_rev"}


def labeled(row: dict[str, Any], column: str, *labels: dict[str, Any]) -> dict[str, Any]:
    row["columns"] = [{**c, "labels": list(labels)} if c["column"] == column else c
                      for c in row["columns"]]
    return row


def label(verification: str = "verified") -> dict[str, Any]:
    return {"table": "rig_kind", "column": "rig_title", "verification": verification,
            "via": [json.dumps([RIG_JOIN])]}


def shared(verification: str = "proven", owner: dict[str, str] | None = HULL_OWNER,
           ) -> dict[str, Any]:
    return {"entity": "Hull", "key_columns": ["hull_no"], "verification": verification,
            "owner": owner}


def lines_for(*rows: dict[str, Any]) -> list[str]:
    return context(MappingGraph({"Hull": list(rows)}, ["Hull"]), QUESTION)


def test_a_verified_label_reaches_its_column_line_with_its_join() -> None:
    lines = lines_for(labeled(mapping_row("hull_rev"), "rig_type", label()))

    line = table_line(lines, "hull_rev")

    assert "rig_type (label: rig_kind.rig_title on rig_kind.rig_code = hull_rev.rig_type)" in line


def test_an_unverified_label_renders_nothing() -> None:
    lines = lines_for(labeled(mapping_row("hull_rev"), "rig_type", label("unverified")))

    assert "label:" not in "\n".join(lines)


def test_a_shared_entity_reaches_the_section() -> None:
    cargo = mapping_row("hull_cargo", shared_entities=[
        shared(), shared("open"), shared(owner=None)])

    body = section_lines(lines_for(mapping_row("hull_rev"), cargo))

    assert body == [f"{fqn('hull_cargo')}: rows belong to Hull of {fqn('hull_rev')} "
                    "through hull_no"]


def test_labels_and_shared_entities_render_the_same_whatever_order_the_graph_returns_them(
) -> None:
    class_join = {**RIG_JOIN, "left_table": "rig_class", "left_column": "class_code"}
    labels = [label(), {**label(), "table": "rig_class", "column": "class_title",
                        "via": [json.dumps([class_join]), json.dumps([RIG_JOIN])]}]
    entities = [shared(), {**shared(), "entity": "Cargo"}]

    def render(order: Any) -> list[str]:
        cargo = mapping_row("hull_cargo", shared_entities=order(entities))
        return lines_for(labeled(mapping_row("hull_rev"), "rig_type", *order(labels)), cargo)

    forward, backward = render(list), render(lambda rows: [
        {**row, "via": row["via"][::-1]} if "via" in row else row for row in rows[::-1]])

    assert "label: rig_class.class_title on" in table_line(forward, "hull_rev")
    assert len(section_lines(forward)) == 2, forward
    assert forward == backward


def test_an_unreadable_identity_states_no_entity() -> None:
    hull = mapping_row("hull_rev", **{**owner_edge(HULL_IDENTITY), "identity": "{not json"})

    assert not [line for line in section_lines(lines_for(hull)) if "several rows" in line]


def sco_for(mapping: dict[str, list[dict[str, Any]]], hits: list[str]) -> SemanticContextObject:
    engine = RetrievalEngine(driver=MappingGraph(mapping, hits), embedder=CaseEmbedder())
    return prune_to_sco(engine.retrieve(QUESTION))


# An entity the identity step creates has no HAS_PROPERTY edge: its mapping row's one column
# entry is empty.
NO_PROPERTY = [{"property": None, "column": None, "data_type": None, "semantic_type": None}]
SEED_ORDERS = [["Hull", "Vessel Revision"], ["Vessel Revision", "Hull"]]
# Names neither entity, so the seed order alone decides which candidate comes first.
PLAIN_QUESTION = "Which is the largest?"


@pytest.mark.parametrize("hits", SEED_ORDERS)
def test_a_created_entity_keeps_its_line_and_its_table_keeps_its_columns(
    hits: list[str],
) -> None:
    mapping = {
        "Hull": [mapping_row("hull_rev", columns=NO_PROPERTY, **owner_edge(HULL_IDENTITY),
                             **HULL_KEY)],
        "Vessel Revision": [mapping_row("hull_rev", **HULL_KEY)],
    }

    lines = context(MappingGraph(mapping, hits), PLAIN_QUESTION)

    assert "paint_band" in "\n".join(lines).split(SECTION)[0]
    assert EXPECTED_SECTION[3] in section_lines(lines)


@pytest.mark.parametrize("hits", SEED_ORDERS)
def test_two_l2_entities_on_one_table_still_resolve_to_one(hits: list[str]) -> None:
    mapping = {name: [mapping_row("hull_rev")] for name in hits}

    sco = sco_for(mapping, hits)

    assert [e.physical_table for e in sco.entities] == [fqn("hull_rev")]


CARGO_ORDER = (ResolvedAttribute(column="cargo_tons", rule=ResolutionRule.MAX_BY, by="cargo_no"),)
SHARED_LINE = f"{fqn('hull_cargo')}: rows belong to Hull of {fqn('hull_rev')} through hull_no"
ORDER_CLAUSE = ("; each Hull has exactly one row per cargo_no, so the greatest cargo_no picks one "
                "row of each")


def shared_with(identity: EntityIdentity) -> dict[str, list[dict[str, Any]]]:
    cargo = mapping_row("hull_cargo", shared_entities=[
        {**shared(), "identity": identity.model_dump_json()}])
    return {"Hull": [mapping_row("hull_rev"), cargo]}


@pytest.mark.parametrize(("attributes", "line"), [
    (CARGO_ORDER, SHARED_LINE + ORDER_CLAUSE),
    ((), SHARED_LINE),
])
def test_a_shared_entity_states_the_ordering_rule_its_edge_carries(
    attributes: tuple[ResolvedAttribute, ...], line: str,
) -> None:
    identity = EntityIdentity(key_columns=("hull_no",), attributes=attributes)

    body = section_lines(context(MappingGraph(shared_with(identity), ["Hull"]), QUESTION))

    assert body == [line]


def test_a_denied_ordering_column_drops_the_clause_and_keeps_the_shared_line() -> None:
    identity = EntityIdentity(key_columns=("hull_no",), attributes=CARGO_ORDER)
    sco = sco_for(shared_with(identity), ["Hull"])

    kept = without_denied(sco, lambda column: column == "cargo_no")

    assert section_lines(render_sections(kept, DIALECT)) == [SHARED_LINE]

"""edge_list_hierarchy: concept ids with a resolving parent column or edge relation, acyclic."""

from __future__ import annotations

import pytest

from tests.unit.intake.structure.shapes.shapes_env import (
    Memory,
    Source,
    base,
    binding,
    candidates,
    over,
    verify,
)

pytestmark = pytest.mark.unit

ID = "edge_list_hierarchy"
NODES = (
    "CREATE TABLE {schema}.nodes AS SELECT * FROM (VALUES ('A', 'Alpha'), ('B', 'Bravo'),"
    " ('C', 'Charlie'), ('D', 'Delta'), ('E', 'Echo'), ('F', 'Foxtrot')) t(id, label)"
)
# Three columns that each resolve into the node ids: a level table, not an edge list.
LEVELS = (
    "CREATE TABLE lv.levels AS SELECT * FROM (VALUES ('A', 'B', 'C'), ('A', 'B', 'D'),"
    " ('A', 'E', 'F')) t(top, mid, leaf)"
)
# Two columns that resolve into the node ids beside one that does not.
PAIRS = (
    "CREATE TABLE pr.pairs AS SELECT * FROM (VALUES ('B', 'A', 'seen once'),"
    " ('C', 'B', 'seen twice'), ('D', 'B', 'seen once'), ('E', 'A', 'seen twice'),"
    " ('F', 'E', 'seen once')) t(child, parent, note)"
)


def _with_slots(bindings: list[object], **slots: str) -> list[object]:
    return [b for b in bindings if b.slots.items() >= slots.items()]  # type: ignore[attr-defined]


def test_v2_parent_column_is_a_candidate_with_roots_and_depth(v2: Source) -> None:
    concepts = v2.ref("concepts")

    found = _with_slots(
        candidates(ID, v2.profiles(), v2),
        concept_relation=concepts,
        id="concept_id",
        parent="parent_id",
    )

    (only,) = found
    assert only.relations == (concepts,)
    assert (only.evidence["root_count"], only.evidence["max_depth"]) == (2, 1)


def test_parent_column_binding_does_not_span_two_relations(v2: Source) -> None:
    concepts = v2.ref("concepts")

    for found in over(candidates(ID, v2.profiles(), v2), concepts):
        if "parent" in found.slots and "edge_relation" not in found.slots:
            assert found.relations == (concepts,)


def test_cycle_in_a_parent_column_is_not_eligible(cycle: Source) -> None:
    assert candidates(ID, cycle.profiles(), cycle) == []


def test_verify_of_a_forced_cycle_fails_no_cycles(cycle: Source) -> None:
    relation = cycle.ref("cycle")
    forced = binding(ID, (relation,), concept_relation=relation, id="child", parent="parent")

    verdict = verify(ID, forced, cycle.profiles(), cycle)

    assert verdict.status == "fail"
    (failure,) = [f for f in verdict.failures if f.check == "no_cycles"]
    assert failure.relation == relation
    assert failure.evidence["cycle_count"] == 2
    assert not {"A", "B"} & {str(value) for value in failure.evidence.values()}


def test_v2_binding_verifies_and_records_the_join_floor(v2: Source) -> None:
    group = v2.profiles()
    (found,) = _with_slots(candidates(ID, group, v2), id="concept_id", parent="parent_id")

    verdict = verify(ID, found, group, v2)

    assert verdict.status == "pass"
    assert verdict.floors["JOIN_RESOLVE_FLOOR"] == base().JOIN_RESOLVE_FLOOR


def test_templates_read_no_value_while_checking_hierarchies(v2: Source, cycle: Source) -> None:
    candidates(ID, v2.profiles(), v2)
    candidates(ID, cycle.profiles(), cycle)

    assert v2.value_bearing() == [] and cycle.value_bearing() == []


def _edge_bindings(source: Source, edges: str) -> list[object]:
    return _with_slots(candidates(ID, source.profiles(), source), edge_relation=edges)


def test_three_link_relation_is_not_an_edge_relation(memory: Memory) -> None:
    source = memory("lv", NODES.format(schema="lv"), LEVELS)

    assert _edge_bindings(source, source.ref("levels")) == []


def test_verify_of_a_forced_edge_binding_over_three_link_columns_fails(memory: Memory) -> None:
    source = memory("lv", NODES.format(schema="lv"), LEVELS)
    nodes, levels = source.ref("nodes"), source.ref("levels")
    slots = {"concept_relation": nodes, "id": "id", "edge_relation": levels}
    forced = binding(ID, (nodes, levels), **slots, child="mid", parent="top")

    assert verify(ID, forced, source.profiles(), source).status == "fail"


def test_two_link_relation_with_a_non_resolving_column_is_an_edge_relation(
    memory: Memory,
) -> None:
    source = memory("pr", NODES.format(schema="pr"), PAIRS)
    nodes, pairs = source.ref("nodes"), source.ref("pairs")

    found = _with_slots(_edge_bindings(source, pairs), child="child", parent="parent")

    (only,) = found
    assert only.relations == (nodes, pairs)
    assert only.slots["concept_relation"] == nodes and only.slots["id"] == "id"

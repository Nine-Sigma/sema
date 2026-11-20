"""level_columns: an ordered list of columns, nulls only below, each prefix path one node."""

from __future__ import annotations

import pytest

from tests.unit.intake.structure.shapes.shapes_env import (
    Memory,
    Source,
    binding,
    candidates,
    over,
    verify,
)

pytestmark = pytest.mark.unit

ID = "level_columns"
HIERARCHY = "codes_a2__Hierarchy"


def test_a2_hierarchy_binds_exactly_the_ordered_levels(a2: Source) -> None:
    relation = a2.ref(HIERARCHY)

    found = over(candidates(ID, a2.profiles(), a2), relation)

    (only,) = found
    assert only.relations == (relation,)
    assert only.slots == {
        "relation": relation,
        "level_1": "l1",
        "level_2": "l2",
        "level_3": "l3",
    }


def test_a2_hierarchy_binding_verifies(a2: Source) -> None:
    group = a2.profiles()
    (found,) = over(candidates(ID, group, a2), a2.ref(HIERARCHY))

    assert verify(ID, found, group, a2).status == "pass"


def test_a_null_level_above_a_filled_level_fails_verify(a2: Source) -> None:
    relation = a2.ref(HIERARCHY)
    forced = binding(ID, (relation,), relation=relation, level_1="l1", level_2="l3", level_3="l2")

    verdict = verify(ID, forced, a2.profiles(), a2)

    assert verdict.status == "fail"
    assert {f.relation for f in verdict.failures} == {relation}


def test_a_node_with_two_prefix_paths_fails_verify(a2: Source) -> None:
    relation = a2.ref(HIERARCHY)
    forced = binding(ID, (relation,), relation=relation, level_1="l2", level_2="l1")

    assert verify(ID, forced, a2.profiles(), a2).status == "fail"


def test_level_checks_read_no_value(a2: Source) -> None:
    candidates(ID, a2.profiles(), a2)

    assert a2.value_bearing() == []


def test_levels_without_an_interior_level_are_a_flat_list(memory: Memory) -> None:
    source = memory(
        "lc",
        "CREATE TABLE lc.flat AS SELECT * FROM (VALUES ('K', 'c1', 'One'), ('K', 'c2', 'Two'),"
        " ('K', 'c3', 'Three')) t(kind, code, name)",
        "CREATE TABLE lc.pairs AS SELECT * FROM (VALUES ('a1', 'b1'), ('a2', 'b2'),"
        " ('a3', 'b3')) t(a, b)",
        "CREATE TABLE lc.tree AS SELECT * FROM (VALUES ('T', 'M1', 'x1'), ('T', 'M1', 'x2'),"
        " ('T', 'M2', 'x3'), ('T', 'M2', 'x4')) t(top, mid, leaf)",
    )
    flat = source.ref("flat")
    forced = binding(ID, (flat,), relation=flat, level_1="kind", level_2="code", level_3="name")

    found = candidates(ID, source.profiles(), source)

    assert [b.relations for b in found] == [(source.ref("tree"),)]
    verdict = verify(ID, forced, source.profiles(), source)
    assert verdict.status == "fail" and {f.relation for f in verdict.failures} == {flat}
    assert source.value_bearing() == []

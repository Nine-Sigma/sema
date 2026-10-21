"""level_columns: an ordered list of columns, nulls only below, each prefix path one node."""

from __future__ import annotations

import pytest

from tests.unit.intake.shapes.shapes_env import Source, binding, candidates, over, verify

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

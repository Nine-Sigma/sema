"""path_codes: a code column whose separator splits consistently; missing prefixes implicit."""

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

ID = "path_codes"


def _dotted(source: Source) -> object:
    found = over(candidates(ID, source.profiles(), source), source.ref("paths"))
    (only,) = [b for b in found if b.slots.get("code") == "code"]
    return only


def test_v2_dotted_codes_are_eligible_with_the_dot_separator(v2: Source) -> None:
    only = _dotted(v2)

    assert only.relations == (v2.ref("paths"),)  # type: ignore[attr-defined]
    assert only.slots["relation"] == v2.ref("paths")  # type: ignore[attr-defined]
    assert only.slots["separator"] == "."  # type: ignore[attr-defined]


def test_v2_binding_counts_one_implicit_prefix(v2: Source) -> None:
    assert _dotted(v2).evidence["implicit_count"] == 1  # type: ignore[attr-defined]


def test_v2_binding_records_the_implicit_count_and_never_a_prefix(v2: Source) -> None:
    evidence = _dotted(v2).evidence  # type: ignore[attr-defined]

    assert evidence["implicit_count"] == 1
    assert not {"1", "1.1", "1.2", "1.2.3"} & {str(value) for value in evidence.values()}


def test_codes_without_a_separator_are_not_path_codes(v2: Source) -> None:
    found = over(candidates(ID, v2.profiles(), v2), v2.ref("level_concepts"))

    assert [b for b in found if b.slots.get("code") == "code"] == []


def test_v2_binding_verifies(v2: Source) -> None:
    assert verify(ID, _dotted(v2), v2.profiles(), v2).status == "pass"


def test_verify_fails_for_a_separator_the_codes_do_not_use(v2: Source) -> None:
    relation = v2.ref("paths")
    forced = binding(ID, (relation,), relation=relation, code="code", separator="/")

    verdict = verify(ID, forced, v2.profiles(), v2)

    assert verdict.status == "fail"
    assert {f.relation for f in verdict.failures} == {relation}


def test_path_checks_read_no_value(v2: Source) -> None:
    candidates(ID, v2.profiles(), v2)

    assert v2.value_bearing() == []


def test_codes_holding_whitespace_are_not_path_codes(memory: Memory) -> None:
    source = memory(
        "pc",
        "CREATE TABLE pc.spaced AS SELECT * FROM (VALUES ('one a'), ('one a.two b'),"
        " ('three c')) t(code)",
        "CREATE TABLE pc.dotted AS SELECT * FROM (VALUES ('1'), ('1.2.3'), ('2')) t(code)",
    )
    spaced = source.ref("spaced")
    forced = binding(ID, (spaced,), relation=spaced, code="code", separator=".")

    found = candidates(ID, source.profiles(), source)
    verdict = verify(ID, forced, source.profiles(), source)

    assert over(found, spaced) == []
    assert [b.slots["separator"] for b in over(found, source.ref("dotted"))] == ["."]
    assert verdict.status == "fail" and {f.relation for f in verdict.failures} == {spaced}
    assert source.value_bearing() == []

"""path_codes: a code column whose separator splits consistently; missing prefixes implicit."""

from __future__ import annotations

import pytest

from tests.unit.intake.shapes.shapes_env import Source, binding, candidates, over, verify

pytestmark = pytest.mark.unit

ID = "path_codes"
IMPLICIT_IS_A_VALUE = (
    "the spec records the missing prefix 1.2 on the binding, but the prefix is a value of"
    " the code column and every template predicate reads aggregates only (and task 9.1 wants"
    " zero value-bearing structure statements under metadata_only); the binding can carry the"
    " implicit prefix count, not the prefix itself, unless a value-bearing read is allowed"
)


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


@pytest.mark.xfail(strict=True, reason=IMPLICIT_IS_A_VALUE)
def test_v2_binding_names_the_implicit_prefix(v2: Source) -> None:
    assert str(_dotted(v2).evidence["implicit"]).split(",") == ["1.2"]  # type: ignore[attr-defined]


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

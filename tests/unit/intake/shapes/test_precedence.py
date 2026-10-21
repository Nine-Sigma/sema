"""Candidate precedence: multi-relation bindings claim relations; the rest is ambiguity."""

from __future__ import annotations

import pytest

from tests.unit.intake.shapes.shapes_env import (
    Memory,
    Source,
    binding,
    candidates,
    every_template_candidates,
    over,
    shapes,
)

pytestmark = pytest.mark.unit

CONCEPTS, VALUES = "codes_a2__Concepts", "codes_a2__Values"


def _loader_fixed_namespace(memory: Memory) -> Source:
    from sema.intake.loaders.athena import REQUIRED_TABLES

    empty = [f"CREATE TABLE ath.{name} (c VARCHAR)" for name in REQUIRED_TABLES]
    extra = (
        "CREATE TABLE ath.extra AS SELECT * FROM (VALUES ('A1', 'Alpha'), ('A2', 'Beta'),"
        " ('A3', 'Gamma')) t(code, label)"
    )
    return memory("ath", *empty, extra)


def _a2_claim(a2: Source) -> object:
    pairs = candidates("concept_and_value_set", a2.profiles(), a2)
    return next(
        b
        for b in pairs
        if set(b.relations) == {a2.ref(CONCEPTS), a2.ref(VALUES)}
        and b.slots["concept_key"] == b.slots["value_key"] == "concept_code"
    )


def test_a2_concept_sheet_claimed_by_concept_and_value_set_drops_concept_list(
    a2: Source,
) -> None:
    claim = _a2_claim(a2)
    single = candidates("concept_list", a2.profiles(), a2)
    assert over(single, a2.ref(CONCEPTS))

    result = shapes().apply_precedence([*single, claim])

    assert claim in result.candidates
    listed = [b for b in result.candidates if b.template == "concept_list"]
    assert over(listed, a2.ref(CONCEPTS)) == []
    assert all(a.subject != a2.ref(CONCEPTS) for a in result.ambiguities)
    assert result.ambiguities == ()


def test_two_multi_relation_claims_over_one_relation_are_ambiguous() -> None:
    first = binding(
        "concept_and_value_set",
        ("s.a", "s.b"),
        concept_relation="s.a",
        value_relation="s.b",
        concept_key="k",
        value_key="k",
    )
    second = binding(
        "entity_model", ("s.b", "s.c"), layout="per_file", dictionary="s.b", header="s.c"
    )

    result = shapes().apply_precedence([first, second])

    (ambiguity,) = [a for a in result.ambiguities if a.subject == "s.b"]
    assert ambiguity.stage == "structure"
    assert len(ambiguity.members) == 2
    assert any("concept_and_value_set" in m for m in ambiguity.members)
    assert any("entity_model" in m for m in ambiguity.members)


def test_precedence_does_not_depend_on_candidate_order() -> None:
    first = binding("semsql", ("s.a", "s.b"))
    second = binding("athena_bundle", ("s.b", "s.c"))

    forward = shapes().apply_precedence([first, second])
    backward = shapes().apply_precedence([second, first])

    assert {a.subject for a in forward.ambiguities} == {a.subject for a in backward.ambiguities}
    assert {a.subject for a in forward.ambiguities} == {"s.b"}


def test_unclaimed_relation_with_two_single_relation_candidates_is_ambiguous() -> None:
    listed = binding("concept_list", ("s.r",), relation="s.r")
    levels = binding("level_columns", ("s.r",), relation="s.r", level_1="x", level_2="y")

    result = shapes().apply_precedence([listed, levels])

    (ambiguity,) = result.ambiguities
    assert ambiguity.subject == "s.r" and len(ambiguity.members) == 2


def test_parent_column_hierarchy_is_single_relation_and_does_not_claim() -> None:
    listed = binding("concept_list", ("s.r",), relation="s.r")
    parent = binding("edge_list_hierarchy", ("s.r",), concept_relation="s.r", id="i", parent="p")

    result = shapes().apply_precedence([listed, parent])

    (ambiguity,) = result.ambiguities
    assert ambiguity.subject == "s.r" and len(ambiguity.members) == 2


def test_one_template_with_two_slot_bindings_is_ambiguous() -> None:
    by_x = binding("concept_with_values_denormalized", ("s.r",), relation="s.r", group_by="x")
    by_y = binding("concept_with_values_denormalized", ("s.r",), relation="s.r", group_by="y")

    result = shapes().apply_precedence([by_x, by_y])

    (ambiguity,) = result.ambiguities
    assert ambiguity.subject == "s.r" and len(ambiguity.members) == 2


def test_a_lone_single_relation_candidate_is_not_ambiguous(acris: Source) -> None:
    (listed,) = candidates("concept_list", acris.group("Country_Codes"), acris)

    result = shapes().apply_precedence([listed])

    assert result.ambiguities == ()
    assert result.candidates == (listed,)


def test_a_loader_fixed_binding_claims_every_relation_of_its_group(memory: Memory) -> None:
    source = _loader_fixed_namespace(memory)
    group = source.profiles()
    assert len(candidates("athena_bundle", group, source)) == 1
    assert over(candidates("concept_list", group, source), source.ref("extra"))

    result = shapes().apply_precedence(every_template_candidates(group, source))

    assert [b.template for b in result.candidates] == ["athena_bundle"]
    assert result.ambiguities == ()

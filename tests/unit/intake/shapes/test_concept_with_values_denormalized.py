"""concept_with_values_denormalized: one relation, a group_by column, codes unique per group."""

from __future__ import annotations

import pytest

from tests.unit.intake.shapes.shapes_env import (
    Source,
    candidates,
    every_template_candidates,
    over,
    shapes,
    verify,
)

pytestmark = pytest.mark.unit

ID = "concept_with_values_denormalized"
DOCUMENT_CONTROL = "Document_Control_Codes"
SYN_GROUP_BY = "data_element_id"
SYN_EQUIVALENT_COLUMNS = (
    "data_element_label, data_element_title, data_element_definition and"
    " data_element_synonyms each hold 4 distinct non-null values over 9 rows (ratio 0.44),"
    " one per data element id, and value_id is unique, so the rule as written makes each a"
    " group_by binding; no aggregate separates the id from its functionally equivalent"
    " label columns without reading a name"
)


def _group_by(bindings: list[object]) -> set[str]:
    return {b.slots["group_by"] for b in bindings}  # type: ignore[attr-defined]


def test_synthetic_data_element_id_is_a_group_by_candidate(syn: Source) -> None:
    element = syn.ref("data_element")

    found = over(candidates(ID, syn.profiles(), syn), element)

    assert SYN_GROUP_BY in _group_by(found)
    assert all(b.relations == (element,) and b.slots["relation"] == element for b in found)


@pytest.mark.xfail(strict=True, reason=SYN_EQUIVALENT_COLUMNS)
def test_synthetic_data_element_id_is_the_only_group_by_candidate(syn: Source) -> None:
    found = over(candidates(ID, syn.profiles(), syn), syn.ref("data_element"))

    assert _group_by(found) == {SYN_GROUP_BY}


def test_constant_low_cardinality_column_is_not_a_group_key(acris: Source) -> None:
    for name in ("Country_Codes", "State_Codes", "UCC_Collateral_Codes"):
        assert candidates(ID, acris.group(name), acris) == [], name


# Spec scenario "Grouped code table is structurally ambiguous" (amended 2026-10-21): the
# group columns determine only the constant column and the code is global, so no group is
# an entity.
def test_grouped_code_table_has_no_group_by_binding(acris: Source) -> None:
    assert candidates(ID, acris.group(DOCUMENT_CONTROL), acris) == []


# Same scenario: with no denormalized member, the concept list claim is uncontested.
def test_grouped_code_table_concept_list_is_not_contested_by_a_group(acris: Source) -> None:
    group = acris.group(DOCUMENT_CONTROL)
    ref = acris.ref(DOCUMENT_CONTROL)
    claims = candidates("concept_list", group, acris) + candidates(ID, group, acris)

    result = shapes().apply_precedence(claims)

    assert [a for a in result.ambiguities if a.subject == ref] == []
    assert [b.template for b in over(result.candidates, ref)] == ["concept_list"]


# Same scenario: members are concept_list and entity_model, as for the other code tables.
def test_every_template_leaves_document_control_with_its_list_and_model_members(
    acris: Source,
) -> None:
    group = acris.group(DOCUMENT_CONTROL)
    expected = [
        *candidates("concept_list", group, acris),
        *candidates("entity_model", group, acris),
    ]

    result = shapes().apply_precedence(every_template_candidates(group, acris))

    (ambiguity,) = [a for a in result.ambiguities if a.subject == acris.ref(DOCUMENT_CONTROL)]
    member = shapes("precedence").member
    assert ambiguity.members == tuple(sorted(member(b) for b in expected))
    assert len(ambiguity.members) == 6  # S5 decision 23: four single_relation readings
    assert over(result.candidates, acris.ref(DOCUMENT_CONTROL)) == []


# Scenario "Synthetic data element relation": the old loop ran over Document_Control
# bindings, which no longer exist, so it moves to a binding that holds.
def test_verify_passes_and_records_the_group_ratio_floor(syn: Source) -> None:
    element = syn.ref("data_element")
    found = [
        b for b in over(candidates(ID, syn.profiles(), syn), element)
        if b.slots["group_by"] == SYN_GROUP_BY
    ]

    assert found
    for kept in found:
        verdict = verify(ID, kept, syn.profiles(), syn)
        assert verdict.status == "pass"
        floor = shapes(ID).DENORMALIZED_GROUP_RATIO_FLOOR
        assert verdict.floors["DENORMALIZED_GROUP_RATIO_FLOOR"] == floor

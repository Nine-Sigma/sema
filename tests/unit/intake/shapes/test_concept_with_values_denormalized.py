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
ALL_TEMPLATES_ON_DOCUMENT_CONTROL = (
    "under the rules as written entity_model per_file (DOC. TYPE is a unique non-null text"
    " column beside other text columns) and path_codes (the unique DOC. TYPE DESCRIPTION"
    " splits into non-empty segments on '-' and on '/') are also eligible, so the ambiguity"
    " has six members; level_columns fails here (a PARTY1 TYPE value sits under more than"
    " one CLASS CODE DESCRIPTION) but holds on the other four code tables"
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


def test_grouped_code_table_has_one_binding_per_non_null_group_column(acris: Source) -> None:
    found = candidates(ID, acris.group(DOCUMENT_CONTROL), acris)

    assert _group_by(found) == {"CLASS CODE DESCRIPTION", "PARTY1 TYPE"}
    assert len(found) == 2


def test_grouped_code_table_is_a_three_member_structure_ambiguity(acris: Source) -> None:
    group = acris.group(DOCUMENT_CONTROL)
    ref = acris.ref(DOCUMENT_CONTROL)
    claims = candidates("concept_list", group, acris) + candidates(ID, group, acris)

    result = shapes().apply_precedence(claims)

    (ambiguity,) = [a for a in result.ambiguities if a.subject == ref]
    assert ambiguity.stage == "structure"
    assert len(ambiguity.members) == 3
    assert sum(ID in member for member in ambiguity.members) == 2
    assert sum("concept_list" in member for member in ambiguity.members) == 1


@pytest.mark.xfail(strict=True, reason=ALL_TEMPLATES_ON_DOCUMENT_CONTROL)
def test_every_template_leaves_document_control_with_exactly_three_members(
    acris: Source,
) -> None:
    group = acris.group(DOCUMENT_CONTROL)

    result = shapes().apply_precedence(every_template_candidates(group, acris))

    (ambiguity,) = [a for a in result.ambiguities if a.subject == acris.ref(DOCUMENT_CONTROL)]
    assert len(ambiguity.members) == 3


def test_verify_passes_and_records_the_group_ratio_floor(acris: Source) -> None:
    group = acris.group(DOCUMENT_CONTROL)
    module = shapes(ID)

    for found in candidates(ID, group, acris):
        verdict = verify(ID, found, group, acris)
        assert verdict.status == "pass"
        floor = module.DENORMALIZED_GROUP_RATIO_FLOOR
        assert verdict.floors["DENORMALIZED_GROUP_RATIO_FLOOR"] == floor

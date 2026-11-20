"""concept_list: one relation with an id signature and a label signature."""

from __future__ import annotations

import pytest

from tests.unit.intake.structure.shapes.shapes_env import Source, base, binding, candidates, over, verify

pytestmark = pytest.mark.unit

ID = "concept_list"
CODE_TABLES = (
    "Country_Codes",
    "Document_Control_Codes",
    "Property_Types_Codes",
    "State_Codes",
    "UCC_Collateral_Codes",
)


def test_code_table_is_eligible_with_one_binding_and_section_unresolved(acris: Source) -> None:
    country = acris.group("Country_Codes")
    ref = acris.ref("Country_Codes")

    found = candidates(ID, country, acris)

    assert len(found) == 1
    (only,) = found
    assert (only.template, only.section, only.relations) == (ID, None, (ref,))
    assert only.slots == {"relation": ref}


def test_every_code_table_is_eligible_once(acris: Source) -> None:
    for name in CODE_TABLES:
        assert len(candidates(ID, acris.group(name), acris)) == 1, name


def test_constant_column_is_in_neither_signature(acris: Source) -> None:
    for name in CODE_TABLES:
        profile = acris.profile(name)
        assert "RECORD TYPE" not in base().id_columns(profile), name
        assert "RECORD TYPE" not in base().label_columns(profile), name


def test_relation_without_a_unique_column_is_not_eligible(a2: Source) -> None:
    hierarchy = a2.ref("codes_a2__Hierarchy")

    assert over(candidates(ID, a2.profiles(), a2), hierarchy) == []


def test_a2_concept_and_value_sheets_are_eligible_alone(a2: Source) -> None:
    found = candidates(ID, a2.profiles(), a2)

    assert {b.relations for b in found} == {
        (a2.ref("codes_a2__Concepts"),),
        (a2.ref("codes_a2__Values"),),
    }


def test_verify_passes_and_records_the_signature_floors(acris: Source) -> None:
    group = acris.group("State_Codes")
    (found,) = candidates(ID, group, acris)

    verdict = verify(ID, found, group, acris)

    assert verdict.status == "pass"
    assert verdict.floors["ID_DISTINCT_RATIO_FLOOR"] == base().ID_DISTINCT_RATIO_FLOOR
    assert verdict.floors["LABEL_MEAN_LENGTH_BOUNDS"] == tuple(base().LABEL_MEAN_LENGTH_BOUNDS)


def test_verify_of_a_forced_binding_names_the_failing_check(a2: Source) -> None:
    group = a2.profiles()
    hierarchy = a2.ref("codes_a2__Hierarchy")
    forced = binding(ID, (hierarchy,), relation=hierarchy)

    verdict = verify(ID, forced, group, a2)

    assert verdict.status == "fail"
    assert verdict.failures and all(f.check for f in verdict.failures)
    assert {f.relation for f in verdict.failures} == {hierarchy}

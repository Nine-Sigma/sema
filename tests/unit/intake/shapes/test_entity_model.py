"""entity_model: per_file dictionaries with an optional header, or one single relation."""

from __future__ import annotations

import pytest

from tests.unit.intake.shapes.shapes_env import Source, base, binding, candidates, shapes, verify

pytestmark = pytest.mark.unit

ID = "entity_model"
WORKBOOK = "ACRIS___Real_Property_Legals_Data_Dictionary"
OMOP = "OMOP_CDMv5_4_Field_Level"
PER_FILE_ON_CODE_TABLES = (
    "the per_file rule as written (a unique non-null text column and at least one other text"
    " column, header optional) holds on every code table: DOC. TYPE or COUNTRY CODE is unique"
    " and non-null beside a text DESCRIPTION; global uniqueness of the property candidates"
    " only rules out single_relation, so the scenario needs a further per_file predicate"
    " (for example a required header relation) that the spec does not state"
)


def _workbook_group(acris: Source, stem: str) -> list[object]:
    return acris.group(f"{stem}__Column_Info", f"{stem}__Dataset_Info")


def _layouts(bindings: list[object]) -> set[str]:
    return {b.slots["layout"] for b in bindings}  # type: ignore[attr-defined]


def test_acris_workbook_group_is_per_file_with_the_key_value_sheet_as_header(
    acris: Source,
) -> None:
    dictionary = acris.ref(f"{WORKBOOK}__Column_Info")
    header = acris.ref(f"{WORKBOOK}__Dataset_Info")

    found = candidates(ID, _workbook_group(acris, WORKBOOK), acris)

    (only,) = found
    assert only.slots == {"layout": "per_file", "dictionary": dictionary, "header": header}
    assert set(only.relations) == {dictionary, header}


def test_every_acris_workbook_group_has_exactly_one_per_file_binding(acris: Source) -> None:
    stems = [p.stem for p in acris.sheet("Column_Info")]

    assert len(stems) == 15
    for stem in stems:
        (only,) = candidates(ID, _workbook_group(acris, stem), acris)
        assert only.slots["header"] == acris.ref(f"{stem}__Dataset_Info"), stem
        assert only.slots["dictionary"] == acris.ref(f"{stem}__Column_Info"), stem


def test_key_value_sheet_alone_is_a_dictionary_candidate(acris: Source) -> None:
    alone = acris.group(f"{WORKBOOK}__Dataset_Info")

    (only,) = candidates(ID, alone, acris)

    assert only.slots["dictionary"] == acris.ref(f"{WORKBOOK}__Dataset_Info")
    assert "header" not in only.slots or not only.slots["header"]


def test_omop_csv_is_single_relation_with_one_entity_and_property_column(omop: Source) -> None:
    relation = omop.ref(OMOP)

    (only,) = candidates(ID, omop.profiles(), omop)

    assert only.slots == {
        "layout": "single_relation",
        "relation": relation,
        "entity": "cdmTableName",
        "property": "cdmFieldName",
    }
    assert only.relations == (relation,)


def test_code_tables_have_no_single_relation_binding(acris: Source) -> None:
    for name in ("Document_Control_Codes", "Country_Codes"):
        found = candidates(ID, acris.group(name), acris)
        assert "single_relation" not in _layouts(found), name


@pytest.mark.xfail(strict=True, reason=PER_FILE_ON_CODE_TABLES)
def test_code_tables_are_not_dictionaries_in_either_layout(acris: Source) -> None:
    for name in ("Document_Control_Codes", "Country_Codes"):
        assert candidates(ID, acris.group(name), acris) == [], name


def test_verify_passes_and_records_the_entity_cap(omop: Source) -> None:
    group = omop.profiles()
    (found,) = candidates(ID, group, omop)

    verdict = verify(ID, found, group, omop)

    assert verdict.status == "pass"
    assert verdict.floors["ENTITY_DISTINCT_RATIO_CAP"] == shapes(ID).ENTITY_DISTINCT_RATIO_CAP
    assert verdict.floors["ID_DISTINCT_RATIO_FLOOR"] == base().ID_DISTINCT_RATIO_FLOOR


def test_verify_fails_a_header_that_is_not_a_key_value_relation(acris: Source) -> None:
    group = _workbook_group(acris, WORKBOOK)
    dictionary = acris.ref(f"{WORKBOOK}__Dataset_Info")
    header = acris.ref(f"{WORKBOOK}__Column_Info")
    forced = binding(
        ID, (dictionary, header), layout="per_file", dictionary=dictionary, header=header
    )

    verdict = verify(ID, forced, group, acris)

    assert verdict.status == "fail"
    assert header in {f.relation for f in verdict.failures}


def test_verify_fails_a_single_relation_entity_column_above_the_cap(omop: Source) -> None:
    relation = omop.ref(OMOP)
    forced = binding(
        ID,
        (relation,),
        layout="single_relation",
        relation=relation,
        entity="cdmFieldName",
        property="cdmTableName",
    )

    assert verify(ID, forced, omop.profiles(), omop).status == "fail"

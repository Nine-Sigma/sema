"""Done-when 4 (N6): where a coded column's label lives, on the loaded relations.

The coded-column step runs over the identity fixture in DuckDB, through the real connector,
with the fixture's dictionary; only the graph is a stand-in (`StepGraph`). A label is read
back as the step writes it: a `LABELED_BY` edge from the coded column to a text column, with
`via` naming the join, and the labels of the column's value set members.
"""

from __future__ import annotations

import pytest

from sema.pipeline.coded_columns import run_coded_columns
from sema.pipeline.dictionary_import_utils import read_dictionary
from tests.support.connector_fixture_identity import (
    CODE_LIST_ROLES,
    DOCK,
    HATCH_LIST_TEXTS,
    IDENTITY_SCHEMA,
    RIG_LIST_TEXTS,
    RIG_TITLES,
    SLIP_REFERENCE,
    column_values,
)
from tests.support.connector_fixture_identity_runs import (
    STEP_RUN_ID,
    StepRun,
    prepare_step,
    run_step,
)

pytestmark = pytest.mark.unit

VERIFIED = "verified"


def _coded_step(prepared: StepRun) -> None:
    run_coded_columns(prepared.graph, prepared.connector, prepared.config,  # type: ignore[arg-type]
                      [IDENTITY_SCHEMA], STEP_RUN_ID)


@pytest.fixture(scope="module")
def coded(tmp_path_factory: pytest.TempPathFactory) -> StepRun:
    return run_step(prepare_step(tmp_path_factory.mktemp("labels")), _coded_step)


def verified_labels(step: StepRun, table: str, column: str) -> list[tuple[str, dict[str, object]]]:
    ref = step.refs[(table, column)]
    return [(label, edge) for label, edge in step.built().labeled_by.get(ref, [])
            if edge.get("verification") == VERIFIED]


def binding(step: StepRun, table: str, column: str) -> dict[str, object]:
    return step.built().edges.get(step.refs[(table, column)], {})


def test_a_code_list_that_is_also_a_loaded_relation_labels_from_that_relation(
    coded: StepRun,
) -> None:
    assert binding(coded, "hull_rev", "rig_type").get("verification") == VERIFIED

    labels = verified_labels(coded, "hull_rev", "rig_type")

    assert [label for label, _ in labels] == [coded.refs[("rig_kind", "rig_title")]]
    ((_, edge),) = labels
    assert edge["via"], "a label in another relation names the join it is reached by"


def test_the_display_role_column_wins_and_the_constant_column_is_dropped(
    coded: StepRun,
) -> None:
    (label, _), = verified_labels(coded, "hull_rev", "rig_type")

    assert label != coded.refs[("rig_kind", "kind_group")]
    assert label != coded.refs[("rig_kind", "rig_note")]
    assert label == coded.refs[("rig_kind", CODE_LIST_ROLES["rig_kinds"][1])]


def test_the_governed_value_labels_are_the_loaded_relations_texts(coded: StepRun) -> None:
    present = {str(v) for v in column_values("hull_rev", "rig_type")}

    labels = coded.built().labels(coded.refs[("hull_rev", "rig_type")])

    assert {code: labels.get(code) for code in present} == {c: RIG_TITLES[c] for c in present}
    assert not set(labels.values()) & set(RIG_LIST_TEXTS.values())


def test_a_dictionary_only_code_list_keeps_its_labels_and_no_edge(coded: StepRun) -> None:
    assert binding(coded, "hull_cargo", "hatch_kind").get("verification") == VERIFIED

    labels = coded.built().labels(coded.refs[("hull_cargo", "hatch_kind")])

    assert verified_labels(coded, "hull_cargo", "hatch_kind") == []
    present = {str(v) for v in column_values("hull_cargo", "hatch_kind")}
    assert {code: labels.get(code) for code in present} == {
        c: HATCH_LIST_TEXTS[c] for c in present}


def test_two_non_constant_texts_and_no_display_role_give_no_label(coded: StepRun) -> None:
    assert binding(coded, "berth", "dock_code").get("verification") == VERIFIED
    assert all(len({t[i] for t in DOCK.values()}) > 1 for i in (0, 1))

    assert verified_labels(coded, "berth", "dock_code") == []
    assert verified_labels(coded, "hull_rev", "rig_type"), "the positive case labels"


def test_read_dictionary_returns_the_roles_and_the_reference_column_lists(
    coded: StepRun,
) -> None:
    files = coded.dictionary

    dictionary = read_dictionary(str(files.spec), str(files.staging))

    assert dictionary is not None
    assert dictionary.roles == CODE_LIST_ROLES
    assert dictionary.reference_keys == [SLIP_REFERENCE]

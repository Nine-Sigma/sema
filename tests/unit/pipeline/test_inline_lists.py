"""Done-when 4 (N5): inline value lists placed and decided by the coded-column step.

The step runs over the declared-join fixture in DuckDB, through the real connector, with the
fixture's dictionary; only the graph is a stand-in (`StepGraph`). Each list a field note
states is a code list of its entity's property: a list the data holds links its column and
gives its labels, one that misses an observed value links nothing, and the list of an entity
the dictionary's join binds links only the bound relation's column.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from sema.engine.coded_column_utils import STATED_DOMAIN
from sema.pipeline.coded_columns import run_coded_columns
from tests.support import connector_fixture_declared as declared
from tests.support.connector_fixture_declared import LID_LIST, TRAY_LIST, prepare_declared_step
from tests.support.connector_fixture_identity import IDENTITY_SCHEMA
from tests.support.connector_fixture_identity_runs import STEP_RUN_ID, StepRun, run_step

pytestmark = pytest.mark.unit

VERIFIED = "verified"


def _coded_step(prepared: StepRun) -> None:
    run_coded_columns(prepared.graph, prepared.connector, prepared.config,  # type: ignore[arg-type]
                      [IDENTITY_SCHEMA], STEP_RUN_ID)


@pytest.fixture(scope="module")
def coded(tmp_path_factory: pytest.TempPathFactory) -> StepRun:
    return run_step(prepare_declared_step(tmp_path_factory.mktemp("inline")), _coded_step)


def _ref(step: StepRun, table: str, column: str) -> str:
    return step.refs[(table, column)]


def _verification(step: StepRun, table: str, column: str) -> object:
    return step.built().edges.get(_ref(step, table, column), {}).get("verification")


def _vocabularies(step: StepRun, table: str, column: str) -> set[str]:
    return {vocabulary for vocabulary, _ in step.built().members.get(_ref(step, table, column), ())}


def test_a_list_the_data_holds_links_its_column_and_gives_its_labels(coded: StepRun) -> None:
    assert _verification(coded, "crate_head", "lid_kind") == VERIFIED

    labels = coded.built().labels(_ref(coded, "crate_head", "lid_kind"))

    assert labels == LID_LIST


def test_a_list_that_misses_an_observed_value_links_nothing_and_is_reported(
    coded: StepRun,
) -> None:
    ref = _ref(coded, "crate_head", "strap_kind")

    assert ref not in coded.built().edges
    reported = [a.payload for a in coded.built().assertions if a.subject_ref == ref]
    assert any(p.get("holds") is False and p.get("linked") is False for p in reported), reported


def test_equal_lists_on_two_properties_stay_two_lists(coded: StepRun) -> None:
    lid, wrap = (_vocabularies(coded, "crate_head", c) for c in ("lid_kind", "wrap_kind"))

    assert _verification(coded, "crate_head", "lid_kind") == VERIFIED
    assert _verification(coded, "crate_head", "wrap_kind") == VERIFIED
    assert len(lid) == len(wrap) == 1
    assert lid != wrap


def test_a_bound_entitys_list_links_only_the_bound_relations_column(coded: StepRun) -> None:
    spare = _ref(coded, "crate_head_spare", "lid_kind")

    assert _verification(coded, "crate_head", "lid_kind") == VERIFIED
    assert spare not in coded.built().edges
    assert not [a for a in coded.built().assertions
                if a.subject_ref == spare and a.payload.get("linked")]


def test_an_unbound_entitys_stated_domain_is_placed_as_today(coded: StepRun) -> None:
    for table in ("tray_stock", "tray_return"):
        assert _verification(coded, table, "tray_kind") == VERIFIED
        labels = coded.built().labels(_ref(coded, table, "tray_kind"))
        assert labels == TRAY_LIST


@pytest.fixture(scope="module")
def stated_on_bound(tmp_path_factory: pytest.TempPathFactory) -> StepRun:
    """The fixture with a spec domain on the bound entity's `crate_bay`, a column the
    unbound decoy relation holds too."""
    entry = replace(declared.CRATE_ENTRY, domains={"crate_bay": "tray_kinds"})
    entries = tuple(entry if e is declared.CRATE_ENTRY else e for e in declared.DICTIONARY_ENTRIES)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(declared, "DICTIONARY_ENTRIES", entries)
        prepared = prepare_declared_step(tmp_path_factory.mktemp("stated"))
    return run_step(prepared, _coded_step)


def _stated(step: StepRun, table: str, column: str) -> list[dict[str, object]]:
    ref = _ref(step, table, column)
    return [a.payload for a in step.built().assertions
            if a.subject_ref == ref and a.payload.get("evidence") == STATED_DOMAIN]


def test_a_bound_entitys_spec_domain_is_placed_only_on_the_bound_relation(
    stated_on_bound: StepRun,
) -> None:
    assert _stated(stated_on_bound, "crate_head", "crate_bay")
    assert not _stated(stated_on_bound, "crate_head_spare", "crate_bay")

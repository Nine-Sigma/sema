"""Done-when 2: the discovery scorecard over fixture facts and the fixture's declared constraints.

Facts and the oracle are `Cells` per fact kind, keyed as the contract's oracle file format
keys them. The produced joins are every join the fixture's build writes (each detected join
and the declared composite key) except the one declared key no detector reaches; the oracle
is the fixture's declared foreign keys. Every expected count is read off the fixture's
definitions, never off a run of the scorecard.
"""

from __future__ import annotations

import importlib
from typing import Any

import pytest

from tests.support.connector_fixture import MAIN_TABLES
from tests.support.connector_fixture_scorecard import (
    COMPOSITE_JOIN_KEYS,
    DECLARED_JOIN_KEYS,
    DETECTED_JOIN_KEYS,
    UNDETECTED_JOIN_KEY,
    Key,
    card_rows,
    join_key,
)

pytestmark = pytest.mark.unit

FACT_KIND_IDS = (
    "candidate_key", "entity_coarser_than_row", "entity_shared_across_relations",
    "null_key_policy", "attribute_resolution_rule", "label_source", "observed_values",
    "column_to_concept_binding", "join", "bridge", "hierarchy", "mapping_coverage",
    "value_encoding", "mapping_cardinality", "measure_semantics", "time_roles",
    "universe_and_special_values", "term_default",
)
CELL_COLUMNS = ("produced", "equal", "missing", "wrong", "extra", "rechecked")
MANY_TO_ONE = "many_to_one"

DECLARED_JOINS = dict.fromkeys(DECLARED_JOIN_KEYS, MANY_TO_ONE)
PRODUCED_JOINS = dict.fromkeys(DETECTED_JOIN_KEYS | COMPOSITE_JOIN_KEYS, MANY_TO_ONE)
PLANTED_WRONG = join_key("transfer", ["to_account_id"], "account", ["account_id"])
CANDIDATE_KEYS = {(t.name, "+".join(sorted(t.primary_key))): "unique" for t in MAIN_TABLES}
RECHECKED = {"join": len(DETECTED_JOIN_KEYS), "candidate_key": len(CANDIDATE_KEYS)}

# Read off the fixture: 6 declared keys, 10 detected joins (4 of them declared), 1 composite key.
JOIN_SHAPE = {"produced": 11, "equal": 5, "missing": 1, "wrong": 0, "extra": 6, "rechecked": 10}


def _scorecard() -> Any:
    return importlib.import_module("sema.eval.discovery_scorecard")


def _card(produced: dict[str, dict[Key, Any]], oracle: dict[str, dict[Key, Any]]) -> str:
    scorecard = _scorecard()
    return str(scorecard.render(scorecard.score_discovery(produced, oracle, RECHECKED)))


def test_the_fixture_reading_behind_the_expected_counts() -> None:
    assert (len(DECLARED_JOINS), len(DETECTED_JOIN_KEYS), len(COMPOSITE_JOIN_KEYS)) == (6, 10, 1)
    assert len(DECLARED_JOIN_KEYS & DETECTED_JOIN_KEYS) == 4
    assert UNDETECTED_JOIN_KEY in DECLARED_JOINS and UNDETECTED_JOIN_KEY not in PRODUCED_JOINS
    assert PLANTED_WRONG in DECLARED_JOINS and PLANTED_WRONG in PRODUCED_JOINS


def test_fact_kinds_are_the_eighteen_catalog_ids_in_order() -> None:
    assert tuple(_scorecard().FACT_KINDS) == FACT_KIND_IDS


def test_the_card_has_one_row_per_fact_kind_in_catalog_order() -> None:
    rows = card_rows(_card({"join": PRODUCED_JOINS}, {"join": DECLARED_JOINS}))

    assert tuple(rows) == FACT_KIND_IDS
    assert set(CELL_COLUMNS) <= set(rows["join"])


def test_join_cells_equal_the_fixture_shape() -> None:
    row = card_rows(_card({"join": PRODUCED_JOINS}, {"join": DECLARED_JOINS}))["join"]

    assert {column: row[column] for column in CELL_COLUMNS} == {
        column: str(count) for column, count in JOIN_SHAPE.items()
    }


def test_a_planted_wrong_fact_counts_as_wrong() -> None:
    planted = {**PRODUCED_JOINS, PLANTED_WRONG: "one_to_one"}
    row = card_rows(_card({"join": planted}, {"join": DECLARED_JOINS}))["join"]

    assert (row["equal"], row["wrong"]) == (str(JOIN_SHAPE["equal"] - 1), "1")
    assert (row["missing"], row["extra"]) == (str(JOIN_SHAPE["missing"]), str(JOIN_SHAPE["extra"]))


def test_a_kind_with_no_producer_prints_not_produced() -> None:
    rows = card_rows(_card({"join": PRODUCED_JOINS}, {"join": DECLARED_JOINS}))

    assert "not produced" in " ".join(rows["hierarchy"].values())
    assert "not produced" not in " ".join(rows["join"].values())


def test_a_kind_only_the_oracle_covers_prints_not_produced_and_misses_every_claim() -> None:
    oracle = {"join": DECLARED_JOINS, "candidate_key": CANDIDATE_KEYS}
    rows = card_rows(_card({"join": PRODUCED_JOINS}, oracle))

    row = rows["candidate_key"]
    assert "not produced" in " ".join(row.values())
    assert row["missing"] == str(len(CANDIDATE_KEYS))
    assert "no oracle" not in " ".join(row.values())


def test_a_kind_absent_from_both_sides_prints_not_produced_and_no_oracle() -> None:
    rows = card_rows(_card({"join": PRODUCED_JOINS}, {"join": DECLARED_JOINS}))

    cells = " ".join(rows["hierarchy"].values())
    assert "not produced" in cells and "no oracle" in cells


def test_a_key_only_join_is_equal_to_an_oracle_one_to_one_on_the_same_key() -> None:
    key = join_key("holder", ["account_id"], "account", ["account_id"])
    row = card_rows(_card({"join": {key: "key_only"}}, {"join": {key: "one_to_one"}}))["join"]

    assert (row["equal"], row["wrong"]) == ("1", "0")


def test_a_kind_with_no_oracle_prints_no_oracle_and_its_recheck_count() -> None:
    produced = {"join": PRODUCED_JOINS, "candidate_key": CANDIDATE_KEYS}
    rows = card_rows(_card(produced, {"join": DECLARED_JOINS}))

    row = rows["candidate_key"]
    assert "no oracle" in " ".join(row.values())
    assert (row["produced"], row["rechecked"]) == (str(len(CANDIDATE_KEYS)),) * 2
    assert "no oracle" not in " ".join(rows["join"].values())

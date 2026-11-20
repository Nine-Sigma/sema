"""Done-when 6 (N8): the scorecard cells of the five identity kinds, and their rechecks.

A key fact is read as the build graph holds it (`KeyRead`: the relation, its key columns,
the grade, the NULL-key row count); an entity fact as `EntityRead` (the relation, the entity,
the edge's key columns and `owner`, the `EntityIdentity`, the grade, the NULL-key row count
and the conflict share per non-key column). An edge L2 wrote has no key columns and no grade.
The reads here are built from the fixture's facts, so the expected cells are the fixture's
claims in the contract's F1 forms; the rechecks run through the DuckDB connector on the
fixture's database.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from sema.connectors.duckdb import DuckDBConnector
from sema.models.config import DuckDBConfig
from sema.models.target.identity import EntityIdentity, ResolvedAttribute, ResolutionRule
from tests.support.connector_fixture_identity import (
    CODE_LIST_TEXTS,
    IDENTITY_CATALOG,
    IDENTITY_SCHEMA,
    PROPOSALS,
    TABLES,
    conflict_shares,
    null_row_count,
    write_identity_database,
)
from tests.support.connector_fixture_identity_runs import (
    COARSER,
    KEYED,
    NULL_PART_KEYS,
    SHARED,
    identity_claims,
    ordered_by,
)
from tests.support.connector_fixture_scorecard import card_rows

pytestmark = pytest.mark.unit

IDENTITY_KINDS = ("candidate_key", "entity_coarser_than_row", "entity_shared_across_relations",
                  "null_key_policy", "attribute_resolution_rule")
CLAIMS = {kind: cells for kind, cells in identity_claims().items() if kind in IDENTITY_KINDS}


def _scorecard() -> Any:
    return importlib.import_module("sema.eval.discovery_scorecard")


def key_reads(grade: str = "proven") -> list[Any]:
    read = _scorecard().KeyRead
    reads = [read(relation=t, columns=k, verification=grade, null_rows=0)
             for t, k in KEYED.items()]
    nullable = "open" if grade == "proven" else grade
    return reads + [read(relation=t, columns=k, verification=nullable,
                         null_rows=null_row_count(t, k)) for t, k in NULL_PART_KEYS.items()]


def _identity(table: str) -> EntityIdentity:
    key, by = COARSER[table], ordered_by(table)
    conflicting = [c for c, share in conflict_shares(table, key).items() if share > 0]
    attributes = tuple(ResolvedAttribute(column=c, rule=ResolutionRule.MAX_BY, by=by)
                       for c in conflicting) if by else ()
    return EntityIdentity(key_columns=key, attributes=attributes)


def entity_reads() -> list[Any]:
    read = _scorecard().EntityRead
    owners = [read(relation=t, entity=PROPOSALS[t].entity, key_columns=k, owner=True,
                   identity=_identity(t), verification="supported",
                   null_rows=null_row_count(t, k), conflict_share=conflict_shares(t, k))
              for t, k in COARSER.items()]
    shared = [read(relation=other, entity=PROPOSALS[owner].entity, key_columns=key,
                   owner=False, identity=None, verification="proven", null_rows=None,
                   conflict_share={})
              for (owner, key), others in SHARED.items() for other in others]
    return owners + shared


def l2_edge(relation: str, entity: str) -> Any:
    return _scorecard().EntityRead(relation=relation, entity=entity, key_columns=None,
                                   owner=None, identity=None, verification=None,
                                   null_rows=None, conflict_share={})


def facts(keys: list[Any], entities: list[Any]) -> dict[str, Any]:
    return dict(_scorecard().identity_facts(keys, entities))


def test_each_identity_kinds_cells_equal_the_fixture_claims() -> None:
    assert facts(key_reads(), entity_reads()) == CLAIMS


def test_the_claims_hold_each_case() -> None:
    assert CLAIMS["entity_shared_across_relations"] == {
        ("hull_rev", "hull_no"): ["hull_cargo", "hull_survey"]}
    assert CLAIMS["null_key_policy"][("mooring", "pier_code+slot_no")] == 3
    assert CLAIMS["null_key_policy"][("crane_rev", "crane_no")] == 2
    rules = CLAIMS["attribute_resolution_rule"]
    assert rules[("hull_rev", "paint_band")] == "latest:rev_no"
    assert rules[("hull_rev", "keel_class")] == "single"
    assert ("buoy_check", "lamp_state") not in rules
    assert ("mooring", "pier_code+slot_no") not in CLAIMS["candidate_key"]


def test_a_planted_wrong_fact_counts_as_wrong() -> None:
    scorecard = _scorecard()
    entities = [read._replace(identity=EntityIdentity(key_columns=("hull_no", "keel_class")))
                if read.relation == "hull_rev" and read.owner else read
                for read in entity_reads()]

    produced = scorecard.identity_facts(key_reads(), entities)
    rows = card_rows(scorecard.render(scorecard.score_discovery(produced, CLAIMS, {})))

    row = rows["entity_coarser_than_row"]
    assert (row["equal"], row["wrong"]) == (str(len(COARSER) - 1), "1")


@pytest.mark.parametrize("grade", ["open", "rejected"])
def test_an_open_or_rejected_fact_is_not_produced(grade: str) -> None:
    entities = [read._replace(verification=grade) for read in entity_reads()]

    produced = facts(key_reads(grade), entities)

    for kind in ("candidate_key", "entity_coarser_than_row", "entity_shared_across_relations",
                 "attribute_resolution_rule"):
        assert produced[kind] == {}, kind
    nulls = produced["null_key_policy"]
    if grade == "open":
        assert nulls == CLAIMS["null_key_policy"], "a NULL-key count is a measurement"
    else:
        assert nulls == {}


def test_an_l2_edge_with_no_key_columns_is_not_a_shared_entity() -> None:
    entities = [*entity_reads(), l2_edge("hull_quote", "Hull"), l2_edge("pump", "pump"),
                l2_edge("valve", "pump")]

    produced = facts(key_reads(), entities)

    assert produced["entity_shared_across_relations"] == CLAIMS["entity_shared_across_relations"]


@pytest.fixture
def connector(tmp_path: Path) -> Iterator[DuckDBConnector]:
    opened = DuckDBConnector(DuckDBConfig(path=str(write_identity_database(tmp_path))))
    yield opened
    opened.close()


def _rechecked(connector: DuckDBConnector, produced: dict[str, Any]) -> dict[str, int]:
    counts = _scorecard().recheck(connector, IDENTITY_CATALOG, [IDENTITY_SCHEMA], produced)
    return {kind: counts.get(kind) for kind in IDENTITY_KINDS}


def test_every_claimed_identity_fact_rechecks_on_the_data(connector: DuckDBConnector) -> None:
    assert _rechecked(connector, CLAIMS) == {kind: len(CLAIMS[kind]) for kind in IDENTITY_KINDS}


def test_a_fact_the_data_refutes_does_not_recheck(connector: DuckDBConnector) -> None:
    """A coincidental key re-proves; the two entities beside the refuted one key the rules."""
    refuted = {
        "candidate_key": {("gate_tally", "gate_code"): "unique",
                          ("tide_log", "reading_mm"): "unique"},
        "entity_coarser_than_row": {("berth",): "berth_no", ("hull_rev",): "hull_no",
                                    ("buoy_check",): "buoy_no"},
        "entity_shared_across_relations": {("hull_rev", "hull_no"): ["hull_quote"]},
        "null_key_policy": {("crane_rev", "crane_no"): 1},
        "attribute_resolution_rule": {("buoy_check", "lamp_state"): "latest:check_day",
                                      ("hull_rev", "paint_band"): "single"},
    }

    assert _rechecked(connector, refuted) == {
        "candidate_key": 1, "entity_coarser_than_row": 2, "entity_shared_across_relations": 0,
        "null_key_policy": 0, "attribute_resolution_rule": 0,
    }


def test_loaded_code_lists_still_names_the_binding_cell(connector: DuckDBConnector) -> None:
    scorecard = _scorecard()
    rig = scorecard.BindingRead(
        schema=IDENTITY_SCHEMA, relation="hull_rev", column="rig_type", code_source="rig_kinds",
        verification="verified", label=None, dictionary_label=True, listed_row_share=1.0,
        member_rows={code: 1 for code in CODE_LIST_TEXTS["rig_kinds"]},
    )

    loaded = scorecard.loaded_code_lists(connector, IDENTITY_CATALOG, [IDENTITY_SCHEMA], [rig],
                                         CODE_LIST_TEXTS)

    assert loaded == {"rig_kinds": "rig_kind"}
    assert set(TABLES["rig_kind"].column_names) >= {"rig_code"}

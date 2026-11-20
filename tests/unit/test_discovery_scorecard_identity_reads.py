"""The identity facts the scorecard reads from the build graph (contract F6, F8).

A stand-in driver answers the one assertion read per predicate with the rows the graph would
return (each assertion's payload and its relation's `key_columns`), and the edge read with the
scope's `ENTITY_ON_TABLE` edges that carry `key_columns`.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from sema.eval.discovery_scorecard import identity_facts, read_identity
from sema.models.target.identity import EntityIdentity, ResolutionRule, ResolvedAttribute

pytestmark = pytest.mark.unit

EDGES = "edges"

LATEST = EntityIdentity(key_columns=("lot_no",), attributes=(
    ResolvedAttribute(column="size", rule=ResolutionRule.MAX_BY, by="rev"),))


def _key(relation: str, columns: list[str], grade: str, nulls: int,
         table_key: list[str] | None) -> dict[str, Any]:
    payload = {"columns": columns, "verification": grade, "alternatives": [],
               "evidence": {"rows": 9, "distinct": 9, "null_rows": nulls, "signals": []}}
    return {"relation": relation, "payload": json.dumps(payload), "table_key": table_key}


def _owner_edge(grade: str) -> dict[str, Any]:
    return {"key_columns": ["lot_no"], "owner": True, "verification": grade,
            "identity": LATEST.model_dump_json()}


def _identity(grade: str) -> dict[str, Any]:
    payload = {"entity": "Lot", "key_columns": ["lot_no"], "owner": True,
               "identity": LATEST.model_dump(mode="json"), "verification": grade,
               "alternatives": [], "evidence": {"rows": 9, "groups": 4, "null_rows": 1,
                                                "conflict_share": {"note": 0.0, "size": 0.5}}}
    return {"relation": "Lot_Rev", "payload": json.dumps(payload), "table_key": ["rev_id"]}


def _edge(relation: str, entity: str, **properties: Any) -> dict[str, Any]:
    return {"relation": relation, "entity": entity, "properties": properties}


def _edges(grade: str) -> list[dict[str, Any]]:
    """The owner's edge, a step edge and a pre-existing edge given key columns on relations
    with no assertion, and a step edge the data rejected."""
    return [
        _edge("Lot_Rev", "Lot", **_owner_edge(grade)),
        _edge("Lot_Note", "Lot", key_columns=["lot_no"], owner=False, verification="proven"),
        _edge("Lot_Tag", "Lot", key_columns=["lot_no"]),
        _edge("Lot_Void", "Lot", key_columns=["lot_no"], owner=False, verification="rejected"),
    ]


class _Session:
    def __init__(self, rows: dict[str, list[dict[str, Any]]]) -> None:
        self.rows = rows

    def __enter__(self) -> _Session:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def run(self, _cypher: str, **params: Any) -> list[dict[str, Any]]:
        return self.rows[params.get("predicate", EDGES)]


class _Driver:
    def __init__(self, rows: dict[str, list[dict[str, Any]]]) -> None:
        self.rows = rows

    def session(self) -> _Session:
        return _Session(self.rows)


def _facts(grade: str) -> dict[str, Any]:
    driver = _Driver({
        "has_key": [_key("lot", ["lot_no"], grade, 0, ["lot_no"]),
                    _key("lot", ["tag"], "supported", 0, ["lot_no"]),
                    _key("Bin", ["B", "a"], "open", 2, None)],
        "has_identity": [_identity(grade)],
        EDGES: _edges(grade),
    })
    return dict(identity_facts(*read_identity(driver, "cat", ["s"])))


def test_the_graph_read_forms_give_each_identity_kind_in_f1_forms() -> None:
    assert _facts("proven") == {
        "candidate_key": {("lot", "lot_no"): "unique"},
        "entity_coarser_than_row": {("lot_rev",): "lot_no"},
        "entity_shared_across_relations": {("lot_rev", "lot_no"): ["lot_note", "lot_tag"]},
        "null_key_policy": {("lot", "lot_no"): 0, ("lot", "tag"): 0, ("bin", "a+b"): 2,
                            ("lot_rev", "lot_no"): 1},
        "attribute_resolution_rule": {("lot_rev", "note"): "single",
                                      ("lot_rev", "size"): "latest:rev"},
    }


def test_a_sharing_edge_on_a_relation_with_no_assertion_is_read() -> None:
    shared = _facts("supported")["entity_shared_across_relations"]

    assert shared == {("lot_rev", "lot_no"): ["lot_note", "lot_tag"]}


def test_an_open_sharing_edge_is_not_counted() -> None:
    driver = _Driver({"has_key": [], "has_identity": [_identity("proven")], EDGES: [
        *_edges("proven"),
        _edge("Lot_Wait", "Lot", key_columns=["lot_no"], owner=False, verification="open"),
    ]})

    facts = identity_facts(*read_identity(driver, "cat", ["s"]))

    assert facts["entity_shared_across_relations"] == {("lot_rev", "lot_no"): ["lot_note", "lot_tag"]}


def test_an_accepted_set_the_table_does_not_hold_as_its_key_is_no_candidate_key() -> None:
    assert ("lot", "tag") not in _facts("supported")["candidate_key"]


def test_a_rejected_read_states_nothing_but_the_open_null_count() -> None:
    facts = _facts("rejected")

    assert facts["entity_coarser_than_row"] == facts["candidate_key"] == {}
    assert facts["entity_shared_across_relations"] == {}
    assert facts["null_key_policy"] == {("lot", "tag"): 0, ("bin", "a+b"): 2}

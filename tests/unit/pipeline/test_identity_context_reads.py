"""Done-when 5 (N7): the graph's labels and shared entities reach the core query context.

The stand-in graph's mapping rows hold what `resolve_physical_mapping` returns: each
column's `LABELED_BY` targets with the predicates of the joins their edge names (each a
`JoinPath.join_predicates` value), and the other entities' edges that share the relation.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from tests.support.connector_fixture_identity import IDENTITY_CATALOG, IDENTITY_SCHEMA
from tests.unit.pipeline.test_identity_context import (
    HULL_IDENTITY,
    QUESTION,
    MappingGraph,
    context,
    fqn,
    mapping_row,
    owner_edge,
    section_lines,
    table_line,
)

pytestmark = pytest.mark.unit

RIG_JOIN = {"left_table": "rig_kind", "left_column": "rig_code", "right_table": "hull_rev",
            "right_column": "rig_type", "operator": "="}
HULL_OWNER = {"catalog": IDENTITY_CATALOG, "schema": IDENTITY_SCHEMA, "table": "hull_rev"}


def labeled(row: dict[str, Any], column: str, *labels: dict[str, Any]) -> dict[str, Any]:
    row["columns"] = [{**c, "labels": list(labels)} if c["column"] == column else c
                      for c in row["columns"]]
    return row


def label(verification: str = "verified") -> dict[str, Any]:
    return {"table": "rig_kind", "column": "rig_title", "verification": verification,
            "via": [json.dumps([RIG_JOIN])]}


def shared(verification: str = "proven", owner: dict[str, str] | None = HULL_OWNER,
           ) -> dict[str, Any]:
    return {"entity": "Hull", "key_columns": ["hull_no"], "verification": verification,
            "owner": owner}


def lines_for(*rows: dict[str, Any]) -> list[str]:
    return context(MappingGraph({"Hull": list(rows)}, ["Hull"]), QUESTION)


def test_a_verified_label_reaches_its_column_line_with_its_join() -> None:
    lines = lines_for(labeled(mapping_row("hull_rev"), "rig_type", label()))

    line = table_line(lines, "hull_rev")

    assert "rig_type (label: rig_kind.rig_title on rig_kind.rig_code = hull_rev.rig_type)" in line


def test_an_unverified_label_renders_nothing() -> None:
    lines = lines_for(labeled(mapping_row("hull_rev"), "rig_type", label("unverified")))

    assert "label:" not in "\n".join(lines)


def test_a_shared_entity_reaches_the_section() -> None:
    cargo = mapping_row("hull_cargo", shared_entities=[
        shared(), shared("open"), shared(owner=None)])

    body = section_lines(lines_for(mapping_row("hull_rev"), cargo))

    assert body == [f"{fqn('hull_cargo')}: rows belong to Hull of {fqn('hull_rev')} "
                    "through hull_no"]


def test_an_unreadable_identity_states_no_entity() -> None:
    hull = mapping_row("hull_rev", **{**owner_edge(HULL_IDENTITY), "identity": "{not json"})

    assert not [line for line in section_lines(lines_for(hull)) if "several rows" in line]

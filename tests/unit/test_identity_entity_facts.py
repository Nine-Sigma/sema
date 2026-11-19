"""Done-when 3 (N5), contracts F4 and F6: one entity fact per relation and entity name, and a
share always spells its owner's key.

A versioned relation owns a coarser entity under its L2 name, and a second relation names the
versioned relation's row key as its entity key, so the row-owned entity takes the same name.
The relation keeps one fact under that name: the coarser one, which carries its identity. The
second relation's rows hold the row key, not the coarser key, so it shares nothing under it.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import duckdb
import pytest

from sema.connectors.duckdb import DuckDBConnector
from sema.engine.identity_utils import Counts, EntityScope, KeyFact, entity_facts, entity_holds
from sema.engine.warehouse_lookup import RelationEvidence
from sema.llm_client import IdentityProposal
from sema.models.config import DuckDBConfig
from sema.models.extraction import ExtractedColumn

pytestmark = pytest.mark.unit

CATALOG, SCHEMA = "facts", "lot"
VERSIONED, CHILD = (SCHEMA, "item_rev"), (SCHEMA, "item_use")
TABLES = {
    VERSIONED: (("rev_id", "item_no", "rev_no", "colour"),
                [(1, 10, 1, "red"), (2, 10, 2, "blue"), (3, 11, 1, "green")]),
    CHILD: (("use_id", "rev_id"), [(100, 1), (101, 1), (102, 3)]),
}
PROPOSALS = {
    VERSIONED: IdentityProposal(entity="item", entity_key=["item_no"], latest_by="rev_no"),
    CHILD: IdentityProposal(entity="usage", entity_key=["rev_id"]),
}


def _key(column: str) -> KeyFact:
    return KeyFact((column,), True, 3, 3, 0, verification="supported")


def _scope(directory: Path, tables: dict[tuple[str, str], Any] = TABLES) -> EntityScope:
    path = directory / f"{CATALOG}.duckdb"
    with duckdb.connect(str(path)) as con:
        con.execute(f"CREATE SCHEMA {SCHEMA}")
        for (_, table), (columns, rows) in tables.items():
            listed = ", ".join(f"{c} {'VARCHAR' if c == 'colour' else 'INTEGER'}" for c in columns)
            con.execute(f"CREATE TABLE {SCHEMA}.{table} ({listed})")
            con.executemany(f"INSERT INTO {SCHEMA}.{table} VALUES "
                            f"({', '.join('?' for _ in columns)})", rows)
    relations = {
        relation: [ExtractedColumn(c, relation[1], CATALOG, SCHEMA, "INTEGER") for c in columns]
        for relation, (columns, _) in tables.items()
    }
    counts = Counts(RelationEvidence(DuckDBConnector(DuckDBConfig(path=str(path))), CATALOG))
    keys = {VERSIONED: _key("rev_id"), CHILD: _key("use_id")}
    return EntityScope(relations, keys, counts, lambda _relation, _columns: False)


def test_a_relation_keeps_its_versioned_entity_and_shares_none_under_another_key(
    tmp_path: Path,
) -> None:
    facts = entity_facts(_scope(tmp_path), PROPOSALS, {VERSIONED: "item", CHILD: None})

    on_versioned = [(f.entity, f.key_columns) for f in facts if f.relation == VERSIONED]
    assert on_versioned == [("item", ("item_no",))]
    [kept] = [f for f in facts if f.relation == VERSIONED]
    assert kept.identity is not None and kept.identity.key_columns == ("item_no",)
    assert [f for f in facts if f.relation == CHILD] == []


def test_a_row_entity_under_its_own_name_stays_beside_the_versioned_entity(
    tmp_path: Path,
) -> None:
    facts = entity_facts(_scope(tmp_path), PROPOSALS, {VERSIONED: "item revision", CHILD: None})

    on_versioned = {f.entity: f.key_columns for f in facts if f.relation == VERSIONED}
    assert on_versioned == {"item": ("item_no",), "item revision": ("rev_id",)}
    assert max(Counter((f.entity, f.relation) for f in facts).values()) == 1


@pytest.mark.parametrize(("distinct", "holds"), [(1, False), (2, True)])
def test_a_one_column_entity_key_needs_more_than_one_group(distinct: int, holds: bool) -> None:
    assert entity_holds(3, distinct, []) is holds


@pytest.mark.parametrize(("second_rev", "ordered_by"), [(2, {"rev_no"}), (None, set())])
def test_an_ordering_column_with_a_null_picks_no_row(
    tmp_path: Path, second_rev: int | None, ordered_by: set[str],
) -> None:
    columns, _ = TABLES[VERSIONED]
    rows = [(1, 10, 1, "red"), (2, 10, second_rev, "blue"), (3, 11, 1, "green")]

    facts = entity_facts(_scope(tmp_path, {**TABLES, VERSIONED: (columns, rows)}), PROPOSALS,
                         {VERSIONED: "item", CHILD: None})

    [kept] = [f for f in facts if f.relation == VERSIONED]
    assert kept.identity is not None
    assert "colour" in kept.conflict_share
    assert {a.by for a in kept.identity.attributes} == ordered_by

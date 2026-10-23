"""Bound entries that fill one single-valued spec section reopen at Structure (decision 18b).

Pinned contract, read through what Review presents:

- When two or more bound entries carry the same single-valued section (``concepts`` here),
  every relation of every such entry moves from bound to open: one structure entry per
  relation, its members the binding's member alone, no choice and no source, and a reason
  naming the section and how many bindings fill it. ``run_metrics.open_structure`` counts
  them and no spec is presented.
- Decisions that keep both bindings reopen them the same way; decisions that keep one by
  naming its member and ``ignore`` the other's relations settle, and the run presents a spec
  with no model call.
- Sections that hold several bindings stay bound: one binding per section, several
  ``entity_model`` bindings, and several flat code tables (``code_lists`` is merged).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import duckdb
import pytest

from sema.intake.shapes.base import SlotBinding
from sema.intake.shapes.precedence import member
from sema.models.schema_metadata import SchemaMetadata
from tests.unit.intake.catalog_env import (
    COLOURS_TABLE,
    GROUP_VALUES_TABLE,
    GROUPS_TABLE,
    ITEMS_TABLE,
    SHAPES_TABLE,
    VALUES_TABLE,
    WAREHOUSE,
    connector_over,
    two_catalogs,
    two_flat_tables,
    two_grouped_tables,
    warehouse_catalog,
    write_descriptor,
)
from tests.unit.intake.decisions_env import write_decisions
from tests.unit.intake.graph_env import (
    ACRIS_DESCRIPTOR,
    APPROVER,
    STRUCTURE_BATCH,
    Answer,
    choose_flat_lists,
    choosing,
    code_tables,
    discover,
    graph,
    routed_client,
    settings,
)
from tests.unit.intake.warehouse_env import CATALOG, SCHEMA, FakeConnector

pytestmark = pytest.mark.unit

STRUCTURE, IGNORE, SECTION = "structure", "ignore", "concepts"
PAIR, GROUPED, ENTITY_MODEL = (
    "concept_and_value_set", "concept_with_values_denormalized", "entity_model",
)
PAIRS = (
    (ITEMS_TABLE, VALUES_TABLE, "concept_code"),
    (GROUPS_TABLE, GROUP_VALUES_TABLE, "group_code"),
)
GROUPED_TABLES, GROUP_KEY = (COLOURS_TABLE, SHAPES_TABLE), "group_key"
Warehouse = tuple[duckdb.DuckDBPyConnection, SchemaMetadata]
Listed = set[tuple[str, tuple[str, ...]]]


def _relation(table: str) -> str:
    return f"{WAREHOUSE}.{table}"


def _member(template: str, slots: dict[str, str], *relations: str) -> str:
    binding = SlotBinding(template=template, section=SECTION, slots=slots, relations=relations)
    found: str = member(binding)
    return found


def _pair_member(concepts: str, values: str, key: str) -> str:
    concept, value = _relation(concepts), _relation(values)
    slots = {
        "concept_key": key, "concept_relation": concept, "value_key": key, "value_relation": value,
    }
    return _member(PAIR, slots, concept, value)


def _grouped_member(table: str) -> str:
    return _member(GROUPED, {"relation": _relation(table), "group_by": GROUP_KEY}, _relation(table))


@contextmanager
def _opened(build: Callable[[], Warehouse]) -> Iterator[FakeConnector]:
    con, metadata = build()
    try:
        yield connector_over(con, metadata)
    finally:
        con.close()


@pytest.fixture
def two_pairs() -> Iterator[FakeConnector]:
    with _opened(two_catalogs) as connector:
        yield connector


@pytest.fixture
def grouped() -> Iterator[FakeConnector]:
    with _opened(two_grouped_tables) as connector:
        yield connector


def _discover(root: Path, connector: FakeConnector, client: MagicMock) -> Any:
    return graph().discover(
        None, warehouse=connector, catalog=CATALOG, schema=SCHEMA,
        descriptor=write_descriptor(root / "inputs"), client=client, **settings(root),
    )


def _approve(
    root: Path, run_id: str, connector: FakeConnector, client: MagicMock,
    rows: list[tuple[str, str, str]],
) -> Any:
    decisions = write_decisions(root / "decisions.yaml", rows)
    return graph().approve(
        run_id, approver=APPROVER, client=client, warehouse=connector,
        decisions=decisions, **settings(root),
    )


def _open_structure(outcome: Any) -> list[dict[str, Any]]:
    return [
        a for a in outcome.payload["ambiguities"]
        if a["stage"] == STRUCTURE and a["choice"] is None and a["source"] is None
    ]


def _assert_reopened(outcome: Any, expected: Listed, bindings: int) -> None:
    listed = _open_structure(outcome)
    assert outcome.node == "review"
    assert {(a["subject"], tuple(a["members"])) for a in listed} == expected, listed
    for entry in listed:
        reason = str(entry["reason"])
        assert SECTION in reason and str(bindings) in reason, reason
    metrics = outcome.payload["run_metrics"]
    assert (metrics["open_structure"], metrics["bound"]) == (len(expected), 0), metrics
    assert outcome.payload["spec"] is None


def _assert_stays_bound(outcome: Any) -> None:
    metrics = outcome.payload["run_metrics"]
    assert _open_structure(outcome) == []
    assert metrics["open_structure"] == 0, metrics
    assert metrics["bound"] == metrics["relations"], metrics


def _choosing_grouped() -> Answer:
    wanted = {_grouped_member(table) for table in GROUPED_TABLES}
    return choosing(wanted.__contains__)


def _grouped_listing() -> Listed:
    return {(_relation(table), (_grouped_member(table),)) for table in GROUPED_TABLES}


def test_two_grouped_bindings_filling_concepts_reopen_at_structure(
    tmp_path: Path, grouped: FakeConnector
) -> None:
    client = routed_client(defaults={STRUCTURE_BATCH: _choosing_grouped()})

    outcome = _discover(tmp_path, grouped, client)

    _assert_reopened(outcome, _grouped_listing(), len(GROUPED_TABLES))


def test_two_pair_bindings_filling_concepts_reopen_every_relation(
    tmp_path: Path, two_pairs: FakeConnector
) -> None:
    client = routed_client()

    outcome = _discover(tmp_path, two_pairs, client)

    expected: Listed = {
        (_relation(table), (_pair_member(*pair),)) for pair in PAIRS for table in pair[:2]
    }
    _assert_reopened(outcome, expected, len(PAIRS))
    assert client.invoke.call_count == 0


def test_keeping_both_conflicting_bindings_reopens_them(
    tmp_path: Path, grouped: FakeConnector
) -> None:
    client = routed_client(defaults={STRUCTURE_BATCH: _choosing_grouped()})
    first = _discover(tmp_path, grouped, client)
    calls = client.invoke.call_count
    rows = [(STRUCTURE, _relation(table), _grouped_member(table)) for table in GROUPED_TABLES]

    outcome = _approve(tmp_path, first.run_id, grouped, client, rows)

    _assert_reopened(outcome, _grouped_listing(), len(GROUPED_TABLES))
    assert client.invoke.call_count == calls


def test_keeping_one_binding_and_ignoring_the_other_assembles_with_no_call(
    tmp_path: Path, two_pairs: FakeConnector
) -> None:
    client = routed_client()
    first = _discover(tmp_path, two_pairs, client)
    (kept, ignored) = PAIRS
    rows = [(STRUCTURE, _relation(table), _pair_member(*kept)) for table in kept[:2]]
    rows += [(STRUCTURE, _relation(table), IGNORE) for table in ignored[:2]]

    outcome = _approve(tmp_path, first.run_id, two_pairs, client, rows)

    metrics, spec = outcome.payload["run_metrics"], outcome.payload["spec"]
    assert outcome.node == "review"
    assert (metrics["open_structure"], metrics["open_roles"]) == (0, 0), metrics
    assert spec is not None, metrics
    assert spec["concepts"]["relation"]["name"] == _relation(ITEMS_TABLE)
    assert spec["value_sets"]["relation"]["name"] == _relation(VALUES_TABLE)
    assert client.invoke.call_count == 0


def test_one_binding_per_section_stays_bound_and_assembles(tmp_path: Path) -> None:
    client = routed_client()
    with _opened(lambda: warehouse_catalog(status=False)) as connector:
        outcome = _discover(tmp_path, connector, client)

    _assert_stays_bound(outcome)
    assert outcome.payload["spec"] is not None
    assert client.invoke.call_count == 0


def test_several_entity_model_bindings_stay_bound_and_assemble(tmp_path: Path) -> None:
    client = routed_client(
        defaults={STRUCTURE_BATCH: choosing(lambda found: found.startswith(f"{ENTITY_MODEL}("))}
    )
    with _opened(two_flat_tables) as connector:
        outcome = _discover(tmp_path, connector, client)

    _assert_stays_bound(outcome)
    assert outcome.payload["spec"] is not None


def test_several_flat_code_tables_stay_bound(tmp_path: Path) -> None:
    client = routed_client(defaults={STRUCTURE_BATCH: choose_flat_lists})

    found = discover(tmp_path, code_tables(tmp_path), ACRIS_DESCRIPTOR, client)

    _assert_stays_bound(found.outcome)

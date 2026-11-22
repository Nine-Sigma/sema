"""One join pass per schema: stored declared keys and detected joins in one batch."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock

import pytest

from sema.graph.loader import GraphLoader
from sema.models.assertions import Assertion, AssertionPredicate, AssertionStatus

pytestmark = pytest.mark.unit

SCHEMA = "main"


def _ref(table: str, column: str | None = None) -> str:
    base = f"databricks://ds/cat/{SCHEMA}/{table}"
    return f"{base}/{column}" if column else base


def _assertion(
    predicate: AssertionPredicate, subject: str, payload: dict[str, Any],
    object_ref: str, source: str, confidence: float,
) -> Assertion:
    return Assertion(
        id=f"{predicate.value}-{subject}", subject_ref=subject, predicate=predicate,
        payload=payload, object_ref=object_ref, source=source, confidence=confidence,
        status=AssertionStatus.AUTO, run_id="run-1",
        observed_at=datetime(2026, 1, 1, tzinfo=timezone.utc), source_schema=SCHEMA,
    )


def _declared() -> Assertion:
    payload = {
        "join_predicates": [{
            "left_table": "item", "left_column": "item_id",
            "right_table": "line", "right_column": "item_id", "operator": "=",
        }],
        "from_ref": _ref("item"), "to_ref": _ref("line"),
        "constraint_name": "fk_line_item", "hop_count": 1,
    }
    return _assertion(
        AssertionPredicate.JOINS_TO, _ref("line"), payload, _ref("item"),
        source="unity_catalog", confidence=0.95,
    )


def _detected(confidence: float) -> Assertion:
    payload = {"pk_table": "item", "pk_column": "item_id",
               "fk_table": "line", "fk_column": "item_id", "tier": 1}
    return _assertion(
        AssertionPredicate.FK_TO, _ref("line", "item_id"), payload, _ref("item", "item_id"),
        source="fk_detector", confidence=confidence,
    )


def _loader_storing(assertions: list[Assertion]) -> MagicMock:
    loader = MagicMock()
    loader._run_read.return_value = GraphLoader(MagicMock())._build_assertion_dicts(
        assertions, source_schema=SCHEMA,
    )
    return loader


def _written_join_paths(loader: MagicMock) -> list[dict[str, Any]]:
    return [
        row for call in loader._run.call_args_list if "JoinPath" in call.args[0]
        for row in call.kwargs.get("rows", [])
    ]


def test_a_detected_join_does_not_overwrite_the_declared_key_it_duplicates() -> None:
    from sema.graph.join_materializer import materialize_schema_joins

    loader = _loader_storing([_declared()])
    materialize_schema_joins(loader, SCHEMA, [_detected(confidence=0.99)])
    rows = _written_join_paths(loader)
    assert [(r["name"], r["source"], r["confidence"]) for r in rows] == [
        ("item/item_id=line/item_id", "unity_catalog", 0.95),
    ]


def test_stored_declared_keys_are_written_with_no_detected_joins() -> None:
    from sema.graph.join_materializer import materialize_schema_joins

    loader = _loader_storing([_declared()])
    materialize_schema_joins(loader, SCHEMA, [])
    assert [r["name"] for r in _written_join_paths(loader)] == ["item/item_id=line/item_id"]
    wired = {call.args[1].table for call in loader.add_join_path_uses.call_args_list}
    assert wired == {"item", "line"}
    loader.add_join_path_entity_links.assert_called_once()


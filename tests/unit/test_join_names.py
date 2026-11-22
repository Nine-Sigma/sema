"""C4 join names, and join records grouped by name rather than by subject."""
from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock

import pytest

from sema.graph import join_materializer_utils as utils
from sema.graph.join_materializer import materialize_join_paths
from sema.log import logger
from sema.models.assertions import Assertion, AssertionPredicate, AssertionStatus
from sema.models.physical_key import PhysicalKey

pytestmark = pytest.mark.unit

CATALOG = "cat"
SCHEMA = "main"


def _key(table: str, schema: str = SCHEMA, catalog: str = CATALOG) -> PhysicalKey:
    return PhysicalKey("ds", catalog, schema, table)


def _ref(table: str, column: str | None = None) -> str:
    base = f"databricks://ds/{CATALOG}/{SCHEMA}/{table}"
    return f"{base}/{column}" if column else base


def _pair(
    left: str, left_col: str, right: str, right_col: str, operator: str = "=",
) -> dict[str, str]:
    return {
        "left_table": left, "left_column": left_col,
        "right_table": right, "right_column": right_col, "operator": operator,
    }


def _assertion(
    predicate: AssertionPredicate, subject: str, payload: dict[str, Any],
    object_ref: str | None = None, id_: str = "a",
) -> Assertion:
    return Assertion(
        id=id_, subject_ref=subject, predicate=predicate, payload=payload,
        object_ref=object_ref, source="connector", confidence=0.95,
        status=AssertionStatus.AUTO, run_id="run-1",
        observed_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def _declared(referenced: str, referencing: str, pairs: list[tuple[str, str]],
              id_: str) -> Assertion:
    payload = {
        "join_predicates": [_pair(referenced, left, referencing, right) for left, right in pairs],
        "from_ref": _ref(referenced), "to_ref": _ref(referencing),
        "constraint_name": f"fk_{id_}", "hop_count": 1,
    }
    return _assertion(
        AssertionPredicate.JOINS_TO, _ref(referencing), payload,
        object_ref=_ref(referenced), id_=id_,
    )


def _detected(pk_table: str, pk_col: str, fk_table: str, fk_col: str) -> Assertion:
    payload = {"pk_table": pk_table, "pk_column": pk_col,
               "fk_table": fk_table, "fk_column": fk_col, "tier": 1}
    return _assertion(
        AssertionPredicate.FK_TO, _ref(fk_table, fk_col), payload,
        object_ref=_ref(pk_table, pk_col), id_="detected",
    )


def _written(assertions: list[Assertion]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[Assertion]] = {}
    for a in assertions:
        groups.setdefault((a.subject_ref, a.predicate.value), []).append(a)
    loader = MagicMock()
    materialize_join_paths(loader, groups, source_schema=SCHEMA)
    batches = [c.kwargs["rows"] for c in loader.method_calls if c[0] == "_run"]
    return [row for rows in batches for row in rows]


@pytest.fixture
def log_lines() -> Iterator[list[str]]:
    lines: list[str] = []
    handler_id = logger.add(lambda message: lines.append(str(message)), level="WARNING")
    yield lines
    logger.remove(handler_id)


class TestJoinPathName:
    def test_same_schema_equality_join_keeps_todays_name(self) -> None:
        name = utils.join_path_name(
            [_pair("account", "account_id", "transfer", "account_id")],
            _key("account"), _key("transfer"),
        )
        assert name == "account/account_id=transfer/account_id"

    def test_pairs_sort_by_left_column(self) -> None:
        name = utils.join_path_name(
            [_pair("head", "order_id", "line", "order_id"),
             _pair("head", "line_no", "line", "line_no")],
            _key("head"), _key("line"),
        )
        assert name == "head/line_no=line/line_no|head/order_id=line/order_id"

    def test_operator_other_than_equality_is_shown(self) -> None:
        name = utils.join_path_name(
            [_pair("period", "valid_from", "event", "event_date", "<=")],
            _key("period"), _key("event"),
        )
        assert name == "period/valid_from<=event/event_date"

    def test_missing_operator_reads_as_equality(self) -> None:
        predicate = _pair("a", "x", "b", "y")
        del predicate["operator"]
        assert utils.join_path_name([predicate], _key("a"), _key("b")) == "a/x=b/y"

    def test_sides_in_different_schemas_are_schema_qualified(self) -> None:
        name = utils.join_path_name(
            [_pair("a", "x", "b", "y")], _key("a", schema="s1"), _key("b", schema="s2"),
        )
        assert name == "s1.a/x=s2.b/y"

    def test_sides_in_different_catalogs_are_catalog_qualified(self) -> None:
        name = utils.join_path_name(
            [_pair("a", "x", "b", "y")],
            _key("a", catalog="c1"), _key("b", catalog="c2"),
        )
        assert name == "c1.main.a/x=c2.main.b/y"


class TestJoinRecordsByName:
    def test_two_joins_on_one_subject_are_two_records(self) -> None:
        rows = _written([
            _declared("account", "transfer", [("account_id", "from_account_id")], "r1"),
            _declared("account", "transfer", [("account_id", "to_account_id")], "r2"),
        ])
        assert sorted(r["name"] for r in rows) == [
            "account/account_id=transfer/from_account_id",
            "account/account_id=transfer/to_account_id",
        ]

    def test_declared_and_detected_join_on_one_pair_are_one_record(self) -> None:
        rows = _written([
            _declared("customer", "account", [("customer_id", "customer_id")], "d"),
            _detected("customer", "customer_id", "account", "customer_id"),
        ])
        assert [r["name"] for r in rows] == ["customer/customer_id=account/customer_id"]

    def test_composite_key_keeps_declared_predicate_order(self) -> None:
        pairs = [("order_id", "order_id"), ("line_no", "line_no")]
        [row] = _written([_declared("head", "line", pairs, "c")])
        assert [p["left_column"] for p in row["join_predicates"]] == ["order_id", "line_no"]

    def test_old_shape_join_is_skipped_with_a_warning(self, log_lines: list[str]) -> None:
        old = _assertion(
            AssertionPredicate.JOINS_TO, _ref("transfer"),
            {"on_column": "account_id", "to_column": "account_id"},
            object_ref=_ref("account"), id_="old-shape",
        )
        assert _written([old]) == []
        assert any("old-shape" in line for line in log_lines), log_lines

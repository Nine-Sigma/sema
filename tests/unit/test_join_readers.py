"""Join readers: a broken join read fails loudly; join ends come from stored predicates."""
from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock

import pytest
from neo4j.exceptions import ClientError

from sema.log import logger
from sema.models.context import SemanticCandidateSet
from sema.pipeline.context_utils import _build_join_paths
from sema.pipeline.retrieval_utils import _expand_joins

pytestmark = pytest.mark.unit


@pytest.fixture
def log_lines() -> Iterator[list[str]]:
    lines: list[str] = []
    handler_id = logger.add(lambda message: lines.append(str(message)), level="ERROR")
    yield lines
    logger.remove(handler_id)


def _pair(left: str, left_col: str, right: str, right_col: str) -> dict[str, str]:
    return {"left_table": left, "left_column": left_col,
            "right_table": right, "right_column": right_col, "operator": "="}  # fmt: skip


def _join_row(predicates: Any) -> dict[str, Any]:
    """A row in the shape `find_join_paths` returns: no from_table or to_table key."""
    return {"type": "join", "id": "jp-1", "name": "jp", "join_predicates": predicates,
            "hop_count": 1, "cardinality_hint": None, "sql_snippet": None,
            "confidence": 0.9}  # fmt: skip


def _ends(predicates: Any) -> tuple[str, str]:
    candidates = SemanticCandidateSet(query="", candidates=[_join_row(predicates)])
    [path] = _build_join_paths(candidates, max_joins=1)
    return path.from_table, path.to_table


class TestExpandJoinsFailsLoudly:
    def test_a_query_error_raises_and_the_log_names_the_tables(
        self, log_lines: list[str]
    ) -> None:
        engine = MagicMock()
        engine._run_query.side_effect = ClientError("join read failed")
        with pytest.raises(ClientError):
            _expand_joins(engine, ["ledger", "account"])
        assert any("ledger" in line and "account" in line for line in log_lines)

    def test_a_programming_error_is_not_swallowed(self) -> None:
        engine = MagicMock()
        engine._run_query.side_effect = TypeError("bad parameter")
        with pytest.raises(TypeError):
            _expand_joins(engine, ["ledger"])

    def test_no_tables_reads_nothing(self) -> None:
        engine = MagicMock()
        assert _expand_joins(engine, []) == []
        engine._run_query.assert_not_called()


class TestJoinEndsFromStoredPredicates:
    def test_equality_join_keeps_its_stored_orientation(self) -> None:
        # Left sorts after right, so a recomputed or sorted direction would flip it.
        stored = json.dumps([_pair("zone", "zone_id", "address", "zone_id")])
        assert _ends(stored) == ("zone", "address")

    def test_two_column_join_names_both_relations(self) -> None:
        stored = [
            _pair("lot", "block", "parcel", "block"),
            _pair("lot", "lot", "parcel", "lot"),
        ]
        assert _ends(stored) == ("lot", "parcel")

    def test_projection_join_keeps_dotted_relation_names(self) -> None:
        stored = json.dumps([_pair("core.person", "person_id", "core.visit", "person_id")])
        assert _ends(stored) == ("core.person", "core.visit")

    def test_a_join_with_no_predicates_has_empty_ends(self) -> None:
        assert _ends(None) == ("", "")

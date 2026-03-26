from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, call

import pytest

from sema.models.assertions import (
    Assertion,
    AssertionPredicate,
    AssertionStatus,
)

pytestmark = pytest.mark.unit

RUN_ID = "batch-run-1"
TABLE_REF = "unity://cat.sch.tbl"
COL_REF = "unity://cat.sch.tbl.status_code"


def _assertion(
    subject_ref: str,
    predicate: AssertionPredicate,
    payload: dict[str, Any],
    source: str = "llm_interpretation",
    confidence: float = 0.9,
) -> Assertion:
    return Assertion(
        id=f"{predicate.value}-{subject_ref}",
        subject_ref=subject_ref,
        predicate=predicate,
        payload=payload,
        source=source,
        confidence=confidence,
        run_id=RUN_ID,
        observed_at=datetime.now(timezone.utc),
    )


def _make_entity_assertions(count: int = 3) -> list[Assertion]:
    entities = []
    for i in range(count):
        ref = f"unity://cat.sch.tbl{i}"
        entities.append(
            _assertion(
                ref,
                AssertionPredicate.HAS_ENTITY_NAME,
                {"value": f"Entity{i}", "description": f"Desc {i}"},
            )
        )
    return entities


def _make_property_assertions(count: int = 3) -> list[Assertion]:
    props = []
    for i in range(count):
        ref = f"unity://cat.sch.tbl.col{i}"
        props.append(
            _assertion(
                ref,
                AssertionPredicate.HAS_PROPERTY_NAME,
                {"value": f"Property{i}"},
            )
        )
    return props


def test_unwind_entities_single_query() -> None:
    mock_driver = MagicMock()
    from sema.graph.loader import GraphLoader

    loader = GraphLoader(mock_driver)
    loader._run = MagicMock()

    entities = [
        {
            "name": f"Entity{i}",
            "description": f"Desc {i}",
            "source": "llm_interpretation",
            "confidence": 0.9,
            "table_name": "tbl",
            "schema_name": "sch",
            "catalog": "cat",
        }
        for i in range(3)
    ]

    loader.batch_upsert_entities(entities)

    loader._run.assert_called_once()
    cypher_arg = loader._run.call_args[0][0]
    assert "UNWIND" in cypher_arg
    assert "Entity" in cypher_arg or "entity" in cypher_arg.lower()


def test_unwind_properties_single_query() -> None:
    mock_driver = MagicMock()
    from sema.graph.loader import GraphLoader

    loader = GraphLoader(mock_driver)
    loader._run = MagicMock()

    properties = [
        {
            "name": f"Prop{i}",
            "semantic_type": "identifier",
            "source": "llm_interpretation",
            "confidence": 0.9,
            "entity_name": "Patient",
            "column_name": f"col{i}",
            "table_name": "tbl",
            "schema_name": "sch",
            "catalog": "cat",
        }
        for i in range(3)
    ]

    loader.batch_upsert_properties(properties)

    loader._run.assert_called_once()
    cypher_arg = loader._run.call_args[0][0]
    assert "UNWIND" in cypher_arg
    assert "Property" in cypher_arg or "property" in cypher_arg.lower()


def test_batch_materialization_matches_individual() -> None:
    mock_driver = MagicMock()
    from sema.graph.loader import GraphLoader

    individual_loader = GraphLoader(mock_driver)
    individual_loader._run = MagicMock()

    batch_loader = GraphLoader(mock_driver)
    batch_loader._run = MagicMock()

    entities = [
        {
            "name": f"Entity{i}",
            "description": f"Desc {i}",
            "source": "llm_interpretation",
            "confidence": 0.9,
            "table_name": "tbl",
            "schema_name": "sch",
            "catalog": "cat",
        }
        for i in range(3)
    ]

    for e in entities:
        individual_loader.upsert_entity(**e)

    batch_loader.batch_upsert_entities(entities)

    individual_calls = individual_loader._run.call_args_list
    batch_call = batch_loader._run.call_args_list

    assert len(individual_calls) == 3
    assert len(batch_call) == 1

    for ind_call in individual_calls:
        ind_cypher = ind_call[0][0]
        assert "MERGE" in ind_cypher
        assert "Entity" in ind_cypher

    batch_cypher = batch_call[0][0][0]
    assert "MERGE" in batch_cypher
    assert "Entity" in batch_cypher

"""Done-when 3 (N5): who owns an entity whose key is a relation's own row key.

F4: the relation where the key was verified owns the entity, and when a proposed entity key is
another relation's verified key, the entity is that relation's and carries its name. The
identity fixture runs in DuckDB with a mocked proposal per relation; the stand-in graph names
no L2 entity, so the owner's name is its own proposal's.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from tests.support.connector_fixture_identity import IDENTITY_SCHEMA, PROPOSALS, Proposal
from tests.support.connector_fixture_identity_runs import (
    STEP_RUN_ID,
    StepRun,
    prepare_step,
    relation_named,
    relation_of,
    run_step,
)

pytestmark = pytest.mark.unit

ACCEPTED = ("proven", "supported")


def client_proposing(proposals: dict[str, Proposal]) -> MagicMock:
    client = MagicMock()

    def invoke(prompt: str, schema: Any, **kwargs: Any) -> Any:
        proposal = proposals[relation_named(prompt, str(kwargs.get("table_ref", ""))) or ""]
        return schema(entity=proposal.entity, entity_key=list(proposal.entity_key),
                      latest_by=proposal.latest_by, row_key=None)

    client.invoke.side_effect = invoke
    return client


def run(directory: Path, proposals: dict[str, Proposal]) -> StepRun:
    def identity_step(prepared: StepRun) -> None:
        run_identity = importlib.import_module("sema.pipeline.identity").run_identity
        run_identity(prepared.graph, prepared.connector, prepared.config, [IDENTITY_SCHEMA],
                     STEP_RUN_ID, client=client_proposing(proposals))

    return run_step(prepare_step(directory), identity_step)


def accepted_on(step: StepRun, table: str) -> list[dict[str, Any]]:
    return [a.payload for a in step.built().asserted("has_identity")
            if relation_of(a.subject_ref) == table and a.payload.get("verification") in ACCEPTED]


def test_a_row_key_entity_is_shared_with_a_relation_that_holds_its_key(tmp_path: Path) -> None:
    step = run(tmp_path, PROPOSALS)

    [owned] = accepted_on(step, "dock_kind")
    [shared] = accepted_on(step, "berth")

    assert relation_of(owned["owner"]) == relation_of(shared["owner"]) == "dock_kind"
    assert owned["entity"] == shared["entity"] == PROPOSALS["dock_kind"].entity
    assert shared["key_columns"] == ["dock_code"] and shared["identity"] is None


def test_an_entity_keyed_by_another_relations_key_is_that_relations_entity(
    tmp_path: Path,
) -> None:
    proposals = {**PROPOSALS, "berth": Proposal("berth group", ("dock_code",))}

    step = run(tmp_path, proposals)

    [on_berth] = accepted_on(step, "berth")
    assert relation_of(on_berth["owner"]) == "dock_kind"
    assert on_berth["entity"] == PROPOSALS["dock_kind"].entity
    assert on_berth["identity"]["key_columns"] == ["dock_code"]

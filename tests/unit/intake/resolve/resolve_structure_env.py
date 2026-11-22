"""Shared setup for test_resolve_structure.py and the tests that reuse it."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock

from pydantic import BaseModel

from sema.intake.resolve.output_models_utils import ABSTAIN
from tests.unit.intake.model_calls_env import stats

CONFIDENCE = 0.9
REASON = "the label column reads as one flat list"

CONFLICT_SQL = (
    "CREATE SCHEMA tc",
    "CREATE TABLE tc.lib__first AS SELECT * FROM (VALUES ('K1', 'Alder grove'),"
    " ('K2', 'Birch hollow'), ('K3', 'Cedar ridge'), ('K4', 'Dune flat')) t(key, label)",
    "CREATE TABLE tc.lib__second AS SELECT * FROM (VALUES ('K1', 'Elm spire'),"
    " ('K2', 'Fir basin'), ('K3', 'Gorse moor'), ('K4', 'Heath knoll')) t(key, label)",
    "CREATE TABLE tc.lib__values AS SELECT * FROM (VALUES ('K1', 'V1', 'Ivy one'),"
    " ('K1', 'V2', 'Juniper two'), ('K2', 'V3', 'Kelp three'), ('K3', 'V4', 'Larch four'),"
    " ('K3', 'V5', 'Maple five')) t(key, value, label)",
)
SHARED = "tc.lib__values"
CLAIMING = "concept_and_value_set("
FLAT_LIST = "concept_list("

Picker = Callable[[tuple[str, ...]], str]


def resolve() -> Any:
    return importlib.import_module("sema.intake.resolve.resolve")


def entry_fields(output_model: type[BaseModel]) -> list[str]:
    names = [name for name in output_model.model_fields if name.startswith("entry_")]
    return sorted(names, key=lambda name: int(name.removeprefix("entry_")))


def entry_members(output_model: type[BaseModel], field: str) -> tuple[str, ...]:
    enum = output_model.model_json_schema()["properties"][field]["enum"]
    return tuple(value for value in enum if value != ABSTAIN)


def answering(pick: Picker, reason: str = REASON) -> MagicMock:
    """A mocked `LLMClient` that answers every entry of a batch with `pick(members)`."""
    client = MagicMock()
    client.last_stats = stats()
    client.last_path = None

    def _invoke(prompt: str, output_model: type[BaseModel], **_: Any) -> BaseModel:
        client.last_path = "structured_output"
        fields = entry_fields(output_model)
        chosen = {field: pick(entry_members(output_model, field)) for field in fields}
        count = len(fields)
        return output_model(**chosen, confidences=[CONFIDENCE] * count, reasons=[reason] * count)

    client.invoke.side_effect = _invoke
    return client


def starting(prefix: str, *, mentioning: str = "") -> Picker:
    def _pick(members: tuple[str, ...]) -> str:
        found = [m for m in members if m.startswith(prefix) and mentioning in m]
        return found[0] if found else ABSTAIN

    return _pick


def claim_own(members: tuple[str, ...]) -> str:
    """Each concept relation claims the shared value relation; the shared entry abstains."""
    for found in members:
        if found.startswith(CLAIMING) and f"concept_relation={SHARED}" not in found:
            subject = found.split("concept_relation=")[1].split(",")[0]
            if all(subject in m for m in members):
                return found
    return ABSTAIN


def prompts(client: MagicMock) -> list[str]:
    return [str(call.args[0]) for call in client.invoke.call_args_list]


def open_subjects(result: Any) -> set[str]:
    return {a.subject for a in result.ambiguities if a.choice is None}


def chosen(result: Any, subject: str) -> Any:
    (found,) = [a for a in getattr(result, "settled", ()) if a.subject == subject]
    return found

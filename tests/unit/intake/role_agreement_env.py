"""Synthetic relations, a per-call mocked model, and recordings for the agreement tests.

Every relation, column, and value is invented. The mocked model answers call ``n`` from
``asks[n]`` (the last repeats): a role field from the ask's mapping, every reference field
with its first member, each call with its own confidence and reason. Under the S4 enum
model a name outside the enum abstains, so the tests fail on behaviour, not on a crash.
A recordings directory holds a ``recording.json`` and no call record, so every ask misses
the replay and reaches the mocked model, which is how the manifest rule is counted.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple, cast
from unittest.mock import MagicMock

from pydantic import BaseModel

from sema.binding.spec_sections import Provider
from sema.intake.output_models import REFERENCE_PREFIX, ROLE_PREFIX
from sema.intake.output_models_utils import ABSTAIN, NONE
from sema.intake.resolve import resolve_roles
from tests.unit.intake.catalog_env import write_catalog
from tests.unit.intake.model_calls_env import WiredRun, intake_config, stats, wired_run
from tests.unit.intake.role_entries_env import concept_list
from tests.unit.intake.role_map_env import ROLES_STAGE, ROWS, Scenario, scenario
from tests.unit.intake.structure_env import fresh_tools

AGREEMENT_ENV = "INTAKE_MODEL_AGREEMENT"
MANIFEST_FILE = "recording.json"
# The key names of a committed recording manifest (values invented).
MANIFEST: dict[str, Any] = {
    "source": "synthetic", "provider": "openrouter", "model": "model-under-test",
    "prompt_versions": ["roles v2"], "calls": 0, "tokens": 0, "cost": 0.0,
    "approver": None, "value_asks": 0,
}
WIDE, OTHER = "rm.wide", "rm.other"
WIDE_COLUMNS = ("zorb", "quax", "vell", "tarn", "mox", "pell")
OTHER_COLUMNS = ("kelt", "wim", "rusk", "fenn", "lusk", "brin")
LONG = "' says a long sentence about item number ' || CAST(i AS VARCHAR)"


class Ask(NamedTuple):
    roles: Mapping[str, str]
    confidence: float
    reason: str


def _wide(name: str, columns: Sequence[str]) -> str:
    key, label, *texts = columns
    expressions = {
        key: f"'{key[0].upper()}' || lpad(CAST(i AS VARCHAR), 2, '0')",
        label: f"'{label.title()} entry ' || i",
        **{text: f"'{text.title()}'{LONG}" for text in texts},
    }
    selected = ", ".join(f"{expression} AS {column}" for column, expression in expressions.items())
    return f"CREATE TABLE {name} AS SELECT {selected} FROM range(1, {ROWS + 1}) r(i)"


def wide(root: Path, name: str = WIDE, columns: Sequence[str] = WIDE_COLUMNS) -> Scenario:
    """A concepts relation of six columns no lexicon token names: every role is open."""
    return scenario([_wide(name, columns)], [concept_list(name, "concepts")], root)


def answers_for(columns: Sequence[str]) -> dict[str, str]:
    return {"concept_id": columns[0], "name": columns[1]}


def _enum(output_model: type[BaseModel], name: str) -> tuple[str, ...] | None:
    enum = output_model.model_json_schema()["properties"][name].get("enum")
    return None if enum is None else tuple(enum)


def _answer(ask: Ask, output_model: type[BaseModel]) -> dict[str, str]:
    chosen: dict[str, str] = {}
    for name in output_model.model_fields:
        enum = _enum(output_model, name) if name not in ("confidences", "reasons") else None
        if name.startswith(ROLE_PREFIX):
            wanted = ask.roles.get(name.removeprefix(ROLE_PREFIX))
            chosen[name] = (wanted or NONE) if enum is None else (
                wanted if wanted in enum else ABSTAIN
            )
        elif name.startswith(REFERENCE_PREFIX) and enum is not None:
            chosen[name] = next((m for m in enum if m != ABSTAIN), ABSTAIN)
    return chosen


def asking(*asks: Ask) -> MagicMock:
    """A mocked client; the output models it was handed are kept on ``client.models``."""
    client = MagicMock()
    client.last_stats = stats()
    client.last_path = None
    client.models = []

    def _invoke(prompt: str, output_model: type[BaseModel], **_: Any) -> BaseModel:
        client.last_path = "structured_output"
        client.models.append(output_model)
        ask = asks[min(len(client.models), len(asks)) - 1]
        chosen = _answer(ask, output_model)
        count = len(chosen)
        return output_model(
            **chosen, confidences=[ask.confidence] * count, reasons=[ask.reason] * count
        )

    client.invoke.side_effect = _invoke
    return client


def resolve_under(scene: Scenario, client: MagicMock, **config: Any) -> tuple[Any, WiredRun]:
    """``resolve_roles`` over ``scene`` with an intake config carrying ``config``."""
    wired = wired_run(scene.root / "calls", client, config=intake_config(**config))
    tools = fresh_tools(scene.source, scene.root / "resolve", stage=ROLES_STAGE,
                        structure=scene.structure)
    result = resolve_roles(
        scene.roles, scene.structure, scene.source.land, tools, wired.calls,
        provider=cast(Provider, scene.source.provider), **scene.kwargs,
    )
    return result, wired


def calls_with(client: MagicMock, prefix: str) -> list[str]:
    """The prompts of the calls whose output model has a field starting ``prefix``."""
    return [
        str(call.args[0])
        for call, model in zip(client.invoke.call_args_list, client.models, strict=True)
        if any(name.startswith(prefix) for name in model.model_fields)
    ]


def column_order(prompt: str, columns: Sequence[str]) -> list[str]:
    """``columns`` in the order their fact lines appear in ``prompt``."""
    lines = prompt.splitlines()
    first = {
        column: next(at for at, line in enumerate(lines) if line.startswith(f"- {column}:"))
        for column in columns
    }
    return sorted(columns, key=first.__getitem__)


def permutation(first: Sequence[str], second: Sequence[str]) -> tuple[int, ...]:
    return tuple(first.index(column) for column in second)


def agreement_of(entry: Any) -> Any:
    return entry.model_dump().get("agreement")


def disagreements(result: Any, relation: str) -> Any:
    """The relation's disagreement count, under whatever field name holds it."""
    fields = [f.name for f in dataclasses.fields(result) if "disagree" in f.name]
    assert fields, [f.name for f in dataclasses.fields(result)]
    held = getattr(result, fields[0])
    return held.get(relation) if isinstance(held, Mapping) else held


def recordings(root: Path, source: str, agreement: bool | None) -> Path:
    """A recordings directory for ``source`` with a manifest and no call record;
    ``agreement=None`` leaves the field out."""
    directory = root / "recordings" / source
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {**MANIFEST, "source": source}
    if agreement is not None:
        manifest["agreement"] = agreement
    (directory / MANIFEST_FILE).write_text(json.dumps(manifest), encoding="utf-8")
    return directory


def catalog_source(root: Path) -> Any:
    """A small workbook whose run reaches Review with no model call."""
    from scripts.record_intake_run_utils import RecordedSource

    workbook, descriptor = write_catalog(root / "inputs", status=False, readings=False)
    return RecordedSource(name="synthetic", path=workbook, descriptor=descriptor)

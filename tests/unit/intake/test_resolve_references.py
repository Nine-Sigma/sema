"""Per-property reference entries settled by a mocked model: one call per relation.

A dictionary property whose text names two code lists leaves its ``domain`` entry open
with both lists and ``none`` as members. After the column roles, each relation with open
reference entries is one call whose output model is ``ReferenceChoice_v1``: one enum field
per open entry over that entry's own members, with positional confidences and reasons. A
reference choice is never inherited through signature memory.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from sema.intake.ledger_utils import prompt_hash
from sema.intake.output_models import REFERENCE_CHOICE
from sema.intake.output_models_utils import ABSTAIN
from tests.unit.intake.model_calls_env import MODEL, WiredRun, wired_run
from tests.unit.intake.resolve_roles_env import (
    CONFIDENCE,
    DICTIONARY,
    REASON,
    REFERENCE_TABLES,
    SLANT,
    Synthetic,
    answering,
    distinct,
    enum_of,
    lines_with,
    output_models,
    prompts,
    reference_fields,
    relation,
)
from tests.unit.intake.structure_env import fresh_tools

pytestmark = pytest.mark.unit

PROMPT_VERSION = "# prompt: roles v1"
DOMAIN = "domain"
SUBJECT = f"{relation(DICTIONARY)}.{DOMAIN}.{SLANT}"


def resolve() -> Any:
    return importlib.import_module("sema.intake.resolve")


@pytest.fixture(scope="module")
def references(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Synthetic]:
    root = tmp_path_factory.mktemp("resolve_references")
    built = Synthetic(REFERENCE_TABLES, root, dictionaries=frozenset({DICTIONARY}))
    yield built
    built.close()


def _resolved(built: Synthetic, client: MagicMock, root: Path) -> tuple[Any, WiredRun]:
    wired = wired_run(root / "calls", client)
    tools = fresh_tools(
        built.source, root / "ledger" / "resolve", stage="roles", structure=built.structure
    )
    result = resolve().resolve_roles(
        built.roles, built.structure, built.source.land, tools, wired.calls,
        provider=built.source.provider,
    )
    return result, wired


def _reference_calls(client: MagicMock) -> list[tuple[str, Any]]:
    sent = zip(prompts(client), output_models(client), strict=True)
    return [(prompt, model) for prompt, model in sent if model.__name__ == REFERENCE_CHOICE]


def _domain(result: Any) -> Any:
    (found,) = [
        e for e in result.eligibilities
        if e.relation == relation(DICTIONARY) and e.role == DOMAIN and e.subject == SLANT
    ]
    return found


def _entry(result: Any) -> Any:
    (found,) = [a for a in result.ambiguities if a.subject == SUBJECT]
    return found


def test_the_open_domain_entry_starts_open_with_both_lists(references: Synthetic) -> None:
    opened = _entry(references.roles)

    assert opened.choice is None
    assert set(opened.members) == {"ribs", "fens", "none"}


def test_open_domain_entry_is_settled_by_one_reference_call(
    references: Synthetic, tmp_path: Path
) -> None:
    client = answering(distinct(), reference=lambda members: "fens")

    result, wired = _resolved(references, client, tmp_path)

    ((prompt, model),) = _reference_calls(client)
    (field,) = reference_fields(model)
    assert set(enum_of(model, field)) == {"ribs", "fens", "none"}
    assert ABSTAIN in model.model_json_schema()["properties"][field]["enum"]
    assert prompt.splitlines()[0] == PROMPT_VERSION
    assert lines_with(prompt, SLANT, DOMAIN)
    assert lines_with(prompt, "Coded by the ribs list or by the fens list")
    chosen = _entry(result)
    assert (chosen.choice, chosen.source, chosen.confidence) == ("fens", "model", CONFIDENCE)
    assert chosen.reason == REASON
    assert chosen.call_key == prompt_hash(MODEL, prompt, REFERENCE_CHOICE)
    assert f"{chosen.call_key}.json" in wired.call_files()
    settled = _domain(result)
    assert (settled.choice, settled.source, settled.confidence) == ("fens", "model", CONFIDENCE)


def test_abstained_reference_stays_open_with_the_reason(
    references: Synthetic, tmp_path: Path
) -> None:
    client = answering(distinct(), reference=lambda members: ABSTAIN)

    result, _ = _resolved(references, client, tmp_path)

    assert len(_reference_calls(client)) == 1
    left = _entry(result)
    assert (left.choice, left.source, left.call_key) == (None, None, None)
    assert left.reason == REASON
    assert _domain(result).choice is None

"""Role ambiguities settled by a mocked model: one call per relation, memory, verify, retry.

Pinned API: ``sema.intake.resolve.resolve.resolve_roles(roles, structure, land, tools, calls, *,
provider="staging", max_retries=1) -> RolesResult``. Open entries are read from the input
``roles.ambiguities`` (an entry with no choice is open), as ``resolve_structure`` reads
its structure. Each relation with open roles is one call whose output model is the role
map ``RoleMap..._v1`` over every role of the relation's role models (S5 decision 1). A
choice is written onto its entry (``choice``, ``confidence``, ``source: model``,
``call_key``) and its eligibility settles
with ``source: model``; the settled choices are fed back to Roles as stated choices, so a
relation sharing the signature inherits them with ``source: memory`` and makes no call.
The Roles verifier re-runs on each answer; a failing answer is re-asked with every earlier
failure, at most ``max_retries`` times, then the failing entries are left open with the
failure as their reason. ``sema.intake.resolve.resolve`` is imported lazily so a missing function
fails each test on its own.

A display column's empty-string rate is a finding (S5 decision 18): the choice stands
and the finding is recorded on its entry.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from sema.intake.calls.ledger_utils import prompt_hash
from sema.intake.resolve.output_models import REFERENCE_CHOICE, ROLE_MAP_PREFIX, role_field
from sema.intake.resolve.output_models_utils import ABSTAIN
from sema.intake.verify.roles import verify_roles
from tests.unit.intake.eligibility_env import Env, by_role
from tests.unit.intake.model_calls_env import MODEL, WiredRun, intake_config, wired_run
from tests.unit.intake.resolve_roles_env import (
    CONFIDENCE,
    CONFLICT_STATUS,
    CONFLICT_TABLES,
    ENTITY_ROLES,
    MEMORY_TABLES,
    VALUE_TABLES,
    Synthetic,
    Tables,
    answering,
    asked_columns,
    code_table_values,
    distinct,
    entry,
    enum_of,
    lines_with,
    opened,
    output_models,
    prompts,
    relation,
    role_fields,
    schema_text,
    shared_column,
)
from tests.unit.intake.structure_env import fresh_tools, sheet, sql_lines

pytestmark = pytest.mark.unit

PROMPT_VERSION = "# prompt: roles v2"
COLUMN_INFO = "Column_Info"
# Column-role calls plus one reference call per relation with open reference entries.
ACRIS_CALL_BUDGET = 40
AcrisRun = tuple[Any, MagicMock, Path, str]


def resolve() -> Any:
    return importlib.import_module("sema.intake.resolve.resolve")


@pytest.fixture(scope="module")
def tmp_module(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("resolve_roles")


def _synthetic(tables: Tables, root: Path) -> Iterator[Synthetic]:
    built = Synthetic(tables, root)
    yield built
    built.close()


@pytest.fixture(scope="module")
def memory(tmp_module: Path) -> Iterator[Synthetic]:
    yield from _synthetic(MEMORY_TABLES, tmp_module / "memory")


@pytest.fixture(scope="module")
def conflict(tmp_module: Path) -> Iterator[Synthetic]:
    yield from _synthetic(CONFLICT_TABLES, tmp_module / "conflict")


@pytest.fixture(scope="module")
def blanks(tmp_module: Path) -> Iterator[Synthetic]:
    yield from _synthetic(VALUE_TABLES, tmp_module / "blanks")


@pytest.fixture(scope="module")
def env(tmp_module: Path) -> Iterator[Env]:
    built = Env(tmp_module / "acris")
    yield built
    built.close()


def _resolved(
    built: Synthetic, client: MagicMock, root: Path, **kwargs: Any
) -> tuple[Any, WiredRun]:
    wired = wired_run(root / "calls", client)
    tools = fresh_tools(
        built.source, root / "ledger" / "resolve", stage="roles", structure=built.structure
    )
    result = resolve().resolve_roles(
        built.roles, built.structure, built.source.land, tools, wired.calls,
        provider=built.source.provider, **kwargs,
    )
    return result, wired


def test_one_call_per_relation_signature_over_its_open_roles(
    memory: Synthetic, tmp_path: Path
) -> None:
    client = answering(distinct())

    _resolved(memory, client, tmp_path)

    models = output_models(client)
    assert len(models) == 2
    assert all(model.__name__.startswith(ROLE_MAP_PREFIX) for model in models)
    expected = {role_field("code"), role_field("display")}
    assert all(expected <= set(role_fields(model)) for model in models)
    columns = [tuple(columns) for columns, _ in MEMORY_TABLES.values()]
    for prompt in prompts(client):
        assert asked_columns(prompt) in columns, asked_columns(prompt)


def test_prompt_states_each_roles_data_fact_members(
    memory: Synthetic, tmp_path: Path
) -> None:
    client = answering(distinct())

    _resolved(memory, client, tmp_path)

    (prompt,) = [p for p in prompts(client) if "mox" in asked_columns(p)]
    (code,) = lines_with(prompt, f"`{role_field('code')}`")
    assert "key_unique_non_null" in code and "`tarn`" in code and "`vell`" in code
    assert "mox" not in code
    (display,) = lines_with(prompt, f"`{role_field('display')}`")
    assert "data fact" not in display


def test_model_choice_is_recorded_and_inherited_by_memory(
    memory: Synthetic, tmp_path: Path
) -> None:
    client = answering(distinct())

    result, wired = _resolved(memory, client, tmp_path)

    sent = zip(prompts(client), output_models(client), strict=True)
    keys = {prompt_hash(MODEL, prompt, model.__name__) for prompt, model in sent}
    first = relation("alpha")
    settled = by_role(result.eligibilities, first)
    pair = (role_field("code"), role_field("display"))
    (answer,) = [a for a in client.answers if {a[f] for f in pair} <= {"zorb", "quax"}]
    for role in ("code", "display"):
        chosen = entry(result, first, role)
        assert chosen.choice == answer[role_field(role)]
        assert (chosen.source, chosen.confidence) == ("model", CONFIDENCE)
        assert chosen.call_key in keys
        assert f"{chosen.call_key}.json" in wired.call_files()
        assert (settled[role].choice, settled[role].source) == (chosen.choice, "model")
        for twin in (relation("beta"), relation("gamma")):
            inherited = by_role(result.eligibilities, twin)[role]
            assert (inherited.choice, inherited.source) == (chosen.choice, "memory")
            assert entry(result, twin, role).source == "memory"
    tools = fresh_tools(
        memory.source, tmp_path / "verify", stage="roles", structure=memory.structure
    )
    verdict = verify_roles(result, tools, spec=None)
    assert verdict.status == "pass", verdict.failures


def test_two_roles_on_one_column_are_retried_once_then_left_open(
    conflict: Synthetic, tmp_path: Path
) -> None:
    zeta = relation("zeta")
    client = answering(shared_column)

    result, _ = _resolved(conflict, client, tmp_path)

    sent = prompts(client)
    assert len(sent) == 2
    assert len(set(sent)) == 2
    for retry in sent[1:]:
        assert lines_with(retry, "one_role_per_column", "model"), retry
        assert lines_with(retry, "one_role_per_column", "zorb")
        assert lines_with(retry, "code") and lines_with(retry, "display")
    for prompt in sent:
        assert lines_with(prompt, "status", "pell"), "the claim in force is not stated"
    for role in ("code", "display"):
        left = entry(result, zeta, role)
        assert (left.choice, left.source, left.call_key) == (None, None, None)
        assert left.reason is not None and "one_role_per_column" in left.reason
    assert not [e for e in result.eligibilities if e.source == "model" and e.column]
    assert by_role(result.eligibilities, zeta)["status"].choice == CONFLICT_STATUS


def test_retries_stop_at_max_retries(conflict: Synthetic, tmp_path: Path) -> None:
    client = answering(shared_column)

    _resolved(conflict, client, tmp_path, max_retries=0)

    assert client.invoke.call_count == 1


def test_a_display_over_blanks_stands_with_its_finding(
    blanks: Synthetic, tmp_path: Path
) -> None:
    eta = relation("eta")
    client = answering(distinct({"display": "wim"}))

    result, _ = _resolved(blanks, client, tmp_path)

    sent = prompts(client)
    assert asked_columns(sent[0]) == ("kelt", "wim", "rusk")
    assert len(sent) == 1
    assert lines_with(sent[0], "kelt", "code"), "the code's data fact is not stated"
    chosen = entry(result, eta, "display")
    assert (chosen.choice, chosen.source) == ("wim", "model")
    assert [hint for hint in chosen.hints if "display_null_rate" in hint and "wim" in hint]
    assert by_role(result.eligibilities, eta)["display"].source == "model"
    assert by_role(result.eligibilities, eta)["code"].choice == "kelt"


@pytest.fixture(scope="module")
def acris_run(env: Env, tmp_module: Path) -> AcrisRun:
    """ACRIS with every Column Info relation's entity roles open, resolved once."""
    source, roles = env.acris.source, env.a10_roles
    dictionaries = sheet(source, COLUMN_INFO)
    assert len(dictionaries) == 15
    (first,) = [
        d for d in dictionaries if by_role(roles.eligibilities, d)["property"].source == "model"
    ]
    oracle_map = by_role(roles.eligibilities, first)
    client = answering(distinct({role: str(oracle_map[role].choice) for role in ENTITY_ROLES}))
    root = tmp_module / "acris_resolve"
    config = intake_config(max_calls=ACRIS_CALL_BUDGET, agreement=False)
    wired = wired_run(root / "calls", client, config)
    tools = fresh_tools(source, root / "ledger", stage="roles", structure=env.a10)
    result = resolve().resolve_roles(
        opened(roles, set(dictionaries)), env.a10, source.land, tools, wired.calls
    )
    return result, client, root / "ledger", first


def _column_info_prompts(client: MagicMock) -> list[str]:
    """Prompts of the calls over entity roles, which only the Column Info relations have."""
    fields = {role_field(role) for role in ENTITY_ROLES}
    sent = zip(prompts(client), output_models(client), strict=True)
    return [prompt for prompt, model in sent if fields & set(role_fields(model))]


def test_column_info_relations_make_one_call_and_fourteen_inherit(
    env: Env, acris_run: AcrisRun
) -> None:
    result, client, _, first = acris_run

    assert len(_column_info_prompts(client)) == 1
    settled = by_role(result.eligibilities, first)
    assert {settled[role].source for role in ENTITY_ROLES} == {"model"}
    others = set(sheet(env.acris.source, COLUMN_INFO)) - {first}
    assert len(others) == 14
    for dictionary in others:
        inherited = by_role(result.eligibilities, dictionary)
        for role in ENTITY_ROLES:
            assert inherited[role].choice == settled[role].choice, (dictionary, role)
            if len(inherited[role].members) > 1:
                assert inherited[role].source == "memory", (dictionary, role)
                assert entry(result, dictionary, role).source == "memory"


def test_column_info_prompt_holds_schema_text_and_no_non_schema_value(
    env: Env, acris_run: AcrisRun, tmp_path: Path
) -> None:
    _, client, ledger, first = acris_run
    (prompt,) = _column_info_prompts(client)
    allowed, column_text, dataset_text = schema_text(env, first, tmp_path)

    assert prompt.splitlines()[0] == PROMPT_VERSION
    assert column_text in prompt
    assert dataset_text in prompt
    values = {value for value in code_table_values(env) if value not in allowed}
    assert len(values) > 50
    assert sorted(value for value in values if value in prompt) == []
    outside = [
        line for line in sql_lines(ledger) if line["value_bearing"] and not line["schema_relation"]
    ]
    assert outside == []


def test_column_info_open_references_make_one_call_per_relation(
    env: Env, acris_run: AcrisRun
) -> None:
    """After the column roles settle, each relation's open reference entries are one call."""
    result, client, _, _ = acris_run
    opened_refs = {
        (e.relation, e.role, e.subject)
        for e in env.a10_roles.eligibilities
        if e.subject is not None and e.choice is None and len(e.members) > 1
    }
    expected = {relation for relation, _, _ in opened_refs}
    sent = zip(prompts(client), output_models(client), strict=True)
    keys = [
        prompt_hash(MODEL, prompt, REFERENCE_CHOICE)
        for prompt, model in sent
        if model.__name__ == REFERENCE_CHOICE
    ]

    assert expected
    assert len(keys) == len(set(keys)) == len(expected)
    by_key: dict[str, set[str]] = {}
    for e in result.eligibilities:
        if (e.relation, e.role, e.subject) in opened_refs:
            assert e.source == "model", (e.relation, e.role, e.subject)
            listed = entry(result, e.relation, f"{e.role}.{e.subject}")
            by_key.setdefault(str(listed.call_key), set()).add(e.relation)
    assert set(by_key) == set(keys)
    assert all(len(relations) == 1 for relations in by_key.values())
    assert not [e for e in result.eligibilities if e.subject is not None and e.source == "memory"]

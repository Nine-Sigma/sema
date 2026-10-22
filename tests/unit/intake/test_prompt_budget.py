"""Prompts fit a policy budget (intake-model-calls, accepted 2026-10-21).

``MAX_PROMPT_TOKENS`` sits beside ``ModelCalls``. A prompt over it is compacted in a
stated order: an entry offering every column but its exclusions becomes one statement
(its output enum unchanged), then schema text is read in a stated order and capped per
relation with a line naming the rows not shown, then only columns an open entry offers
or a claim in force holds keep full facts and the rest are named. A prompt that still
does not fit makes no provider call, is not counted, logs ``call_refused`` with its key
and size, and leaves its entries open. A prompt under the budget renders as before.

Every width and row count is computed from the constant, never from a fixture. A
prompt's size is what reaches the provider: the client's own estimate over the rendered
prompt plus its output schema JSON.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path
from typing import Any

import pytest

from sema.intake.ledger_utils import prompt_hash
from sema.intake.output_models import ROLES_CHOICE
from tests.unit.intake.model_calls_env import MODEL
from tests.unit.intake.prompt_budget_env import (
    LABELS,
    REFUSED_EVENT,
    Built,
    Watched,
    abstaining,
    budget,
    chars,
    code_list,
    column_name,
    dictionary,
    dictionary_key,
    dictionary_table,
    enum_of,
    lines_naming,
    measured,
    mixed,
    output_models,
    prompts,
    relation,
    resolve,
    resolved_roles,
    text_table,
    size,
    wide,
    width_over,
)
from tests.unit.intake.structure_env import fresh_tools, run

pytestmark = pytest.mark.unit

OPEN_ROLES = ("code", "display")
MEMBER_LINE = len("- `col_0000`\n")
STATEMENT_ALLOWANCE = 400
NAMED_ALLOWANCE = len("col_0000, ")
DICTIONARY_ROW = len(
    "- k000000 | generic label 000000 | a generic sentence describing entry 000000\n"
)
KEY = re.compile(r"\bk\d{6}\b")
CONSTANT_LINE = re.compile(r"^MAX_PROMPT_TOKENS(: int)? = [\d_]+\s*(#.*)?$", re.MULTILINE)
# Captured at HEAD f945592 for prompts far under any budget; compaction must not touch them.
UNDER_BUDGET_KEYS = {
    "wide_6": "dd11a935a369c48d26144f435313bbaaf237a4b888b3c4eada5bcf5a532474b5",
    "dictionary_5_descending": (
        "a4f558c403827a3958bc163669a466334d2ca5a8c2c928d3b18a424d80c32cf6"
    ),
}


@pytest.fixture(scope="module")
def wide_shape(tmp_path_factory: pytest.TempPathFactory) -> tuple[int, int]:
    return measured(wide, tmp_path_factory.mktemp("wide_shape"))


@pytest.fixture(scope="module")
def mixed_shape(tmp_path_factory: pytest.TempPathFactory) -> tuple[int, int]:
    return measured(mixed, tmp_path_factory.mktemp("mixed_shape"))


def numbered(prompt: str, count: int) -> list[str]:
    """Lines stating ``count`` as a number of its own, not as part of a name or key."""
    pattern = re.compile(rf"(?<![\w.]){count}(?![\w.])")
    return [line for line in prompt.splitlines() if pattern.search(line)]


def facts(prompt: str, name: str) -> list[str]:
    return [line for line in lines_naming(prompt, name) if "type" in line]


def compacted_wide_width(shape: tuple[int, int]) -> int:
    """A width whose full rendering exceeds the budget while one statement per entry fits."""
    fixed, per_column = shape
    width = width_over(fixed, per_column, 1.05)
    compacted = fixed + width * (per_column - len(OPEN_ROLES) * MEMBER_LINE)
    if compacted + len(OPEN_ROLES) * STATEMENT_ALLOWANCE > chars(budget()) * 0.97:
        pytest.fail(f"no width separates full {per_column}/column from compacted at this budget")
    return width


def roles_run(built: Built, root: Path, run_id: str = "run-a") -> tuple[Watched, Any]:
    client = abstaining()
    watch = Watched(root, client, run_id)
    return watch, resolved_roles(built, built.roles, watch, root)


def dictionary_rows(share: float) -> int:
    return int(chars(budget()) * share // DICTIONARY_ROW) + 1


def dictionary_built(rows: int, root: Path, descending: bool) -> Built:
    statement = dictionary_table("dict", rows, descending)
    return Built([statement], root, lambda: [dictionary("dict")])


def shown_keys(prompt: str) -> set[str]:
    return set(KEY.findall(prompt))


def test_budget_is_a_policy_constant_beside_model_calls() -> None:
    from sema.intake import model_calls

    assert budget() == model_calls.MAX_PROMPT_TOKENS
    assert CONSTANT_LINE.search(inspect.getsource(model_calls)), "a literal, not derived"


def test_wide_relation_entries_become_statements_and_fit_one_call(
    wide_shape: tuple[int, int], tmp_path: Path
) -> None:
    width = compacted_wide_width(wide_shape)
    built = wide(width, tmp_path / "source")
    watch, _ = roles_run(built, tmp_path)
    client = watch.wired.client

    assert client.invoke.call_count == 1
    (prompt,), (output_model,) = prompts(client), output_models(client)
    assert size(prompt, output_model) <= budget()
    columns = tuple(column_name(i) for i in range(width))
    for role in OPEN_ROLES:
        assert f"{relation('wide')}.{role}" in prompt
        assert enum_of(output_model, f"role_{role}") == columns
    for name in columns:
        assert len(facts(prompt, name)) == 1, name
        assert len(lines_naming(prompt, name)) <= 2, name
    assert watch.wired.row().model_calls == 1
    built.close()


def test_compacted_prompt_and_key_are_the_same_in_two_runs(
    wide_shape: tuple[int, int], tmp_path: Path
) -> None:
    width = compacted_wide_width(wide_shape)
    built = wide(width, tmp_path / "source")
    runs = [roles_run(built, tmp_path / run_id, run_id)[0] for run_id in ("run-a", "run-b")]
    built.close()

    rendered = [prompts(watch.wired.client) for watch in runs]
    assert all(len(found) == 1 for found in rendered)
    assert size(rendered[0][0], output_models(runs[0].wired.client)[0]) <= budget()
    assert rendered[0] == rendered[1]
    assert runs[0].keys() == runs[1].keys()


def test_schema_text_is_read_in_a_stated_order_and_capped_per_relation(
    tmp_path: Path,
) -> None:
    rows = dictionary_rows(1.2)
    rendered, models = [], []
    for descending in (False, True):
        root = tmp_path / ("descending" if descending else "ascending")
        built = dictionary_built(rows, root / "source", descending)
        watch, _ = roles_run(built, root)
        built.close()
        assert watch.wired.client.invoke.call_count == 1
        rendered += prompts(watch.wired.client)
        models += output_models(watch.wired.client)

    prompt = rendered[0]
    assert rendered[1] == prompt, "the shown rows depend on the order rows were stored in"
    assert size(prompt, models[0]) <= budget()
    shown = len(shown_keys(prompt))
    assert 0 < shown < rows
    assert [line for line in numbered(prompt, rows - shown) if "not shown" in line]
    for index in range(3):
        assert len(facts(prompt, column_name(index))) == 1


def test_only_offered_and_claimed_columns_keep_full_facts(
    mixed_shape: tuple[int, int], tmp_path: Path
) -> None:
    fixed, per_column = mixed_shape
    numbers = width_over(fixed, per_column, 1.2)
    if fixed + numbers * NAMED_ALLOWANCE + STATEMENT_ALLOWANCE > chars(budget()) * 0.9:
        pytest.fail("no numeric width separates full facts from names at this budget")
    built = mixed(numbers, tmp_path / "source")
    watch, _ = roles_run(built, tmp_path)
    built.close()

    assert watch.wired.client.invoke.call_count == 1
    (prompt,), (output_model,) = prompts(watch.wired.client), output_models(watch.wired.client)
    assert size(prompt, output_model) <= budget()
    for index in range(LABELS + 1):
        assert len(facts(prompt, column_name(index))) == 1, "offered or in force"
    for index in range(LABELS + 1, LABELS + 1 + numbers):
        name = column_name(index)
        assert lines_naming(prompt, name), f"{name} is not named"
        assert not facts(prompt, name), f"{name} kept its facts"
    assert numbered(prompt, numbers), "the omission of the other columns is not stated"


def test_prompt_that_cannot_fit_is_refused_without_a_call(
    wide_shape: tuple[int, int], tmp_path: Path
) -> None:
    fixed, per_column = wide_shape
    compacted_column = per_column - len(OPEN_ROLES) * MEMBER_LINE
    width = int(chars(budget()) * 1.1 // compacted_column) + 1
    built = wide(width, tmp_path / "source")
    watch, result = roles_run(built, tmp_path)
    built.close()

    assert watch.wired.client.invoke.call_count == 0
    keys = watch.keys()
    assert len(keys) == 1, "a refused prompt was asked again"
    row = watch.wired.row()
    assert (row.model_calls, row.cost) == (0, 0.0)
    assert watch.wired.call_files() == []
    (refused,) = watch.wired.events(REFUSED_EVENT)
    payload = refused["payload"]
    assert payload["prompt_hash"] == keys[0]
    numbers = [v for v in payload.values() if isinstance(v, int) and not isinstance(v, bool)]
    assert any(number > budget() for number in numbers), payload
    for role in OPEN_ROLES:
        (left,) = [a for a in result.ambiguities if a.subject == f"{relation('wide')}.{role}"]
        assert (left.choice, left.source, left.call_key) == (None, None, None)
        assert left.reason is not None and "budget" in left.reason


def test_structure_prompt_shares_the_schema_text_cap(tmp_path: Path) -> None:
    rows = dictionary_rows(1.2)
    built = dictionary_built(rows, tmp_path / "source", descending=True)
    source = built.source
    found = run(source, fresh_tools(source, tmp_path / "structure"))
    assert [a.subject for a in found.ambiguities] == [relation("dict")]
    # The bound dictionary is what lets the run read its text as schema.
    readable = found.model_copy(update={"bound": (dictionary("dict"),), "ambiguities": ()})
    tools = fresh_tools(source, tmp_path / "resolve", stage="resolve_structure", structure=readable)
    client = abstaining()
    watch = Watched(tmp_path, client, "run-a")
    try:
        resolve().resolve_structure(found, source.land, tools, watch.wired.calls,
                                    provider=source.provider)
    finally:
        watch.stop()
        built.close()

    assert client.invoke.call_count == 1
    (prompt,), (output_model,) = prompts(client), output_models(client)
    assert size(prompt, output_model) <= budget()
    shown = len(shown_keys(prompt))
    assert 0 < shown < rows
    assert [line for line in numbered(prompt, rows - shown) if "not shown" in line]


@pytest.mark.parametrize("name", sorted(UNDER_BUDGET_KEYS))
def test_prompt_under_the_budget_renders_as_before(name: str, tmp_path: Path) -> None:
    if name == "wide_6":
        built = Built([text_table("wide", 6)], tmp_path, lambda: [code_list("wide")])
    else:
        built = dictionary_built(5, tmp_path, descending=True)
    watch, _ = roles_run(built, tmp_path)
    built.close()

    (prompt,) = prompts(watch.wired.client)
    assert prompt_hash(MODEL, prompt, ROLES_CHOICE) == UNDER_BUDGET_KEYS[name], prompt

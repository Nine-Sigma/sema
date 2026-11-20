"""Prompts fit a policy budget (intake-model-calls, accepted 2026-10-21).

``MAX_PROMPT_TOKENS`` sits beside ``ModelCalls``. A role-map prompt shows a fixed sample
of each schema relation's text, read in a stated order, with a line naming the rows not
shown (S5 decision 6). A prompt over the budget is compacted in the delta's order, each
omission stated: comments are truncated, then the schema sample shrinks to its first
rows, then every column keeps only its name and type. A prompt that still does not fit
makes no provider call, is not counted, logs ``call_refused`` with its key and size, and
leaves its entries open. A prompt under the budget renders in full. The Structure prompt
keeps its own ladder (``resolve_budget_utils``).

Every width and row count is computed from the constant, never from a fixture. A
prompt's size is what reaches the provider: the client's own estimate over the rendered
prompt plus its output schema JSON.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from sema.intake.resolve.output_models import ROLE_MAP_PREFIX
from sema.intake.resolve.resolve_utils import member_bindings
from sema.intake.structure.shapes import concept_list, path_codes
from sema.intake.structure.shapes.base import profile_relations
from sema.intake.structure.shapes.precedence import STAGE
from sema.intake.verify.base import Ambiguity
from tests.unit.intake.resolve.prompt_budget_env import (
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
    lines_naming,
    long_dictionary_table,
    measured,
    mixed,
    output_models,
    prompts,
    relation,
    resolve,
    resolved_roles,
    schema_json,
    text_table,
    size,
    wide,
    width_over,
)
from tests.unit.intake.structure_env import fresh_tools, run

pytestmark = pytest.mark.unit

OPEN_ROLES = ("code", "display")
OFFERED = (concept_list.ID, path_codes.ID)
NAME_AND_TYPE = len("- col_0000: type VARCHAR\n")
OMITTED = "columns show only their name and type"
STATEMENT_ALLOWANCE = 400
NAMED_ALLOWANCE = len("col_0000, ")
DICTIONARY_ROW = len(
    "- k000000 | generic label 000000 | a generic sentence describing entry 000000\n"
)
KEY = re.compile(r"\bk\d{6}\b")
CONSTANT_LINE = re.compile(r"^MAX_PROMPT_TOKENS(: int)? = [\d_]+\s*(#.*)?$", re.MULTILINE)
# Prompts far under any budget, by name: their column count and schema rows shown.
UNDER_BUDGET_SHAPES = {"wide_6": (6, 0), "dictionary_5_descending": (3, 5)}


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
    """A width whose full rendering exceeds the budget while name and type per column fit."""
    fixed, per_column = shape
    width = width_over(fixed, per_column, 1.05)
    if fixed + width * NAME_AND_TYPE + STATEMENT_ALLOWANCE > chars(budget()) * 0.97:
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
    from sema.intake.calls import model_calls

    assert budget() == model_calls.MAX_PROMPT_TOKENS
    assert CONSTANT_LINE.search(inspect.getsource(model_calls)), "a literal, not derived"


def test_wide_relation_keeps_name_and_type_per_column_and_fits_one_call(
    wide_shape: tuple[int, int], tmp_path: Path
) -> None:
    width = compacted_wide_width(wide_shape)
    built = wide(width, tmp_path / "source")
    watch, _ = roles_run(built, tmp_path)
    client = watch.wired.client

    assert client.invoke.call_count == 1
    (prompt,), (output_model,) = prompts(client), output_models(client)
    assert size(prompt, output_model) <= budget()
    assert output_model.__name__.startswith(ROLE_MAP_PREFIX)
    columns = tuple(column_name(i) for i in range(width))
    for role in OPEN_ROLES:
        assert f"`role_{role}`" in prompt, role
    for name in columns:
        assert facts(prompt, name) == [f"- {name}: type VARCHAR"], name
    assert lines_naming(prompt, OMITTED)
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


def test_compaction_keeps_every_column_named_with_its_type(
    mixed_shape: tuple[int, int], tmp_path: Path
) -> None:
    fixed, per_column = mixed_shape
    numbers = width_over(fixed, per_column, 1.2)
    if fixed + (LABELS + 1 + numbers) * NAME_AND_TYPE + STATEMENT_ALLOWANCE > chars(budget()):
        pytest.fail("no numeric width separates full facts from name and type at this budget")
    built = mixed(numbers, tmp_path / "source")
    watch, _ = roles_run(built, tmp_path)
    built.close()

    assert watch.wired.client.invoke.call_count == 1
    (prompt,), (output_model,) = prompts(watch.wired.client), output_models(watch.wired.client)
    assert size(prompt, output_model) <= budget()
    for index in range(LABELS + 1 + numbers):
        assert len(facts(prompt, column_name(index))) == 1, column_name(index)
    for index in range(LABELS + 1, LABELS + 1 + numbers):
        name = column_name(index)
        assert facts(prompt, name) == [f"- {name}: type DOUBLE"], name
    assert lines_naming(prompt, OMITTED), "the omission of the column facts is not stated"


def test_prompt_that_cannot_fit_is_refused_without_a_call(tmp_path: Path) -> None:
    """Five schema rows, each longer than the budget shares out, still do not fit once
    every step of the order has run."""
    sentence = int(chars(budget()) * 1.1 // 5) + 1
    statement = long_dictionary_table("dict", 5, sentence)
    built = Built([statement], tmp_path / "source", lambda: [dictionary("dict")])
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
    (left,) = [a for a in result.ambiguities if a.subject == f"{relation('dict')}.property"]
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


@pytest.mark.parametrize("name", sorted(UNDER_BUDGET_SHAPES))
def test_prompt_under_the_budget_renders_in_full(name: str, tmp_path: Path) -> None:
    """Compaction does not touch it: every column keeps its full facts, every schema row
    is shown, and nothing is stated as omitted."""
    if name == "wide_6":
        built = Built([text_table("wide", 6)], tmp_path, lambda: [code_list("wide")])
    else:
        built = dictionary_built(5, tmp_path, descending=True)
    watch, _ = roles_run(built, tmp_path)
    built.close()

    (prompt,) = prompts(watch.wired.client)
    columns, rows = UNDER_BUDGET_SHAPES[name]
    assert "Omitted to fit the budget" not in prompt
    assert [line for line in prompt.splitlines() if "more rows not shown" in line] == []
    for index in range(columns):
        (line,) = facts(prompt, column_name(index))
        assert "distinct" in line and "null rate" in line, line
    assert len(shown_keys(prompt)) == rows


def dotted_table(numeric_width: int) -> str:
    """A unique dotted code a ``path_codes`` slot holds, then empty numbers no slot holds;
    empty, so no template's checks run over thousands of columns."""
    numbers = [f"NULL::DOUBLE AS {column_name(i)}" for i in range(1, numeric_width + 1)]
    code = f"'g' || (r % 4)::VARCHAR || '.' || r::VARCHAR AS {column_name(0)}"
    cells = ", ".join([code, *numbers])
    return f"CREATE TABLE {relation('wide')} AS SELECT {cells} FROM range(20) t(r)"


def structure_ask(numeric_width: int, root: Path) -> MagicMock:
    """One open entry on the wide relation offering a relation-only and a column-slot binding."""
    built = Built([dotted_table(numeric_width)], root / "source", lambda: [])
    source = built.source
    tools = fresh_tools(source, root / "structure")
    profiles = profile_relations(source.land, tools, source.provider)
    found = member_bindings(built.structure, profiles, tools)
    offered = sorted(m for m, b in found[relation("wide")].items() if b.template in OFFERED)
    assert len(offered) == len(OFFERED), offered
    entry = Ambiguity(stage=STAGE, subject=relation("wide"), members=tuple(offered))
    structure = built.structure.model_copy(update={"ambiguities": (entry,)})
    tools = fresh_tools(source, root / "resolve", stage="resolve_structure", structure=structure)
    client = abstaining()
    watch = Watched(root, client, "run-a")
    try:
        resolve().resolve_structure(structure, source.land, tools, watch.wired.calls,
                                    provider=source.provider)
    finally:
        watch.stop()
        built.close()
    return client


def test_structure_prompt_keeps_facts_only_for_slotted_columns(tmp_path: Path) -> None:
    lengths = []
    for width in (4, 8):
        client = structure_ask(width, tmp_path / f"calibrate_{width}")
        (prompt,), (output_model,) = prompts(client), output_models(client)
        lengths.append(len(prompt) + len(schema_json(output_model)))
    per_column = (lengths[1] - lengths[0]) // 4
    numbers = width_over(lengths[0] - 4 * per_column, per_column, 1.05)
    client = structure_ask(numbers, tmp_path / "wide")

    assert client.invoke.call_count == 1
    (prompt,), (output_model,) = prompts(client), output_models(client)
    assert size(prompt, output_model) <= budget()
    assert len(facts(prompt, column_name(0))) == 1, "the slotted column lost its facts"
    for index in range(1, numbers + 1):
        name = column_name(index)
        assert lines_naming(prompt, name), f"{name} is not named"
        assert not facts(prompt, name), f"{name} kept its facts"
    assert [line for line in numbered(prompt, numbers) if "named only" in line]

"""Arm S applies the run's deny policy, the same one the shared SQL validator enforces."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from showcase.ontology_query_benchmark import runner_utils
from showcase.ontology_query_benchmark.profile import Profile
from tests.unit.ontology_query_benchmark.test_sema_bindings import EVENTS, PEOPLE, RID
from tests.unit.ontology_query_benchmark.test_sema_context import (
    EVENTS_ENTITY,
    PEOPLE_ENTITY,
    PEOPLE_VALUES,
    QUESTION,
    _col,
    _entity,
    _provider,
)

pytestmark = pytest.mark.unit

EVENTS_QUESTION = "How many events per person?"


def _policy(harness: SimpleNamespace, *denied: str, patterns: tuple[str, ...] = ()) -> Any:
    return harness.models.ColumnPolicy(denied=denied, deny_patterns=patterns)


def _join(column: str) -> dict[str, Any]:
    predicate = {"left_table": EVENTS, "left_column": column,
                 "right_table": PEOPLE, "right_column": column}  # fmt: skip
    return {"type": "join", "from_table": EVENTS, "to_table": PEOPLE, "confidence": 0.9,
            "join_predicates": [predicate], "cardinality_hint": "many_to_one"}  # fmt: skip


def test_a_name_another_product_denies_or_a_pattern_matches_is_never_shown(
    tmp_path: Path, harness: SimpleNamespace
) -> None:
    people = _entity(PEOPLE, [*PEOPLE_ENTITY["columns"], _col("contact_secret_note")])
    events = _entity(EVENTS, [*EVENTS_ENTITY["columns"], _col("first_name")])
    policy = _policy(harness, "first_name", patterns=("secret",))
    ctx = _provider(tmp_path, [people, events, *PEOPLE_VALUES])(QUESTION, policy.is_denied)
    assert "first_name" not in ctx.text
    assert "secret" not in ctx.text


def test_a_denied_grain_key_or_join_column_drops_the_line_with_a_finding(
    tmp_path: Path, harness: SimpleNamespace
) -> None:
    candidates = [PEOPLE_ENTITY, EVENTS_ENTITY, _join(RID)]
    policy = _policy(harness, RID.upper())
    ctx = _provider(tmp_path, candidates)(EVENTS_QUESTION, policy.is_denied)
    assert RID.lower() not in ctx.text.lower()
    assert "key row_id" in ctx.text
    withheld = [(f.relation, f.column) for f in ctx.findings if "withheld" in f.message]
    assert (f"cat.sch.{PEOPLE}", RID) in withheld
    assert sum(column == RID for _, column in withheld) >= 3


def test_a_join_given_as_sql_naming_a_denied_column_is_withheld(
    tmp_path: Path, harness: SimpleNamespace
) -> None:
    snippet = {**_join(RID), "join_predicates": [],
               "sql_snippet": f"{EVENTS}.secret_ref = {PEOPLE}.secret_ref"}  # fmt: skip
    policy = _policy(harness, patterns=("secret",))
    ctx = _provider(tmp_path, [PEOPLE_ENTITY, EVENTS_ENTITY, snippet])(QUESTION, policy.is_denied)
    assert "secret" not in ctx.text
    assert [f.column for f in ctx.findings if "join path" in f.message] == ["secret_ref"]


def test_a_grain_note_naming_a_denied_column_is_withheld_not_rewritten(
    tmp_path: Path, harness: SimpleNamespace
) -> None:
    policy = _policy(harness, "row_id")
    ctx = _provider(tmp_path, [EVENTS_ENTITY])(EVENTS_QUESTION, policy.is_denied)
    assert "row_id" not in ctx.text
    assert "surrogate" not in ctx.text
    assert any(f.column == "row_id" and "note" in f.message for f in ctx.findings)


def test_a_grain_note_that_does_not_name_a_denied_column_is_kept(
    tmp_path: Path, harness: SimpleNamespace
) -> None:
    note = "Event grain columns (supplied): {columns}."
    provider = _provider(tmp_path, [EVENTS_ENTITY], grain_note=note)
    ctx = provider(EVENTS_QUESTION, _policy(harness, "row_id").is_denied)
    assert "Event grain columns (supplied):" in ctx.text
    assert "row_id" not in ctx.text


def test_a_denied_key_drops_only_the_key_clause_and_keeps_the_grain_sentence(
    tmp_path: Path, harness: SimpleNamespace
) -> None:
    policy = _policy(harness, "row_id")
    ctx = _provider(tmp_path, [EVENTS_ENTITY])(EVENTS_QUESTION, policy.is_denied)
    assert f"  cat.sch.{EVENTS}: one row per event row\n" in ctx.text
    assert "row_id" not in ctx.text
    assert (f"cat.sch.{EVENTS}", "row_id") in [(f.relation, f.column) for f in ctx.findings]


def test_a_grain_sentence_naming_a_denied_column_is_withheld_and_its_key_kept(
    tmp_path: Path, harness: SimpleNamespace
) -> None:
    ctx = _provider(tmp_path, [PEOPLE_ENTITY])(QUESTION, _policy(harness, "rid").is_denied)
    assert "one row per rid" not in ctx.text
    assert f"  cat.sch.{PEOPLE}: key {RID}" in ctx.text
    assert any(f.column == "rid" and "grain sentence" in f.message for f in ctx.findings)


@pytest.mark.parametrize("field", ["{surrogate_row_id!s}", "{surrogate_row_id:>10}"])
def test_a_grain_note_field_with_a_conversion_or_format_is_still_checked(
    tmp_path: Path, harness: SimpleNamespace, field: str
) -> None:
    provider = _provider(tmp_path, [EVENTS_ENTITY], grain_note=f"Grain: {{columns}}. {field}.")
    ctx = provider(EVENTS_QUESTION, _policy(harness, "row_id").is_denied)
    assert "row_id" not in ctx.text
    assert any(f.column == "row_id" and "note" in f.message for f in ctx.findings)


def test_without_a_cross_product_denied_name_the_context_is_unchanged(
    tmp_path: Path, harness: SimpleNamespace
) -> None:
    candidates = [PEOPLE_ENTITY, EVENTS_ENTITY, _join(RID), *PEOPLE_VALUES]
    provider = _provider(tmp_path, candidates)
    policy = _policy(harness, "first_name", patterns=("no_such_fragment",))
    assert provider(QUESTION, policy.is_denied) == provider(QUESTION)


def test_arm_s_is_given_the_policy_its_validator_enforces(
    tmp_path: Path, frozen: Path, harness: SimpleNamespace, profile: Profile
) -> None:
    copy = shutil.copytree(frozen, tmp_path / "frozen")
    config = copy / profile.harness.config
    raw = json.loads(config.read_text())
    config.write_text(json.dumps(raw | {"policy": {"denied": ["Secret_Col"]}}))
    seen: list[Any] = []

    def provider(question: str, is_denied: Any = None) -> SimpleNamespace:
        seen.append(is_denied)
        return SimpleNamespace(text="T", identity="i", findings=())

    runtimes = runner_utils.arm_runtimes(harness, copy, profile.harness, {"S": None}, provider,
                                         profile.limits)  # fmt: skip
    runtimes["S"].context("q")
    assert seen == [runtimes["S"].policy.is_denied]
    assert seen[0]("SECRET_COL") and not seen[0]("rid")

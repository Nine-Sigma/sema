"""Profiling tools: policy-gated values and schema text."""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.unit.intake.profile_env import (
    A2_CONCEPT_TEXT,
    A2_LABELS,
    CONCEPTS,
    SAMPLE_CAP,
    VALUES,
    WIDE,
    WIDE_DISTINCT,
    BoundStructure,
    Env,
    make_policy,
    make_tools,
    open_env,
    reads,
    value_bearing,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Env]:
    yield from open_env(tmp_path, monkeypatch)


# sample_values


def test_metadata_only_returns_a_needs_values_marker(env: Env) -> None:
    from sema.intake.land.profile import NeedsValues

    marker = make_tools(env, stage="roles").sample_values(VALUES, "label", 5)

    assert isinstance(marker, NeedsValues)
    assert marker.relation == VALUES.name
    assert tuple(marker.columns) == ("label",)
    assert not value_bearing(env)


def test_bounded_allowlisted_column_returns_values(env: Env) -> None:
    policy = make_policy("bounded_values", allowlist=frozenset({(VALUES.name, "label")}))

    sample = make_tools(env, policy, stage="roles").sample_values(VALUES, "label", 5)

    assert len(sample.values) == 5 and set(sample.values) <= A2_LABELS
    flagged = value_bearing(env)
    assert flagged and all(not line["schema_relation"] for line in flagged)
    assert {line["stage"] for line in env.sql_lines()} == {"roles"}


def test_bounded_with_150_distinct_returns_at_most_20(env: Env) -> None:
    policy = make_policy("bounded_values", allowlist=frozenset({(WIDE.name, "v")}))

    sample = make_tools(env, policy).sample_values(WIDE, "v", 50)

    assert len(sample.values) == SAMPLE_CAP
    assert value_bearing(env)


def test_bounded_column_off_the_allowlist_needs_values(env: Env) -> None:
    from sema.intake.land.profile import NeedsValues

    policy = make_policy("bounded_values", allowlist=frozenset({(VALUES.name, "label")}))

    marker = make_tools(env, policy).sample_values(VALUES, "code", 5)

    assert isinstance(marker, NeedsValues) and tuple(marker.columns) == ("code",)
    assert not value_bearing(env)


def test_bounded_above_max_distinct_values_read_needs_values(env: Env) -> None:
    from sema.intake.land.profile import NeedsValues

    policy = make_policy(
        "bounded_values",
        allowlist=frozenset({(WIDE.name, "v")}),
        max_distinct_values_read=WIDE_DISTINCT - 50,
    )

    marker = make_tools(env, policy).sample_values(WIDE, "v", 5)

    assert isinstance(marker, NeedsValues)
    assert not value_bearing(env)


def test_full_policy_samples_any_column_capped_at_20(env: Env) -> None:
    sample = make_tools(env, make_policy("full")).sample_values(WIDE, "v", 100)

    assert len(sample.values) == SAMPLE_CAP
    assert all(isinstance(value, str) and value.startswith("v") for value in sample.values)
    assert value_bearing(env)


def test_sensitive_column_never_sampled(env: Env) -> None:
    from sema.intake.land.profile import ProfileRefusedError

    policy = make_policy("full", sensitive=frozenset({(VALUES.name, "label")}))

    with pytest.raises(ProfileRefusedError, match="label"):
        make_tools(env, policy).sample_values(VALUES, "label", 5)

    assert env.issued == [] and env.sql_lines() == []


# read_schema_text


def test_schema_text_of_a_bound_dictionary_is_logged_as_schema_relation(env: Env) -> None:
    structure = BoundStructure(frozenset({CONCEPTS.name}))
    tools = make_tools(env, structure=structure)

    text = tools.read_schema_text(CONCEPTS, ["concept_name", "definition"])

    assert {tuple(row) for row in text} == A2_CONCEPT_TEXT
    flagged = value_bearing(env)
    assert flagged and all(line["schema_relation"] for line in flagged)


def test_schema_text_refused_for_an_unbound_relation(env: Env) -> None:
    from sema.intake.land.profile import ProfileRefusedError

    tools = make_tools(env, structure=BoundStructure(frozenset({CONCEPTS.name})))

    with pytest.raises(ProfileRefusedError, match=re.escape(VALUES.name)):
        tools.read_schema_text(VALUES, ["label"])

    assert env.issued == [] and env.sql_lines() == []


def test_schema_text_refused_without_a_structure_output(env: Env) -> None:
    from sema.intake.land.profile import ProfileRefusedError

    with pytest.raises(ProfileRefusedError, match=re.escape(CONCEPTS.name)):
        make_tools(env).read_schema_text(CONCEPTS, ["definition"])

    assert env.issued == [] and env.sql_lines() == []


def test_metadata_only_value_statements_all_read_schema_relations(env: Env) -> None:
    tools = make_tools(env, structure=BoundStructure(frozenset({CONCEPTS.name})))

    tools.describe_relation(VALUES)
    tools.sample_values(VALUES, "label", 5)
    tools.read_schema_text(CONCEPTS, ["definition"])

    flagged = value_bearing(env)
    assert flagged and all(line["schema_relation"] for line in flagged)
    assert all(reads(line["statement"], '"land_a2"."codes_a2__Concepts"') for line in flagged)

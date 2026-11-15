"""Profiling tools: aggregate-only describes, structural checks, policy-gated values."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from sema.binding.spec_sections import RelationRef
from tests.unit.binding.fixtures.concept_library import query_fn
from tests.unit.intake.profile_env import (
    A3_DEPTH,
    A3_EDGES,
    CONCEPTS,
    CYCLE,
    HIERARCHY,
    ODD,
    ODD_COLUMN,
    PATHS,
    SYN_CONCEPT,
    VALUES,
    Env,
    FakeVocabulary,
    column_of,
    make_tools,
    open_env,
    reads,
    value_bearing,
)
from tests.unit.intake.warehouse_env import (
    CATALOG,
    SCHEMA,
    TABLE,
    FakeConnector,
    create_codes,
    schema_metadata,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Env]:
    yield from open_env(tmp_path, monkeypatch)


# list_relations


def test_list_relations_names_counts_and_comments(env: Env) -> None:
    listing = make_tools(env).list_relations("staging", "land_a2")

    by_name = {r.name: r for r in listing}
    assert {name: r.row_count for name, r in by_name.items()} == {
        CONCEPTS.name: 5,
        HIERARCHY.name: 3,
        VALUES.name: 9,
    }
    assert all(r.comment is None for r in listing)
    assert env.sql_lines() and not value_bearing(env)


# describe_relation


def test_describe_derives_rates_and_flags_from_aggregates(env: Env) -> None:
    described = make_tools(env).describe_relation(VALUES)

    assert described.row_count == 9
    concept_code = column_of(described, "concept_code")
    assert (concept_code.null_rate, concept_code.distinct_count) == (0.0, 3)
    assert concept_code.distinct_ratio == pytest.approx(1 / 3)
    assert not concept_code.is_unique and not concept_code.is_constant
    code = column_of(described, "code")
    assert (code.distinct_ratio, code.is_unique) == (1.0, True)
    assert column_of(described, "label").mean_length == pytest.approx(49 / 9)
    assert column_of(described, "rank").mean_length == pytest.approx(1.0)
    assert column_of(described, "successor").null_rate == pytest.approx(8 / 9)
    assert (code.type, column_of(described, "rank").type) == ("VARCHAR", "BIGINT")


def test_describe_all_null_and_constant_columns(env: Env) -> None:
    described = make_tools(env).describe_relation(CONCEPTS)

    empty = column_of(described, "_empty_3")
    assert (empty.null_rate, empty.distinct_count, empty.distinct_ratio) == (1.0, 0, 0.0)
    assert empty.mean_length is None
    assert not empty.is_unique and not empty.is_constant
    data_type = column_of(described, "data_type")
    assert data_type.is_constant and not data_type.is_unique
    assert column_of(described, "concept_code").is_unique


def test_describe_issues_one_aggregate_statement_per_relation(env: Env) -> None:
    tools = make_tools(env)
    provider = env.provider()

    for ref in (CONCEPTS, VALUES, HIERARCHY):
        tools.describe_relation(ref)

    for ref in (CONCEPTS, VALUES, HIERARCHY):
        relation_sql = provider.relation_sql(ref)
        assert sum(reads(s, relation_sql) for s in env.statements()) == 1, ref.name
    lines = env.sql_lines()
    assert lines and all(not line["value_bearing"] for line in lines)
    assert all(not line["schema_relation"] for line in lines)
    assert {(line["stage"], line["provider"]) for line in lines} == {("structure", "staging")}


def test_describe_logs_every_issued_statement(env: Env) -> None:
    from sema.intake.calls.ledger import Ledger
    from sema.intake.land.loaders.warehouse import land_warehouse

    create_codes(env.warehouse)
    connector = FakeConnector(schema_metadata(), query=query_fn(env.warehouse))
    land_ledger = Ledger(env.ledger_dir.parent).run("land")
    land = land_warehouse(connector, catalog=CATALOG, schema=SCHEMA, ledger=land_ledger)
    ref = RelationRef(provider="warehouse", name=f"{CATALOG}.{SCHEMA}.{TABLE}")

    described = make_tools(env, land=land).describe_relation(ref)

    assert described.row_count == 7
    concept_id = column_of(described, "concept_id")
    assert (concept_id.distinct_ratio, concept_id.is_unique) == (1.0, True)
    assert column_of(described, "vocabulary_id").is_constant
    assert env.issued and env.statements() == env.issued
    assert {line["provider"] for line in env.sql_lines()} == {"warehouse"}
    assert not value_bearing(env)


def test_describe_quotes_identifiers_through_the_provider(env: Env) -> None:
    described = make_tools(env).describe_relation(ODD)

    column = column_of(described, ODD_COLUMN)
    assert described.row_count == 3
    assert column.null_rate == pytest.approx(1 / 3)
    assert column.mean_length == pytest.approx(1.5)
    assert column.comment == "a note"


# check_join


def test_join_resolves_one_to_many_in_both_directions(env: Env) -> None:
    joined = make_tools(env).check_join(CONCEPTS, "concept_code", VALUES, "concept_code")

    assert joined.resolve_a_to_b == pytest.approx(3 / 5)
    assert joined.resolve_b_to_a == pytest.approx(1.0)
    assert joined.cardinality == "one_to_many"
    assert len(env.sql_lines()) == 1 and not value_bearing(env)


def test_join_reversed_reports_many_to_one(env: Env) -> None:
    joined = make_tools(env).check_join(VALUES, "concept_code", CONCEPTS, "concept_code")

    assert (joined.resolve_a_to_b, joined.cardinality) == (pytest.approx(1.0), "many_to_one")
    assert joined.resolve_b_to_a == pytest.approx(3 / 5)


def test_join_across_providers_refused_before_a_statement(env: Env) -> None:
    from sema.intake.land.profile import ProfileRefusedError

    with pytest.raises(ProfileRefusedError) as refused:
        make_tools(env).check_join(VALUES, "concept_code", SYN_CONCEPT, "concept_id")

    assert "staging" in str(refused.value) and "warehouse" in str(refused.value)
    assert env.issued == [] and env.sql_lines() == []


# check_hierarchy


def test_hierarchy_on_a3_edges(env: Env) -> None:
    checked = make_tools(env).check_hierarchy(A3_EDGES, "subject", "object")

    assert checked.cycle_count == 0
    assert checked.root_count == 1
    assert checked.max_depth == A3_DEPTH
    assert env.sql_lines() and not value_bearing(env)


def test_hierarchy_reports_a_two_node_cycle(env: Env) -> None:
    checked = make_tools(env).check_hierarchy(CYCLE, "child", "parent")

    assert checked.cycle_count > 0
    assert checked.root_count == 0
    assert not value_bearing(env)


# count_distinct, check_levels, check_paths


@pytest.mark.parametrize(
    ("columns", "expected"),
    [
        (["concept_code"], 3),
        (["concept_code", "code"], 9),
        (["concept_code", "status"], 4),
        (["concept_code", "successor"], 1),
    ],
)
def test_count_distinct_counts_tuples_over_non_null_rows(
    env: Env, columns: list[str], expected: int
) -> None:
    assert make_tools(env).count_distinct(VALUES, columns) == expected
    assert len(env.sql_lines()) == 1 and not value_bearing(env)


@pytest.mark.parametrize(
    ("columns", "misplaced", "multi_parent"),
    [(["l1", "l2", "l3"], 0, 0), (["l2", "l1"], 0, 1), (["l3", "l2"], 1, 1)],
)
def test_check_levels_counts_misplaced_nulls_and_multi_parent_nodes(
    env: Env, columns: list[str], misplaced: int, multi_parent: int
) -> None:
    checked = make_tools(env).check_levels(HIERARCHY, columns)

    assert (checked.misplaced_nulls, checked.multi_parent) == (misplaced, multi_parent)
    assert len(env.sql_lines()) == 1 and not value_bearing(env)


def test_check_levels_refuses_one_column_before_a_statement(env: Env) -> None:
    from sema.intake.land.profile import ProfileRefusedError

    with pytest.raises(ProfileRefusedError):
        make_tools(env).check_levels(HIERARCHY, ["l1"])

    assert env.issued == [] and env.sql_lines() == []


def test_check_paths_counts_codes_and_implicit_prefixes(env: Env) -> None:
    tools = make_tools(env)

    dotted = tools.check_paths(PATHS, "code", ".")
    slashed = tools.check_paths(PATHS, "code", "/")
    odd = tools.check_paths(PATHS, "odd", ".")

    assert (dotted.code_count, dotted.separated, dotted.malformed) == (4, 1, 0)
    assert dotted.implicit_count == 1
    assert (slashed.separated, slashed.malformed, slashed.implicit_count) == (0, 0, 0)
    assert (odd.code_count, odd.separated, odd.malformed) == (5, 4, 3)
    assert len(env.sql_lines()) == 3 and not value_bearing(env)


@pytest.mark.parametrize("separator", ["", "..", " / "])
def test_check_paths_refuses_a_separator_that_is_not_one_character(
    env: Env, separator: str
) -> None:
    from sema.intake.land.profile import ProfileRefusedError

    with pytest.raises(ProfileRefusedError, match="one character"):
        make_tools(env).check_paths(PATHS, "code", separator)

    assert env.issued == [] and env.sql_lines() == []


# resolve_external_refs


def test_external_refs_resolve_against_a_loaded_system(env: Env) -> None:
    vocabulary = FakeVocabulary({("SYNVOCAB", "C1"), ("SYNVOCAB", "C2")})
    tools = make_tools(env, vocabulary=vocabulary, loaded_systems=frozenset({"SYNVOCAB"}))

    resolved = tools.resolve_external_refs("SYNVOCAB", ["C1", "C2", "C9"])

    assert resolved.unverifiable is False
    assert resolved.resolve_rate == pytest.approx(2 / 3)


def test_external_refs_unverifiable_when_the_system_is_not_loaded(env: Env) -> None:
    vocabulary = FakeVocabulary({("OTHER", "C1")})
    tools = make_tools(env, vocabulary=vocabulary, loaded_systems=frozenset({"SYNVOCAB"}))

    resolved = tools.resolve_external_refs("OTHER", ["C1"])

    assert resolved.unverifiable is True
    assert resolved.resolve_rate is None
    assert vocabulary.calls == []


def test_external_refs_unverifiable_without_a_vocabulary(env: Env) -> None:
    resolved = make_tools(env).resolve_external_refs("SYNVOCAB", ["C1"])

    assert resolved.unverifiable is True and resolved.resolve_rate is None

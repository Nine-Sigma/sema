"""concept_and_value_set: a concept relation, a value relation, and a resolving key pair."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from tests.unit.intake.structure.shapes.shapes_env import (
    Memory,
    Source,
    binding,
    candidates,
    land_of_schema,
    shapes,
    staging_source,
    verify,
)

pytestmark = pytest.mark.unit

ID = "concept_and_value_set"
CONCEPTS, VALUES, HIERARCHY = "codes_a2__Concepts", "codes_a2__Values", "codes_a2__Hierarchy"
RENAMED = {
    CONCEPTS: (
        "t_a",
        {"concept_code": "k1", "concept_name": "k2", "_empty_3": "k3", "definition": "k4",
         "data_type": "k5"},
    ),
    VALUES: (
        "t_b",
        {"concept_code": "k6", "code": "k7", "label": "k8", "rank": "k9", "status": "k10",
         "successor": "k11"},
    ),
    HIERARCHY: ("t_c", {"l1": "k12", "l2": "k13", "l3": "k14"}),
}


def _keys(bindings: list[object]) -> set[tuple[str, str, str, str]]:
    return {
        (
            b.slots["concept_relation"],  # type: ignore[attr-defined]
            b.slots["value_relation"],  # type: ignore[attr-defined]
            b.slots["concept_key"],  # type: ignore[attr-defined]
            b.slots["value_key"],  # type: ignore[attr-defined]
        )
        for b in bindings
    }


def _expected(a2: Source) -> tuple[str, str, str, str]:
    return (a2.ref(CONCEPTS), a2.ref(VALUES), "concept_code", "concept_code")


def test_a2_concept_code_is_a_join_key_on_both_sides(a2: Source) -> None:
    found = candidates(ID, a2.profiles(), a2)

    assert _expected(a2) in _keys(found)
    pair = next(b for b in found if b.slots["value_relation"] == a2.ref(VALUES))
    assert set(pair.relations) == {a2.ref(CONCEPTS), a2.ref(VALUES)}


def test_a2_has_no_other_key_pair(a2: Source) -> None:
    assert _keys(candidates(ID, a2.profiles(), a2)) == {_expected(a2)}


def test_value_relation_needs_a_non_null_code_unique_within_each_key_group(
    memory: Memory,
) -> None:
    source = memory(
        "cvs",
        "CREATE TABLE cvs.concepts AS SELECT * FROM (VALUES ('a', 'Alpha'), ('b', 'Beta'),"
        " ('c', 'Gamma')) t(k, name)",
        "CREATE TABLE cvs.repeating AS SELECT * FROM (VALUES ('a', 'x'), ('a', 'x'),"
        " ('b', 'y'), ('b', 'y'), ('c', 'z'), ('c', 'z')) t(k, tag)",
        "CREATE TABLE cvs.nullable AS SELECT * FROM (VALUES ('a', '1'), ('a', '2'),"
        " ('b', '1'), ('b', NULL), ('c', '1'), ('c', '2')) t(k, code)",
        "CREATE TABLE cvs.valued AS SELECT * FROM (VALUES ('a', '1'), ('a', '2'),"
        " ('b', '1'), ('b', '2'), ('c', '1'), ('c', '2')) t(k, code)",
    )

    found = candidates(ID, source.profiles(), source)

    assert _keys(found) == {(source.ref("concepts"), source.ref("valued"), "k", "k")}
    assert source.value_bearing() == []


def test_a2_values_and_concepts_pair_only_through_concept_code(a2: Source) -> None:
    found = candidates(ID, a2.profiles(), a2)
    between = {a2.ref(CONCEPTS), a2.ref(VALUES)}

    assert {k for k in _keys(found) if set(k[:2]) == between} == {_expected(a2)}


@pytest.fixture
def renamed(a2: Source) -> Iterator[Source]:
    staging = a2.tools.provider.staging
    staging.execute('CREATE SCHEMA IF NOT EXISTS "land_r"')
    for sheet, (table, columns) in RENAMED.items():
        selected = ", ".join(f'"{old}" AS "{new}"' for old, new in columns.items())
        staging.execute(
            f'CREATE OR REPLACE TABLE "land_r"."{table}" AS SELECT {selected},'
            f' "_sema_locator" FROM "land_a2"."{sheet}"'
        )
    land = land_of_schema(lambda sql: staging.execute(sql).fetchall(), "land_r")
    yield staging_source(staging, land, a2.ledger_dir.parent / "renamed")


def test_bindings_do_not_depend_on_relation_or_column_names(a2: Source, renamed: Source) -> None:
    def rename(key: tuple[str, str, str, str]) -> tuple[str, str, str, str]:
        concept_sheet = next(s for s in RENAMED if a2.ref(s) == key[0])
        value_sheet = next(s for s in RENAMED if a2.ref(s) == key[1])
        (concept_table, concept_cols), (value_table, value_cols) = (
            RENAMED[concept_sheet],
            RENAMED[value_sheet],
        )
        return (
            f"land_r.{concept_table}",
            f"land_r.{value_table}",
            concept_cols[key[2]],
            value_cols[key[3]],
        )

    original = _keys(candidates(ID, a2.profiles(), a2))
    translated = _keys(candidates(ID, renamed.profiles(), renamed))

    assert translated == {rename(key) for key in original}


def test_verify_passes_and_records_the_join_floor(a2: Source) -> None:
    group = a2.profiles()
    found = next(
        b for b in candidates(ID, group, a2) if b.slots["value_relation"] == a2.ref(VALUES)
    )

    verdict = verify(ID, found, group, a2)

    assert verdict.status == "pass"
    assert verdict.floors["JOIN_RESOLVE_FLOOR"] == shapes("base").JOIN_RESOLVE_FLOOR


def test_verify_fails_join_resolves_on_a_key_that_does_not_resolve(a2: Source) -> None:
    concepts, values = a2.ref(CONCEPTS), a2.ref(VALUES)
    forced = binding(
        ID,
        (concepts, values),
        concept_relation=concepts,
        value_relation=values,
        concept_key="concept_code",
        value_key="label",
    )

    verdict = verify(ID, forced, a2.profiles(), a2)

    assert verdict.status == "fail"
    assert "join_resolves" in {f.check for f in verdict.failures}

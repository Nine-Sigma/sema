"""concept_and_value_set: values are not concepts (Dean, 2026-10-21).

Pinned rule (shape-templates, MODIFIED "Concept and value set", scenario "Values that are
concepts"): a binding fails when a value code of the value relation (non-null, unique within
each key group, an id of the value relation) resolves into the concept key at or above
``JOIN_RESOLVE_FLOOR`` through the join check. Every value is then a concept, so the value
relation links concepts to concepts. ``candidates`` and ``verify`` agree, and ``verify`` fails
with the check ``values_are_not_concepts`` naming the value code column.

Guards: every correct value set measured in ``s3_cvs_meaning_check.md`` stays eligible.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from sema.ingest.duckdb_staging import Staging
from sema.intake.loaders.database import land_database
from tests.unit.intake.shapes.shapes_env import (
    FIXTURES,
    Memory,
    Source,
    binding,
    candidates,
    staging_source,
    verify,
)

pytestmark = pytest.mark.unit

ID = "concept_and_value_set"
VALUES_ARE_NOT_CONCEPTS = "values_are_not_concepts"
Key = tuple[str, str, str, str]

# Three groups of two values; the concept relation lists the groups and the values alike.
GROUP_ROWS = "('G1', 'Group one'), ('G2', 'Group two'), ('G3', 'Group three')"
VALUE_ROWS = (
    "('V1', 'Value one'), ('V2', 'Value two'), ('V3', 'Value three'),"
    " ('V4', 'Value four'), ('V5', 'Value five'), ('V6', 'Value six')"
)
MEMBERS_SQL = (
    "CREATE TABLE vc.members AS SELECT * FROM (VALUES ('G1', 'V1'), ('G1', 'V2'),"
    " ('G2', 'V3'), ('G2', 'V4'), ('G3', 'V5'), ('G3', 'V6')) t(group_id, value_id)"
)
VALUE_KEY, VALUE_CODE = "group_id", "value_id"

LIBRARY = FIXTURES / "partner_sl" / "library.duckdb"
LIBRARY_CONCEPTS = "standard_library_concept"
LIBRARY_VALUES = "standard_library_data_element"
LIBRARY_KEY, LIBRARY_VALUE_KEY = "concept_prt_id", "data_element_prt_id"
LIBRARY_VALUE_CODE = "permissible_value_prt_id"

CVS_SQL = (
    "CREATE TABLE cvs.concepts AS SELECT * FROM (VALUES ('a', 'Alpha'), ('b', 'Beta'),"
    " ('c', 'Gamma')) t(k, name)",
    "CREATE TABLE cvs.valued AS SELECT * FROM (VALUES ('a', '1'), ('a', '2'),"
    " ('b', '1'), ('b', '2'), ('c', '1'), ('c', '2')) t(k, code)",
)
FIX_SQL = (
    "CREATE TABLE fix.topic AS SELECT * FROM (VALUES ('T1', 'First topic'),"
    " ('T2', 'Second topic'), ('T3', 'Third topic')) t(code, label)",
    "CREATE TABLE fix.member AS SELECT * FROM (VALUES ('T1', 'M1', 'One'),"
    " ('T1', 'M2', 'Two'), ('T2', 'M1', 'One'), ('T2', 'M3', 'Three'),"
    " ('T3', 'M2', 'Two'), ('T3', 'M4', 'Four')) t(topic, item, item_label)",
)
TC_SQL = (
    "CREATE TABLE tc.lib__first AS SELECT * FROM (VALUES ('K1', 'Alder grove'),"
    " ('K2', 'Birch hollow'), ('K3', 'Cedar ridge'), ('K4', 'Dune flat')) t(key, label)",
    "CREATE TABLE tc.lib__second AS SELECT * FROM (VALUES ('K1', 'Elm spire'),"
    " ('K2', 'Fir basin'), ('K3', 'Gorse moor'), ('K4', 'Heath knoll')) t(key, label)",
    "CREATE TABLE tc.lib__values AS SELECT * FROM (VALUES ('K1', 'V1', 'Ivy one'),"
    " ('K1', 'V2', 'Juniper two'), ('K2', 'V3', 'Kelp three'), ('K3', 'V4', 'Larch four'),"
    " ('K3', 'V5', 'Maple five')) t(key, value, label)",
)


@pytest.fixture(scope="module")
def library(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Source]:
    root = tmp_path_factory.mktemp("cvs_library")
    staging = Staging(str(root / "staging.duckdb"), schemas=())
    landed = land_database(LIBRARY, staging, schema="land_library")
    yield staging_source(staging, landed, root / "ledger" / "run")
    staging.close()


def _keys(bindings: list[object]) -> set[Key]:
    return {
        (
            b.slots["concept_relation"],  # type: ignore[attr-defined]
            b.slots["value_relation"],  # type: ignore[attr-defined]
            b.slots["concept_key"],  # type: ignore[attr-defined]
            b.slots["value_key"],  # type: ignore[attr-defined]
        )
        for b in bindings
    }


def _forced(concepts: str, values: str, key: str, value_key: str) -> object:
    return binding(
        ID,
        (concepts, values),
        concept_relation=concepts,
        value_relation=values,
        concept_key=key,
        value_key=value_key,
    )


def _crosswalk(memory: Memory) -> Source:
    concepts = (
        "CREATE TABLE vc.concepts AS SELECT * FROM (VALUES "
        f"{GROUP_ROWS}, {VALUE_ROWS}) t(id, label)"
    )
    return memory("vc", concepts, MEMBERS_SQL)


def _owners_only(memory: Memory) -> Source:
    concepts = f"CREATE TABLE vc.concepts AS SELECT * FROM (VALUES {GROUP_ROWS}) t(id, label)"
    return memory("vc", concepts, MEMBERS_SQL)


def _checks_on(verdict: object, check: str) -> list[object]:
    return [f for f in verdict.failures if f.check == check]  # type: ignore[attr-defined]


def test_value_codes_that_resolve_into_the_concept_key_are_not_eligible(
    memory: Memory,
) -> None:
    source = _crosswalk(memory)

    found = candidates(ID, source.profiles(), source)

    pair = (source.ref("concepts"), source.ref("members"), "id", VALUE_KEY)
    assert pair not in _keys(found)
    assert found == []


def test_verify_fails_naming_the_value_code_that_is_a_concept(memory: Memory) -> None:
    source = _crosswalk(memory)
    concepts, values = source.ref("concepts"), source.ref("members")

    verdict = verify(ID, _forced(concepts, values, "id", VALUE_KEY), source.profiles(), source)

    assert verdict.status == "fail"
    (found,) = _checks_on(verdict, VALUES_ARE_NOT_CONCEPTS)
    assert found.relation == values  # type: ignore[attr-defined]
    assert found.column == VALUE_CODE  # type: ignore[attr-defined]


def test_control_value_codes_of_their_own_stay_eligible(memory: Memory) -> None:
    source = _owners_only(memory)
    concepts, values = source.ref("concepts"), source.ref("members")

    found = candidates(ID, source.profiles(), source)
    verdict = verify(ID, _forced(concepts, values, "id", VALUE_KEY), source.profiles(), source)

    assert (concepts, values, "id", VALUE_KEY) in _keys(found)
    assert verdict.status == "pass", verdict.failures


def test_library_crosswalk_is_not_a_value_set_for_its_data_elements(library: Source) -> None:
    found = candidates(ID, library.profiles(), library)

    values = library.ref(LIBRARY_VALUES)
    assert [k for k in _keys(found) if k[1] == values] == []


def test_library_crosswalk_verify_names_the_permissible_value_id(library: Source) -> None:
    concepts, values = library.ref(LIBRARY_CONCEPTS), library.ref(LIBRARY_VALUES)
    forced = _forced(concepts, values, LIBRARY_KEY, LIBRARY_VALUE_KEY)

    verdict = verify(ID, forced, library.profiles(), library)

    assert verdict.status == "fail"
    (found,) = _checks_on(verdict, VALUES_ARE_NOT_CONCEPTS)
    assert found.relation == values  # type: ignore[attr-defined]
    assert found.column == LIBRARY_VALUE_CODE  # type: ignore[attr-defined]


def test_v2_level_table_is_not_a_value_set(v2: Source) -> None:
    found = candidates(ID, v2.profiles(), v2)

    assert [k for k in _keys(found) if k[1] == v2.ref("levels")] == []


def test_v2_concepts_and_their_values_stay_eligible(v2: Source) -> None:
    found = candidates(ID, v2.profiles(), v2)

    expected = (v2.ref("concepts"), v2.ref("vals"), "concept_id", "concept_id")
    assert expected in _keys(found)


def test_a2_concepts_and_values_stay_eligible(a2: Source) -> None:
    found = candidates(ID, a2.profiles(), a2)

    concepts, values = a2.ref("codes_a2__Concepts"), a2.ref("codes_a2__Values")
    assert _keys(found) == {(concepts, values, "concept_code", "concept_code")}


@pytest.mark.parametrize(
    ("schema", "statements", "expected"),
    [
        ("cvs", CVS_SQL, {("concepts", "valued", "k", "k")}),
        ("fix", FIX_SQL, {("topic", "member", "code", "topic")}),
        (
            "tc",
            TC_SQL,
            {
                ("lib__first", "lib__values", "key", "key"),
                ("lib__second", "lib__values", "key", "key"),
            },
        ),
    ],
)
def test_synthetic_value_sets_stay_eligible(
    memory: Memory, schema: str, statements: tuple[str, ...], expected: set[Key]
) -> None:
    source = memory(schema, *statements)

    found = candidates(ID, source.profiles(), source)

    qualified = {(source.ref(c), source.ref(v), key, value) for c, v, key, value in expected}
    assert qualified <= _keys(found)

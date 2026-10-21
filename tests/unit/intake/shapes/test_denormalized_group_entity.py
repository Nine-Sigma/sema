"""concept_with_values_denormalized: the group must be an entity (Dean, 2026-10-21).

Pinned rule (shape-templates, MODIFIED "Concept with denormalized values"): a ``group_by``
binding holds only if the group is an entity, shown by either

1. own attribute: a non-constant column other than ``group_by`` is functionally determined
   by it, ``count_distinct([G, C]) == count_distinct([G])`` over rows where C is not null;
2. weak value key: a value code column unique within each group is not unique across the
   relation.

Both tests use aggregates only. ``candidates`` and ``verify`` agree, and a rejected binding
fails ``verify`` with the check ``group_is_entity``.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Callable, Iterator

import pytest

from sema.ingest.duckdb_staging import Staging
from sema.intake.loaders.database import land_database
from tests.unit.intake.layout_proposal_env import CODEBOOK_COLUMNS, CODEBOOK_CSV, VARIABLE
from tests.unit.intake.shapes.shapes_env import (
    FIXTURES,
    Source,
    binding,
    candidates,
    every_template_candidates,
    over,
    shapes,
    staging_source,
    verify,
)

pytestmark = pytest.mark.unit

ID = "concept_with_values_denormalized"
GROUP_IS_ENTITY = "group_is_entity"
DOCUMENT_CONTROL = "Document_Control_Codes"
DOCUMENT_CONTROL_GROUPS = ("CLASS CODE DESCRIPTION", "PARTY1 TYPE")
SYN_GROUP_BY = "data_element_id"
OMOP_ENTITY = "cdmTableName"
LIBRARY = FIXTURES / "partner_sl" / "library.duckdb"
LIBRARY_RELATION = "standard_library_data_element"
LIBRARY_GROUP_BY = "data_element_prt_id"
# Three distinct lists over 36 rows: low cardinality, but it determines no other column and
# the value codes are globally unique, so the group describes nothing.
LIBRARY_SPURIOUS_GROUP_BY = "permissible_value_synonym_name"

# Weak key only: no column but the code depends on the variable, and codes repeat across
# variables while unique within each.
MINIMAL_CODEBOOK_SQL = (
    "CREATE TABLE cb.minimal AS SELECT * FROM (VALUES"
    " ('colour', 1, 'red'), ('colour', 2, 'green'), ('colour', 3, 'blue'),"
    " ('size', 1, 'small'), ('size', 2, 'medium'), ('size', 3, 'large'),"
    " ('shape', 1, 'round'), ('shape', 2, 'square'), ('shape', 3, 'oval'))"
    " t(variable, code, label)"
)
# Neither prong: `grade` determines only the constant `kind`, and `code` is globally unique.
FLAT_LIST_SQL = (
    "CREATE TABLE fl.flat AS SELECT * FROM (VALUES"
    " ('A1', 'Alder', 'low', 'tree'), ('A2', 'Birch', 'low', 'tree'),"
    " ('A3', 'Cedar', 'mid', 'tree'), ('A4', 'Dogwood', 'mid', 'tree'),"
    " ('A5', 'Elm', 'high', 'tree'), ('A6', 'Fir', 'high', 'tree'),"
    " ('A7', 'Gum', 'low', 'tree'), ('A8', 'Hazel', 'mid', 'tree'))"
    " t(code, label, grade, kind)"
)

Memory = Callable[..., Source]


@pytest.fixture(scope="module")
def library(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Source]:
    root = tmp_path_factory.mktemp("library")
    staging = Staging(str(root / "staging.duckdb"), schemas=())
    landed = land_database(LIBRARY, staging, schema="land_library")
    yield staging_source(staging, landed, root / "ledger" / "run")
    staging.close()


def _group_by(bindings: list[object]) -> set[str]:
    return {b.slots["group_by"] for b in bindings}  # type: ignore[attr-defined]


def _checks(verdict: object) -> set[str]:
    return {f.check for f in verdict.failures}  # type: ignore[attr-defined]


def _filled_codebook_sql() -> tuple[str, ...]:
    """The codebook as ``fill_down`` reads it: an empty variable takes the row above's."""
    rows = list(csv.reader(io.StringIO(CODEBOOK_CSV)))[1:]
    filled: list[list[str]] = []
    for row in rows:
        previous = filled[-1] if filled else row
        filled.append([row[0] or previous[0], row[1] or previous[1], *row[2:]])
    values = ", ".join(f"('{v}', '{d}', {c}, '{lb}')" for v, d, c, lb in filled)
    columns = ", ".join(CODEBOOK_COLUMNS)
    return (f"CREATE TABLE fd.codebook AS SELECT * FROM (VALUES {values}) t({columns})",)


def _over_one(source: Source, schema: str, table: str) -> list[object]:
    group = list(source.profiles())
    return over(candidates(ID, group, source), f"{schema}.{table}")


def test_document_control_offers_no_group_by_binding(acris: Source) -> None:
    assert candidates(ID, acris.group(DOCUMENT_CONTROL), acris) == []


@pytest.mark.parametrize("group_by", DOCUMENT_CONTROL_GROUPS)
def test_document_control_group_binding_fails_verify_as_no_entity(
    acris: Source, group_by: str
) -> None:
    ref = acris.ref(DOCUMENT_CONTROL)
    stated = binding(ID, (ref,), relation=ref, group_by=group_by)

    verdict = verify(ID, stated, acris.group(DOCUMENT_CONTROL), acris)

    assert verdict.status == "fail"
    assert _checks(verdict) == {GROUP_IS_ENTITY}
    (found,) = verdict.failures
    assert (found.relation, found.column) == (ref, group_by)


def test_document_control_members_are_concept_list_and_entity_model(acris: Source) -> None:
    group = acris.group(DOCUMENT_CONTROL)

    result = shapes().apply_precedence(every_template_candidates(group, acris))

    (ambiguity,) = [a for a in result.ambiguities if a.subject == acris.ref(DOCUMENT_CONTROL)]
    templates = sorted(m.partition("(")[0] for m in ambiguity.members)
    assert templates == ["concept_list", "entity_model"]


def test_synthetic_data_element_id_stays_eligible_and_verifies(syn: Source) -> None:
    element = syn.ref("data_element")
    found = over(candidates(ID, syn.profiles(), syn), element)
    (kept,) = [b for b in found if b.slots["group_by"] == SYN_GROUP_BY]

    assert verify(ID, kept, syn.profiles(), syn).status == "pass"


def test_library_data_element_id_stays_eligible(library: Source) -> None:
    found = over(candidates(ID, library.profiles(), library), library.ref(LIBRARY_RELATION))

    assert LIBRARY_GROUP_BY in _group_by(found)
    (kept,) = [b for b in found if b.slots["group_by"] == LIBRARY_GROUP_BY]
    assert verify(ID, kept, library.profiles(), library).status == "pass"


def test_library_spurious_group_by_is_gone(library: Source) -> None:
    ref = library.ref(LIBRARY_RELATION)
    found = over(candidates(ID, library.profiles(), library), ref)
    stated = binding(ID, (ref,), relation=ref, group_by=LIBRARY_SPURIOUS_GROUP_BY)

    assert LIBRARY_SPURIOUS_GROUP_BY not in _group_by(found)
    verdict = verify(ID, stated, library.profiles(), library)
    assert _checks(verdict) == {GROUP_IS_ENTITY}


def test_filled_down_codebook_keeps_variable(memory: Memory) -> None:
    source = memory("fd", *_filled_codebook_sql())

    found = _over_one(source, "fd", "codebook")

    assert VARIABLE in _group_by(found)


def test_minimal_codebook_keeps_variable_eligible_by_its_weak_key(memory: Memory) -> None:
    """The grid is symmetric, so ``code`` is eligible too; a model or reviewer settles it."""
    source = memory("cb", MINIMAL_CODEBOOK_SQL)

    found = _over_one(source, "cb", "minimal")

    assert VARIABLE in _group_by(found)
    (kept,) = [b for b in found if b.slots["group_by"] == VARIABLE]
    assert verify(ID, kept, list(source.profiles()), source).status == "pass"


def test_omop_entity_column_stays_eligible(omop: Source) -> None:
    assert OMOP_ENTITY in _group_by(candidates(ID, omop.profiles(), omop))


def test_flat_list_with_a_descriptive_grade_is_not_denormalized(memory: Memory) -> None:
    source = memory("fl", FLAT_LIST_SQL)
    ref = "fl.flat"
    stated = binding(ID, (ref,), relation=ref, group_by="grade")

    assert _over_one(source, "fl", "flat") == []
    assert _checks(verify(ID, stated, list(source.profiles()), source)) == {GROUP_IS_ENTITY}


def test_group_entity_check_reads_aggregates_only(memory: Memory) -> None:
    source = memory("fl", FLAT_LIST_SQL)

    candidates(ID, list(source.profiles()), source)

    assert source.sql_lines()
    assert source.value_bearing() == []


@pytest.mark.parametrize("fixture", ["acris", "syn", "omop", "library"])
def test_candidates_and_verify_agree(fixture: str, request: pytest.FixtureRequest) -> None:
    source: Source = request.getfixturevalue(fixture)
    for group in source.stem_groups():
        for found in candidates(ID, group, source):
            assert verify(ID, found, group, source).status == "pass", found.slots

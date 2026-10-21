"""Candidate precedence: conflicting claims open a relation (Dean, 2026-10-21).

Pinned rule (shape-templates, MODIFIED "Candidate precedence", amended scenario "Two
multi-relation claims"): a relation claimed by exactly one multi-relation binding drops its
single-relation candidates; a relation claimed by two or more keeps them, so its ambiguity
members are the claims plus its verified single-relation candidates.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from sema.ingest.duckdb_staging import Staging
from sema.intake.loaders.database import land_database
from tests.unit.intake.shapes.shapes_env import (
    FIXTURES,
    Source,
    binding,
    every_template_candidates,
    shapes,
    staging_source,
)
from tests.unit.intake.structure_env import fresh_tools, run

pytestmark = pytest.mark.unit

CONCEPTS, EDGES, OTHER = "s.a", "s.b", "s.c"
CONCEPTS_A2 = "codes_a2__Concepts"
LIBRARY = FIXTURES / "partner_sl" / "library.duckdb"
LIBRARY_VALUES = "standard_library_data_element"
LIBRARY_GROUP_BY = "data_element_prt_id"
# A group key the group-is-entity rule rejects (it describes nothing).
LIBRARY_SPURIOUS_GROUP_BY = "permissible_value_synonym_name"


def _edge_list(child: str, parent: str, edges: str = EDGES) -> object:
    return binding(
        "edge_list_hierarchy",
        (CONCEPTS, edges),
        concept_relation=CONCEPTS,
        id="i",
        edge_relation=edges,
        child=child,
        parent=parent,
    )


def _listed(relation: str) -> object:
    return binding("concept_list", (relation,), relation=relation)


def _denormalized(relation: str, group_by: str) -> object:
    return binding(
        "concept_with_values_denormalized", (relation,), relation=relation, group_by=group_by
    )


def _member(found: object) -> str:
    return str(shapes("precedence").member(found))


def _members_of(result: object, subject: str) -> set[str]:
    ambiguities = result.ambiguities  # type: ignore[attr-defined]
    (ambiguity,) = [a for a in ambiguities if a.subject == subject]
    return set(ambiguity.members)


@pytest.fixture(scope="module")
def library(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Source]:
    root = tmp_path_factory.mktemp("precedence_library")
    staging = Staging(str(root / "staging.duckdb"), schemas=())
    landed = land_database(LIBRARY, staging, schema="land_library")
    yield staging_source(staging, landed, root / "ledger" / "run")
    staging.close()


def test_relation_claimed_twice_keeps_its_single_relation_candidates() -> None:
    claims = [_edge_list("x", "y"), _edge_list("y", "x")]
    singles = [_listed(EDGES), _denormalized(EDGES, "x")]

    result = shapes().apply_precedence([*claims, *singles])

    assert _members_of(result, EDGES) == {_member(b) for b in [*claims, *singles]}
    assert not [b for b in result.candidates if b in [*claims, *singles]]


def test_every_relation_claimed_twice_keeps_its_own_single_candidates() -> None:
    claims = [_edge_list("x", "y"), _edge_list("y", "x")]
    concept_list = _listed(CONCEPTS)

    result = shapes().apply_precedence([*claims, concept_list])

    assert _members_of(result, CONCEPTS) == {_member(b) for b in [*claims, concept_list]}


def test_relation_claimed_once_drops_its_singles_even_beside_a_contested_relation() -> None:
    claims = [_edge_list("x", "y"), _edge_list("y", "x", edges=OTHER)]
    once = _listed(EDGES)

    result = shapes().apply_precedence([*claims, once])

    assert once not in result.candidates
    assert all(a.subject != EDGES for a in result.ambiguities)
    assert _members_of(result, CONCEPTS) == {_member(b) for b in claims}


def test_a_single_claim_drops_the_single_relation_candidates_of_its_relations() -> None:
    claim = _edge_list("x", "y")
    singles = [_listed(EDGES), _listed(CONCEPTS)]

    result = shapes().apply_precedence([claim, *singles])

    assert result.candidates == (claim,)
    assert result.ambiguities == ()


def test_conflicting_claims_do_not_depend_on_candidate_order() -> None:
    bindings = [_edge_list("x", "y"), _listed(EDGES), _edge_list("y", "x"), _listed(CONCEPTS)]

    forward = shapes().apply_precedence(bindings)
    backward = shapes().apply_precedence(list(reversed(bindings)))

    assert forward == backward


def test_a2_concept_sheet_with_one_claim_still_drops_concept_list(a2: Source) -> None:
    concepts = a2.ref(CONCEPTS_A2)
    found = every_template_candidates(a2.profiles(), a2)
    assert any(b.template == "concept_list" and b.relations == (concepts,) for b in found)

    result = shapes().apply_precedence(found)

    listed = [b for b in result.candidates if b.template == "concept_list"]
    assert all(concepts not in b.relations for b in listed)
    assert all(a.subject != concepts for a in result.ambiguities)


def test_library_data_element_ambiguity_offers_the_denormalized_reading(
    library: Source, tmp_path: Path
) -> None:
    """partner SL, database medium, no hint: the data element relation stays open with every reading.

    Needs three rules: "values are not concepts" (the crosswalk value set leaves), "conflicting
    claims open it" (two edge-list claims keep the relation's single-relation candidates), and
    "group is an entity" (the spurious synonym group key leaves; on HEAD since 53d084e, and
    dev-denorm's open work must keep it). It stays red until all three hold.
    """
    relation = library.ref(LIBRARY_VALUES)

    result = run(library, fresh_tools(library, tmp_path / "structure"))

    (ambiguity,) = [a for a in result.ambiguities if a.subject == relation]
    members = set(ambiguity.members)
    assert _member(_denormalized(relation, LIBRARY_GROUP_BY)) in members
    assert _member(_denormalized(relation, LIBRARY_SPURIOUS_GROUP_BY)) not in members
    assert not any(m.startswith("concept_and_value_set(") for m in members)
    assert sum(m.startswith("edge_list_hierarchy(") for m in members) == 2

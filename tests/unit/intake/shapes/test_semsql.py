"""semsql: eligible exactly when the loader's fixed relations are present."""

from __future__ import annotations

import pytest

from tests.unit.intake.shapes.shapes_env import (
    Source,
    candidates,
    every_template_candidates,
    shapes,
    verify,
)

pytestmark = pytest.mark.unit

ID = "semsql"
REQUIRED = (
    "semsql_terms",
    "semsql_edges",
    "semsql_synonyms",
    "semsql_mappings",
    "statements",
    "entailed_edge",
)
OTHER_CLAIMS_ON_A3 = (
    "amendments A and E hold, but the A3 namespace still yields 32 concept_and_value_set and"
    " 30 edge_list_hierarchy edge-relation candidates (e.g. class_node.id <- edge.object, and"
    " class_node with edge as the edge relation)"
    " plus 8 entity_model per_file candidates when the whole namespace is one group (prefix as"
    " header); each is a multi-relation claim on a relation semsql's namespace-wide claim also"
    " covers, and the spec leaves two multi-relation claims on one relation ambiguous and"
    " forbids preferring a template by identity, so 12 of the 15 A3 relations are ambiguous"
    " and nothing is kept"
)


def test_a3_is_eligible_with_one_binding_over_the_loader_relations(a3: Source) -> None:
    found = candidates(ID, a3.profiles(), a3)

    (only,) = found
    assert only.template == ID
    assert set(only.relations) >= {a3.ref(name) for name in REQUIRED}


def test_a3_binding_verifies_with_the_label_coverage_floor(a3: Source) -> None:
    group = a3.profiles()
    (found,) = candidates(ID, group, a3)

    verdict = verify(ID, found, group, a3)

    assert verdict.status == "pass"
    assert verdict.floors["LABEL_COVERAGE_FLOOR"] == shapes(ID).LABEL_COVERAGE_FLOOR


def test_namespace_missing_entailed_edge_is_not_eligible(a3: Source) -> None:
    group = [p for p in a3.profiles() if p.landed.name != "entailed_edge"]

    assert candidates(ID, group, a3) == []


def test_tabular_source_is_not_eligible(a2: Source) -> None:
    assert candidates(ID, a2.profiles(), a2) == []


@pytest.mark.xfail(strict=True, reason=OTHER_CLAIMS_ON_A3)
def test_semsql_is_the_only_template_left_for_the_a3_group(a3: Source) -> None:
    result = shapes().apply_precedence(every_template_candidates(a3.profiles(), a3))

    assert {b.template for b in result.candidates} == {ID}
    assert result.ambiguities == ()

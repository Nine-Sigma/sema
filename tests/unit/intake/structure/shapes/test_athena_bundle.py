"""athena_bundle: eligible exactly when every required Athena table is present."""

from __future__ import annotations

import pytest

from tests.unit.intake.structure.shapes.shapes_env import Source, candidates, verify

pytestmark = pytest.mark.unit

ID = "athena_bundle"
REQUIRED = ("concept", "concept_relationship", "concept_ancestor", "vocabulary", "domain")


def test_minimal_bundle_is_eligible_with_one_binding(athena: Source) -> None:
    (only,) = candidates(ID, athena.profiles(), athena)

    assert only.template == ID
    assert set(only.relations) >= {athena.ref(name) for name in REQUIRED}


def test_minimal_bundle_binding_verifies(athena: Source) -> None:
    group = athena.profiles()
    (found,) = candidates(ID, group, athena)

    assert verify(ID, found, group, athena).status == "pass"


def test_namespace_without_concept_relationship_is_not_eligible(athena: Source) -> None:
    group = [p for p in athena.profiles() if p.landed.name != "concept_relationship"]

    assert any(p.landed.name == "concept" for p in group)
    assert candidates(ID, group, athena) == []


def test_tabular_source_is_not_eligible(a2: Source) -> None:
    assert candidates(ID, a2.profiles(), a2) == []

"""Tests for TargetPropertyDecl + endpoint property fields."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from sema.models.target.properties import PropertyKind, TargetPropertyDecl


pytestmark = pytest.mark.unit


def test_columnar_property_minimum_fields() -> None:
    p = TargetPropertyDecl(name="person_id", type="int", nullable=False)
    assert p.property_kind is PropertyKind.COLUMN
    assert p.endpoint_role is None
    assert p.endpoint_target_entity_qualified_name is None
    assert p.materialized_as_edge_property is True


def test_columnar_property_round_trip() -> None:
    p = TargetPropertyDecl(
        name="gender_concept_id",
        type="int",
        nullable=False,
        synonyms=["gender", "sex"],
        decoded_values={"8507": "MALE", "8532": "FEMALE"},
    )
    assert TargetPropertyDecl.model_validate_json(p.model_dump_json()) == p


@pytest.mark.parametrize("reserved", ["subject", "object"])
def test_columnar_property_rejects_reserved_endpoint_names(reserved: str) -> None:
    with pytest.raises(ValidationError):
        TargetPropertyDecl(name=reserved, type="string", nullable=False)


def test_endpoint_kind_permits_reserved_names_for_normalizer_use() -> None:
    p = TargetPropertyDecl(
        name="subject",
        type="entity_ref",
        nullable=False,
        property_kind=PropertyKind.ENDPOINT,
        endpoint_role="subject",
        endpoint_target_entity_qualified_name="acris.LLC",
        endpoint_cardinality="one",
        endpoint_nullable=False,
        materialized_as_edge_property=False,
    )
    assert p.property_kind is PropertyKind.ENDPOINT
    assert p.endpoint_role == "subject"


def test_property_extra_field_rejected() -> None:
    with pytest.raises(ValidationError):
        TargetPropertyDecl(  # type: ignore[call-arg]
            name="x",
            type="int",
            nullable=True,
            wat=1,
        )


def test_property_frozen() -> None:
    p = TargetPropertyDecl(name="x", type="int", nullable=False)
    with pytest.raises(ValidationError):
        p.name = "y"  # type: ignore[misc]

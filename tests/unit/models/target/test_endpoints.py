"""Tests for EdgeEndpointDecl and EdgeEndpointsDecl."""

from __future__ import annotations

import pytest

from sema.models.planner._enums import TargetArtifactKind
from sema.models.target.endpoints import EdgeEndpointDecl, EdgeEndpointsDecl
from sema.models.target.refs import TargetEntityRef


pytestmark = pytest.mark.unit


def _ref(name: str, kind: TargetArtifactKind = TargetArtifactKind.GRAPH_NODE) -> TargetEntityRef:
    return TargetEntityRef(target_model_id="acris", qualified_name=f"acris.{name}", kind=kind)


def test_endpoint_decl_defaults() -> None:
    e = EdgeEndpointDecl(role="subject", target_entity=_ref("LLC"))
    assert e.cardinality == "one"
    assert e.nullable is False


def test_endpoints_decl_round_trip() -> None:
    eds = EdgeEndpointsDecl(
        subject=EdgeEndpointDecl(role="subject", target_entity=_ref("LLC")),
        object=EdgeEndpointDecl(role="object", target_entity=_ref("Property")),
    )
    blob = eds.model_dump_json()
    assert EdgeEndpointsDecl.model_validate_json(blob) == eds

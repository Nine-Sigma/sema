"""Tests for TargetEntityDecl + endpoints/kind invariants."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from sema.models.planner._enums import TargetArtifactKind
from sema.models.target.endpoints import EdgeEndpointDecl, EdgeEndpointsDecl
from sema.models.target.entity import TargetEntityDecl
from sema.models.target.properties import TargetPropertyDecl
from sema.models.target.refs import TargetEntityRef


pytestmark = pytest.mark.unit


def _ref(name: str, kind: TargetArtifactKind) -> TargetEntityRef:
    return TargetEntityRef(target_model_id="acris", qualified_name=name, kind=kind)


def _columnar_property(name: str = "id") -> TargetPropertyDecl:
    return TargetPropertyDecl(name=name, type="int", nullable=False)


def _endpoints() -> EdgeEndpointsDecl:
    return EdgeEndpointsDecl(
        subject=EdgeEndpointDecl(
            role="subject",
            target_entity=_ref("acris.LLC", TargetArtifactKind.GRAPH_NODE),
        ),
        object=EdgeEndpointDecl(
            role="object",
            target_entity=_ref("acris.Property", TargetArtifactKind.GRAPH_NODE),
        ),
    )


def test_table_row_entity_no_endpoints() -> None:
    e = TargetEntityDecl(
        ref=_ref("omop.person", TargetArtifactKind.TABLE_ROW),
        properties=[_columnar_property()],
    )
    assert e.endpoints is None


def test_graph_edge_entity_requires_endpoints() -> None:
    with pytest.raises(ValidationError):
        TargetEntityDecl(
            ref=_ref("acris.OWNS", TargetArtifactKind.GRAPH_EDGE),
            properties=[_columnar_property("valid_from")],
        )


def test_table_row_entity_rejects_endpoints() -> None:
    with pytest.raises(ValidationError):
        TargetEntityDecl(
            ref=_ref("omop.person", TargetArtifactKind.TABLE_ROW),
            properties=[_columnar_property()],
            endpoints=_endpoints(),
        )


def test_graph_node_entity_rejects_endpoints() -> None:
    with pytest.raises(ValidationError):
        TargetEntityDecl(
            ref=_ref("acris.LLC", TargetArtifactKind.GRAPH_NODE),
            properties=[_columnar_property()],
            endpoints=_endpoints(),
        )


def test_graph_edge_entity_round_trip() -> None:
    e = TargetEntityDecl(
        ref=_ref("acris.OWNS", TargetArtifactKind.GRAPH_EDGE),
        properties=[_columnar_property("valid_from")],
        endpoints=_endpoints(),
    )
    assert TargetEntityDecl.model_validate_json(e.model_dump_json()) == e

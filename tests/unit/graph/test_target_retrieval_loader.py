"""Unit tests for the target-retrieval projection helpers (no Neo4j)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from sema.graph.physical_catalog import (
    CatalogDriftError,
    PhysicalColumn,
    StaticPhysicalCatalogSource,
)
from sema.graph.target_retrieval_loader import (
    EntityTableBinding,
    PhysicalSchemaBinding,
    project_target_to_retrieval_graph,
)
from sema.graph.target_retrieval_loader_utils import (
    CATEGORICAL,
    entity_id,
    property_id,
    semantic_type_for,
)
from sema.models.planner._enums import TargetArtifactKind
from sema.models.target.completeness import (
    SemanticCompleteness,
    SemanticCompletenessAnnotations,
)
from sema.models.target.descriptor import TargetModelDescriptor
from sema.models.target.entity import TargetEntityDecl
from sema.models.target.normalized import NormalizedTargetModel
from sema.models.target.properties import TargetPropertyDecl
from sema.models.target.refs import TargetEntityRef

pytestmark = pytest.mark.unit


def _prop(name: str, type_: str, *, bound: bool = False) -> TargetPropertyDecl:
    return TargetPropertyDecl(
        name=name, type=type_, nullable=True,
        vocabulary_binding="V" if bound else None,
    )


def _model(props: list[TargetPropertyDecl]) -> NormalizedTargetModel:
    ref = TargetEntityRef(
        target_model_id="m", qualified_name="ns.thing",
        kind=TargetArtifactKind.TABLE_ROW,
    )
    return NormalizedTargetModel(
        descriptor=TargetModelDescriptor(
            target_model_id="m", target_model_version="1", display_name="M",
            completeness=SemanticCompletenessAnnotations(
                structure=SemanticCompleteness.COMPLETE,
                obligations=SemanticCompleteness.NONE,
                vocabulary_bindings=SemanticCompleteness.NONE,
                semantic_aliases=SemanticCompleteness.NONE,
                terms=SemanticCompleteness.NONE,
            ),
        ),
        entities=[TargetEntityDecl(ref=ref, properties=props)],
    )


def test_semantic_type_categorical_when_vocabulary_bound() -> None:
    assert semantic_type_for(_prop("x", "integer", bound=True)) == CATEGORICAL


def test_semantic_type_maps_scalar_when_unbound() -> None:
    assert semantic_type_for(_prop("x", "integer")) == "numeric"
    assert semantic_type_for(_prop("x", "date")) == "temporal"
    assert semantic_type_for(_prop("x", "varchar")) == "free_text"


def test_deterministic_ids() -> None:
    assert entity_id(model_role="TARGET", schema_name="s", name="t") == "TARGET|s|t"
    assert property_id(
        model_role="TARGET", schema_name="s", entity_name="t", name="c"
    ) == "TARGET|s|t.c"


def test_binding_table_lookup() -> None:
    binding = PhysicalSchemaBinding(
        catalog="c", schema="s",
        entity_tables=(EntityTableBinding("ns.thing", "thing"),),
    )
    assert binding.table_for("ns.thing") == "thing"
    assert binding.table_for("ns.missing") is None


def test_projection_raises_on_catalog_drift() -> None:
    model = _model([_prop("present", "integer"), _prop("absent", "integer")])
    binding = PhysicalSchemaBinding(
        catalog="c", schema="s",
        entity_tables=(EntityTableBinding("ns.thing", "thing"),),
    )
    catalog = StaticPhysicalCatalogSource(
        {("c", "s", "thing"): [PhysicalColumn("present", "INT", True)]}
    )
    with pytest.raises(CatalogDriftError, match="absent"):
        project_target_to_retrieval_graph(MagicMock(), model, binding, catalog)

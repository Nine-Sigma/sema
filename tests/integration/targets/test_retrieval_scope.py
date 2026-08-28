"""S2-07 integration (M6): with source + target layers coexisting, a scoped
retrieval resolves a shared entity name to its own layer's binding."""

from __future__ import annotations

import pytest

from sema.graph.loader import GraphLoader
from sema.graph.physical_catalog import (
    PhysicalColumn,
    StaticPhysicalCatalogSource,
)
from sema.graph.target_retrieval_loader import (
    EntityTableBinding,
    PhysicalSchemaBinding,
    project_target_to_retrieval_graph,
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
from sema.pipeline.retrieval import RetrievalEngine, RetrievalScope

pytestmark = pytest.mark.integration

_SHARED = "widget"


def _seed_source(driver) -> None:
    loader = GraphLoader(driver)
    loader.upsert_table(_SHARED, "src_schema", "cat", ref="cat.src_schema.widget")
    loader.upsert_column(
        "id", _SHARED, "src_schema", "cat", data_type="INT",
        ref="cat.src_schema.widget.id",
    )
    loader.upsert_entity(
        _SHARED, description=None, source="build", confidence=0.9,
        table_name=_SHARED, schema_name="src_schema", catalog="cat",
        source_schema="src_schema",
    )


def _target_model() -> NormalizedTargetModel:
    ref = TargetEntityRef(
        target_model_id="m", qualified_name="tgt.widget",
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
        entities=[TargetEntityDecl(
            ref=ref,
            properties=[TargetPropertyDecl(name="id", type="integer", nullable=False)],
        )],
    )


def _seed_target(driver) -> None:
    binding = PhysicalSchemaBinding(
        catalog="cat", schema="tgt_schema",
        entity_tables=(EntityTableBinding("tgt.widget", _SHARED),),
    )
    catalog = StaticPhysicalCatalogSource(
        {("cat", "tgt_schema", _SHARED): [PhysicalColumn("id", "BIGINT", False)]}
    )
    project_target_to_retrieval_graph(
        GraphLoader(driver), _target_model(), binding, catalog
    )


def _schemas(engine: RetrievalEngine) -> set[str]:
    physical = engine.expand_from_entities([_SHARED])["physical"]
    return {r["schema_name"] for r in physical}


def test_target_scope_returns_only_target_binding(clean_neo4j) -> None:
    _seed_source(clean_neo4j)
    _seed_target(clean_neo4j)
    engine = RetrievalEngine(
        clean_neo4j, scope=RetrievalScope(model_role="TARGET")
    )
    assert _schemas(engine) == {"tgt_schema"}


def test_source_scope_returns_only_source_binding(clean_neo4j) -> None:
    _seed_source(clean_neo4j)
    _seed_target(clean_neo4j)
    engine = RetrievalEngine(
        clean_neo4j, scope=RetrievalScope(model_role="SOURCE")
    )
    assert _schemas(engine) == {"src_schema"}


def test_unscoped_returns_both_bindings(clean_neo4j) -> None:
    _seed_source(clean_neo4j)
    _seed_target(clean_neo4j)
    engine = RetrievalEngine(clean_neo4j)
    assert _schemas(engine) == {"src_schema", "tgt_schema"}

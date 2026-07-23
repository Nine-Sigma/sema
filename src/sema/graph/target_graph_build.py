"""Generic composition of the target-retrieval graph (the S2-03/04/05/06 path).

One entry point that projects a target semantic model + physical binding into
the retrieval graph, materializes its FK join paths, populates Column-anchored
concept value sets, and writes the value-level cross-layer bridges. A thin
adapter (e.g. a showcase command) supplies the concrete manifest, physical
binding, catalog source, concept source, and the resolved value bindings read
from a value-mapping store.

Fully target-agnostic: all target/domain specifics arrive as data.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from sema.graph.concept_source import ConceptSource
from sema.graph.concept_value_set import (
    ValueSetColumn,
    materialize_concept_value_set,
)
from sema.graph.cross_layer_bridge import write_value_bridge
from sema.graph.loader import GraphLoader
from sema.graph.physical_catalog import PhysicalCatalogSource
from sema.graph.target_join_paths import materialize_target_join_paths
from sema.graph.target_retrieval_loader import (
    PhysicalSchemaBinding,
    project_target_to_retrieval_graph,
)
from sema.models.target.normalized import NormalizedTargetModel


@dataclass(frozen=True)
class ConceptFieldSpec:
    """A concept-governed target column and the codes its values resolve to."""

    entity_qualified_name: str
    property_name: str
    concept_codes: tuple[str, ...]


@dataclass(frozen=True)
class ValueBridgeSpec:
    """A resolved source value → target concept crosswalk (one edge)."""

    source_vocabulary: str
    source_code: str
    concept_code: str


def build_target_retrieval_graph(
    loader: GraphLoader,
    normalized: NormalizedTargetModel,
    binding: PhysicalSchemaBinding,
    catalog: PhysicalCatalogSource,
    *,
    concepts: ConceptSource,
    concept_vocabulary: str,
    concept_fields: Sequence[ConceptFieldSpec],
    value_bridges: Sequence[ValueBridgeSpec],
    bridge_source_schema: str,
    load_ancestors: bool = True,
) -> None:
    project_target_to_retrieval_graph(loader, normalized, binding, catalog)
    materialize_target_join_paths(loader, normalized, binding)
    for field in concept_fields:
        table = binding.table_for(field.entity_qualified_name)
        if table is None:
            continue
        materialize_concept_value_set(
            loader,
            column=ValueSetColumn(
                binding.catalog, binding.schema, table, field.property_name
            ),
            concept_vocabulary=concept_vocabulary,
            concept_codes=field.concept_codes,
            concepts=concepts,
            source_schema=binding.schema,
            load_ancestors=load_ancestors,
        )
    for bridge in value_bridges:
        write_value_bridge(
            loader,
            source_vocabulary=bridge.source_vocabulary,
            source_code=bridge.source_code,
            concept_vocabulary=concept_vocabulary,
            concept_code=bridge.concept_code,
            source_schema=bridge_source_schema,
        )


__all__ = [
    "ConceptFieldSpec",
    "ValueBridgeSpec",
    "build_target_retrieval_graph",
]

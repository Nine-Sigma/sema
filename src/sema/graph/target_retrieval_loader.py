"""Project a normalized target semantic model into the *retrieval* graph.

Slice-2 reconciliation A (M1/M2/G-phys): the abstract target materializer
(`sema.targets`) writes a governance model keyed by ``qualified_name``, which
retrieval cannot read. This loader writes a second projection in the exact
source contract retrieval reads — ``:Entity {name}`` / ``:Property
{entity_name, name, semantic_type}`` bound to physical ``:Table`` / ``:Column``
nodes read from the live catalog (D1) — so a materialized target becomes
reachable by ``resolve_physical_mapping``.

Fully target-agnostic: input is ``(normalized model, schema binding, catalog
source)``. Target/domain specifics live in the manifest + showcase.
"""

from __future__ import annotations

from dataclasses import dataclass

from sema.graph.loader import GraphLoader
from sema.graph.physical_catalog import (
    CatalogDriftError,
    PhysicalCatalogSource,
    PhysicalColumn,
)
from sema.graph.target_retrieval_loader_utils import (
    is_column_property,
    semantic_type_for,
    upsert_target_entity,
    upsert_target_property,
)
from sema.models.target.normalized import NormalizedTargetModel

_DEFAULT_MODEL_ROLE = "TARGET"


@dataclass(frozen=True)
class EntityTableBinding:
    """Binds a target entity's ``qualified_name`` to a physical table."""

    entity_qualified_name: str
    table: str


@dataclass(frozen=True)
class PhysicalSchemaBinding:
    """Physical placement of a target model in one catalog.schema."""

    catalog: str
    schema: str
    entity_tables: tuple[EntityTableBinding, ...]
    model_role: str = _DEFAULT_MODEL_ROLE

    def table_for(self, qualified_name: str) -> str | None:
        for binding in self.entity_tables:
            if binding.entity_qualified_name == qualified_name:
                return binding.table
        return None


def project_target_to_retrieval_graph(
    loader: GraphLoader,
    normalized: NormalizedTargetModel,
    binding: PhysicalSchemaBinding,
    catalog: PhysicalCatalogSource,
) -> None:
    """Write the retrieval-shaped projection of ``normalized`` into the graph."""
    for entity in normalized.entities:
        table = binding.table_for(entity.ref.qualified_name)
        if table is None:
            continue
        physical = _read_physical_columns(binding, table, catalog)
        _load_physical_table(loader, binding, table, physical)
        upsert_target_entity(
            loader,
            name=table,
            schema_name=binding.schema,
            catalog=binding.catalog,
            table=table,
            table_ref=_table_ref(binding, table),
            model_role=binding.model_role,
            source_schema=binding.schema,
        )
        for prop in entity.properties:
            if not is_column_property(prop):
                continue
            _assert_column_present(prop.name, physical, binding, table)
            upsert_target_property(
                loader,
                name=prop.name,
                entity_name=table,
                semantic_type=semantic_type_for(prop),
                schema_name=binding.schema,
                catalog=binding.catalog,
                table=table,
                column_name=prop.name,
                model_role=binding.model_role,
                source_schema=binding.schema,
            )


def _read_physical_columns(
    binding: PhysicalSchemaBinding,
    table: str,
    catalog: PhysicalCatalogSource,
) -> dict[str, PhysicalColumn]:
    columns = catalog.columns(
        catalog=binding.catalog, schema=binding.schema, table=table
    )
    return {col.name: col for col in columns}


def _load_physical_table(
    loader: GraphLoader,
    binding: PhysicalSchemaBinding,
    table: str,
    physical: dict[str, PhysicalColumn],
) -> None:
    table_ref = _table_ref(binding, table)
    loader.upsert_table(table, binding.schema, binding.catalog, ref=table_ref)
    for col in physical.values():
        loader.upsert_column(
            col.name, table, binding.schema, binding.catalog,
            data_type=col.data_type, nullable=col.nullable,
            ref=f"{table_ref}.{col.name}",
        )


def _assert_column_present(
    column: str,
    physical: dict[str, PhysicalColumn],
    binding: PhysicalSchemaBinding,
    table: str,
) -> None:
    if column not in physical:
        raise CatalogDriftError(
            f"manifest property {column!r} has no column in "
            f"{binding.catalog}.{binding.schema}.{table}"
        )


def _table_ref(binding: PhysicalSchemaBinding, table: str) -> str:
    return f"{binding.catalog}.{binding.schema}.{table}"


__all__ = [
    "EntityTableBinding",
    "PhysicalSchemaBinding",
    "project_target_to_retrieval_graph",
]

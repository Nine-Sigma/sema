"""Helpers for the target-retrieval projection loader.

Deterministic id derivation, semantic-type mapping, and the Cypher MERGE
shapes that write a target semantic model in the *source contract* retrieval
reads (M1/M2): `:Entity {name}` keyed by name + schema + model_role (D5
option a), `:Property {entity_name, name, semantic_type}`, and the physical
`ENTITY_ON_TABLE` / `PROPERTY_ON_COLUMN` edges. No domain literals — every
value comes from the caller.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sema.models.target.properties import PropertyKind, TargetPropertyDecl

if TYPE_CHECKING:
    from sema.graph.loader import GraphLoader

CATEGORICAL = "categorical"

_SCALAR_SEMANTIC_TYPE = {
    "integer": "numeric",
    "int": "numeric",
    "bigint": "numeric",
    "float": "numeric",
    "double": "numeric",
    "decimal": "numeric",
    "number": "numeric",
    "date": "temporal",
    "datetime": "temporal",
    "timestamp": "temporal",
    "boolean": "boolean",
    "bool": "boolean",
}


def entity_id(*, model_role: str, schema_name: str, name: str) -> str:
    return f"{model_role}|{schema_name}|{name}"


def property_id(
    *, model_role: str, schema_name: str, entity_name: str, name: str
) -> str:
    return f"{model_role}|{schema_name}|{entity_name}.{name}"


def semantic_type_for(prop: TargetPropertyDecl) -> str:
    """A value-set-governed property is categorical; else map its scalar type.

    Only ``categorical`` is load-bearing for retrieval (it gates value-set
    expansion in ``_expand_values``); the rest are informational.
    """
    if prop.vocabulary_binding is not None:
        return CATEGORICAL
    return _SCALAR_SEMANTIC_TYPE.get(prop.type.lower(), "free_text")


def is_column_property(prop: TargetPropertyDecl) -> bool:
    return prop.property_kind is PropertyKind.COLUMN


def upsert_target_entity(
    loader: GraphLoader,
    *,
    name: str,
    schema_name: str,
    catalog: str,
    table: str,
    table_ref: str,
    model_role: str,
    source_schema: str,
) -> None:
    """MERGE a target `:Entity` keyed on name+schema+model_role (D5-a).

    A wider key than the source loader's ``{name}`` so a target entity never
    collapses into a same-named source entity; retrieval then scopes by
    model_role/schema (M6/S2-07).
    """
    loader._run(
        "MERGE (e:Entity {name: $name, schema_name: $schema_name, "
        "model_role: $model_role}) "
        "ON CREATE SET e.id = $id "
        "SET e.source_id = $source_schema "
        "WITH e "
        "MERGE (t:Table {name: $table, schema_name: $schema_name, "
        "catalog: $catalog}) "
        "ON CREATE SET t.ref = $table_ref "
        "MERGE (e)-[:ENTITY_ON_TABLE {source_schema: $source_schema}]->(t)",
        name=name,
        schema_name=schema_name,
        model_role=model_role,
        id=entity_id(model_role=model_role, schema_name=schema_name, name=name),
        source_schema=source_schema,
        table=table,
        catalog=catalog,
        table_ref=table_ref,
    )


def upsert_target_property(
    loader: GraphLoader,
    *,
    name: str,
    entity_name: str,
    semantic_type: str,
    schema_name: str,
    catalog: str,
    table: str,
    column_name: str,
    model_role: str,
    source_schema: str,
) -> None:
    """MERGE a target `:Property {entity_name, name, semantic_type}` and its
    physical `PROPERTY_ON_COLUMN` / owning `HAS_PROPERTY` edges (M2)."""
    loader._run(
        "MERGE (p:Property {entity_name: $entity_name, name: $name, "
        "schema_name: $schema_name, model_role: $model_role}) "
        "ON CREATE SET p.id = $id "
        "SET p.semantic_type = $semantic_type, p.source_id = $source_schema "
        "WITH p "
        "MERGE (e:Entity {name: $entity_name, schema_name: $schema_name, "
        "model_role: $model_role}) "
        "MERGE (e)-[:HAS_PROPERTY {source_schema: $source_schema}]->(p) "
        "WITH p "
        "MERGE (c:Column {name: $column_name, table_name: $table, "
        "schema_name: $schema_name, catalog: $catalog}) "
        "MERGE (p)-[:PROPERTY_ON_COLUMN {source_schema: $source_schema}]->(c)",
        name=name,
        entity_name=entity_name,
        schema_name=schema_name,
        model_role=model_role,
        id=property_id(
            model_role=model_role, schema_name=schema_name,
            entity_name=entity_name, name=name,
        ),
        semantic_type=semantic_type,
        source_schema=source_schema,
        column_name=column_name,
        table=table,
        catalog=catalog,
    )


__all__ = [
    "CATEGORICAL",
    "entity_id",
    "is_column_property",
    "property_id",
    "semantic_type_for",
    "upsert_target_entity",
    "upsert_target_property",
]

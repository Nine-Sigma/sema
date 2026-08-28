"""Re-load lifecycle for the target-retrieval projection (D4).

Decision: the projection is a *rebuildable* view keyed on stable D5-a
identities (name + schema + model_role), so a re-load is an idempotent
in-place MERGE — NOT a hash-versioned new generation (that is the model D5
rejected). Re-running S2-03/04/05/06 therefore leaves node/edge counts stable.

For the rare case where the target model *shrinks* (a field/entity is removed),
``delete_target_layer`` scope-deletes the target layer for one physical schema
before a rebuild. It deletes only schema-scoped, target-role nodes — it leaves
the shared concept spine (``:Term`` + ``MAPS_TO_CONCEPT``, scoped by the source
study) untouched, so removing one target's layer never breaks another study's
bridge.
"""

from __future__ import annotations

from sema.graph.loader import GraphLoader


def delete_target_layer(
    loader: GraphLoader,
    *,
    schema_name: str,
    catalog: str,
    model_role: str = "TARGET",
) -> None:
    """Remove the target semantic + physical + join projection for a schema.

    Concept ``:Term`` nodes and their ``MAPS_TO_CONCEPT`` edges are preserved:
    they are the shared spine, scoped by the source study, not by this target
    schema.
    """
    loader._run(
        "MATCH (e:Entity {schema_name: $schema_name, model_role: $model_role}) "
        "DETACH DELETE e",
        schema_name=schema_name, model_role=model_role,
    )
    loader._run(
        "MATCH (p:Property {schema_name: $schema_name, "
        "model_role: $model_role}) DETACH DELETE p",
        schema_name=schema_name, model_role=model_role,
    )
    loader._run(
        "MATCH (vs:ValueSet) WHERE vs.column_ref STARTS WITH $prefix "
        "DETACH DELETE vs",
        prefix=f"{catalog}.{schema_name}.",
    )
    loader._run(
        "MATCH (jp:JoinPath {source_schema: $schema_name}) DETACH DELETE jp",
        schema_name=schema_name,
    )
    loader._run(
        "MATCH (c:Column {schema_name: $schema_name}) DETACH DELETE c",
        schema_name=schema_name,
    )
    loader._run(
        "MATCH (t:Table {schema_name: $schema_name}) DETACH DELETE t",
        schema_name=schema_name,
    )


__all__ = ["delete_target_layer"]

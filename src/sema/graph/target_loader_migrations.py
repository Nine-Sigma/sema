"""Cypher migrations for the target-model-loader storage shape.

Extends the planner contract's `planner-graph-storage` migrations with
the target-side schema artifacts produced by `TargetModelMaterializer`:
the `EnrichmentDecision` label, hash-versioned uniqueness constraints,
and indexes used by enrichment-status query workloads.
"""

from __future__ import annotations


_FACETS = (
    "structure",
    "obligations",
    "vocabulary_bindings",
    "semantic_aliases",
    "terms",
)


def cypher_up() -> list[str]:
    """Forward migration: target-loader uniqueness constraints + indexes."""
    statements: list[str] = []
    statements.extend(_uniqueness_constraints())
    statements.extend(_indexes())
    return statements


def cypher_down() -> list[str]:
    """Reverse migration: drop target-loader constraints, indexes, labels."""
    statements: list[str] = []
    statements.extend(
        f"DROP INDEX entity_enrichment_{f}_status IF EXISTS" for f in _FACETS
    )
    statements.extend(
        [
            "DROP INDEX entity_is_current IF EXISTS",
            "DROP INDEX property_property_kind IF EXISTS",
            "DROP INDEX property_is_current IF EXISTS",
            "DROP INDEX target_obligation_is_current IF EXISTS",
            "DROP INDEX target_term_is_current IF EXISTS",
            "DROP INDEX target_constraint_is_current IF EXISTS",
            "DROP INDEX target_vocab_binding_is_current IF EXISTS",
            "DROP INDEX target_context_card_is_current IF EXISTS",
            "DROP CONSTRAINT enrichment_decision_unique IF EXISTS",
            "DROP CONSTRAINT target_entity_hash_unique IF EXISTS",
            "DROP CONSTRAINT target_property_hash_unique IF EXISTS",
            "DROP CONSTRAINT target_obligation_hash_unique IF EXISTS",
            "DROP CONSTRAINT target_term_hash_unique IF EXISTS",
            "DROP CONSTRAINT target_constraint_hash_unique IF EXISTS",
            "DROP CONSTRAINT target_vocab_binding_hash_unique IF EXISTS",
            "DROP CONSTRAINT target_context_card_hash_unique IF EXISTS",
            "MATCH (n:EnrichmentDecision) DETACH DELETE n",
            "MATCH (n:VocabularyBinding) DETACH DELETE n",
            "MATCH (n:ContextCard) DETACH DELETE n",
        ]
    )
    return statements


def _uniqueness_constraints() -> list[str]:
    return [
        "CREATE CONSTRAINT enrichment_decision_unique IF NOT EXISTS "
        "FOR (n:EnrichmentDecision) "
        "REQUIRE (n.target_model_id, n.target_model_version, "
        "n.target_schema_snapshot_hash, n.entity_ref) IS UNIQUE",
        "CREATE CONSTRAINT target_entity_hash_unique IF NOT EXISTS "
        "FOR (n:Entity) "
        "REQUIRE (n.target_model_id, n.target_model_version, "
        "n.target_schema_snapshot_hash, n.qualified_name) IS UNIQUE",
        "CREATE CONSTRAINT target_property_hash_unique IF NOT EXISTS "
        "FOR (n:Property) "
        "REQUIRE (n.target_model_id, n.target_model_version, "
        "n.target_schema_snapshot_hash, n.parent_entity_qualified_name, n.name) "
        "IS UNIQUE",
        "CREATE CONSTRAINT target_obligation_hash_unique IF NOT EXISTS "
        "FOR (n:TargetObligation) "
        "REQUIRE (n.target_model_id, n.target_model_version, "
        "n.target_schema_snapshot_hash, n.target_entity) IS UNIQUE",
        "CREATE CONSTRAINT target_term_hash_unique IF NOT EXISTS "
        "FOR (n:Term) "
        "REQUIRE (n.target_model_id, n.target_model_version, "
        "n.target_schema_snapshot_hash, n.vocabulary_name, n.code) IS UNIQUE",
        "CREATE CONSTRAINT target_constraint_hash_unique IF NOT EXISTS "
        "FOR (n:Constraint) "
        "REQUIRE (n.target_model_id, n.target_model_version, "
        "n.target_schema_snapshot_hash, n.attached_property_id, "
        "n.constraint_kind, n.payload_hash) IS UNIQUE",
        "CREATE CONSTRAINT target_vocab_binding_hash_unique IF NOT EXISTS "
        "FOR (n:VocabularyBinding) "
        "REQUIRE (n.target_model_id, n.target_model_version, "
        "n.target_schema_snapshot_hash, n.parent_entity_qualified_name, "
        "n.property_name, n.vocabulary_name) IS UNIQUE",
        "CREATE CONSTRAINT target_context_card_hash_unique IF NOT EXISTS "
        "FOR (n:ContextCard) "
        "REQUIRE (n.target_model_id, n.target_model_version, "
        "n.target_schema_snapshot_hash, n.entity_qualified_name, "
        "n.card_version) IS UNIQUE",
    ]


def _indexes() -> list[str]:
    indexes = [
        f"CREATE INDEX entity_enrichment_{f}_status IF NOT EXISTS "
        f"FOR (n:Entity) ON (n.enrichment_{f}_status)"
        for f in _FACETS
    ]
    indexes.extend(
        [
            "CREATE INDEX entity_is_current IF NOT EXISTS "
            "FOR (n:Entity) ON (n.is_current)",
            "CREATE INDEX property_property_kind IF NOT EXISTS "
            "FOR (n:Property) ON (n.property_kind)",
            "CREATE INDEX property_is_current IF NOT EXISTS "
            "FOR (n:Property) ON (n.is_current)",
            "CREATE INDEX target_obligation_is_current IF NOT EXISTS "
            "FOR (n:TargetObligation) ON (n.is_current)",
            "CREATE INDEX target_term_is_current IF NOT EXISTS "
            "FOR (n:Term) ON (n.is_current)",
            "CREATE INDEX target_constraint_is_current IF NOT EXISTS "
            "FOR (n:Constraint) ON (n.is_current)",
            "CREATE INDEX target_vocab_binding_is_current IF NOT EXISTS "
            "FOR (n:VocabularyBinding) ON (n.is_current)",
            "CREATE INDEX target_context_card_is_current IF NOT EXISTS "
            "FOR (n:ContextCard) ON (n.is_current)",
        ]
    )
    return indexes


__all__ = ["cypher_up", "cypher_down"]

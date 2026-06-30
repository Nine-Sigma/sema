"""Unit checks for the target-loader Cypher migration shape (5.12, 5.13)."""

from __future__ import annotations

import pytest

from sema.graph.target_loader_migrations import cypher_down, cypher_up

pytestmark = pytest.mark.unit


def test_up_declares_enrichment_decision_uniqueness() -> None:
    stmts = cypher_up()
    assert any(
        "CONSTRAINT enrichment_decision_unique" in s
        and "EnrichmentDecision" in s
        and "target_schema_snapshot_hash" in s
        and "n.entity_ref" in s
        for s in stmts
    )


def test_up_declares_hash_versioned_entity_uniqueness() -> None:
    stmts = cypher_up()
    assert any(
        "target_entity_hash_unique" in s
        and "Entity" in s
        and "target_schema_snapshot_hash" in s
        and "qualified_name" in s
        for s in stmts
    )


def test_up_declares_hash_versioned_property_uniqueness() -> None:
    stmts = cypher_up()
    assert any(
        "target_property_hash_unique" in s
        and "Property" in s
        and "parent_entity_qualified_name" in s
        and "target_schema_snapshot_hash" in s
        for s in stmts
    )


def test_up_declares_hash_versioned_obligation_uniqueness() -> None:
    stmts = cypher_up()
    assert any(
        "target_obligation_hash_unique" in s
        and "TargetObligation" in s
        and "target_schema_snapshot_hash" in s
        and "target_entity" in s
        for s in stmts
    )


def test_up_declares_indexes_on_five_facets_status() -> None:
    stmts = cypher_up()
    facets = (
        "structure",
        "obligations",
        "vocabulary_bindings",
        "semantic_aliases",
        "terms",
    )
    for f in facets:
        assert any(
            f"entity_enrichment_{f}_status" in s
            and f"enrichment_{f}_status" in s
            and "Entity" in s
            for s in stmts
        ), f"missing index for facet {f}"


def test_up_declares_is_current_indexes() -> None:
    stmts = cypher_up()
    assert any("entity_is_current" in s for s in stmts)
    assert any("property_is_current" in s for s in stmts)


def test_up_declares_property_kind_index() -> None:
    stmts = cypher_up()
    assert any("property_property_kind" in s for s in stmts)


def test_down_drops_every_constraint_and_index_added_by_up() -> None:
    up = cypher_up()
    down = cypher_down()
    drops = "\n".join(down)
    assert "DROP CONSTRAINT enrichment_decision_unique" in drops
    assert "DROP CONSTRAINT target_entity_hash_unique" in drops
    assert "DROP CONSTRAINT target_property_hash_unique" in drops
    assert "DROP CONSTRAINT target_obligation_hash_unique" in drops
    for f in (
        "structure",
        "obligations",
        "vocabulary_bindings",
        "semantic_aliases",
        "terms",
    ):
        assert f"DROP INDEX entity_enrichment_{f}_status" in drops
    assert "DROP INDEX entity_is_current" in drops
    assert "DROP INDEX property_property_kind" in drops
    assert "DROP INDEX property_is_current" in drops
    assert "MATCH (n:EnrichmentDecision)" in drops
    assert len(up) > 0


def test_up_declares_term_constraint_vocab_binding_card_uniqueness() -> None:
    stmts = "\n".join(cypher_up())
    assert "target_term_hash_unique" in stmts and "Term" in stmts
    assert "target_constraint_hash_unique" in stmts and "Constraint" in stmts
    assert "target_vocab_binding_hash_unique" in stmts
    assert "target_context_card_hash_unique" in stmts


def test_up_declares_is_current_indexes_for_all_versioned_labels() -> None:
    stmts = "\n".join(cypher_up())
    for name in (
        "target_obligation_is_current",
        "target_term_is_current",
        "target_constraint_is_current",
        "target_vocab_binding_is_current",
        "target_context_card_is_current",
    ):
        assert name in stmts


def test_down_drops_new_uniqueness_constraints_and_labels() -> None:
    drops = "\n".join(cypher_down())
    for c in (
        "DROP CONSTRAINT target_term_hash_unique",
        "DROP CONSTRAINT target_constraint_hash_unique",
        "DROP CONSTRAINT target_vocab_binding_hash_unique",
        "DROP CONSTRAINT target_context_card_hash_unique",
        "MATCH (n:VocabularyBinding)",
        "MATCH (n:ContextCard)",
    ):
        assert c in drops

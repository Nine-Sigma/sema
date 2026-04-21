"""Generic (industry-agnostic) few-shot examples for staged L2 prompts.

These examples teach the LLM the archetypal table and column patterns that
appear across every industry: event-stream, transaction-N-per-parent,
dimension, bridge, and wide-profile tables; identifier / temporal / numeric /
categorical / free-text / boolean / ordinal columns; and common value-
decoding patterns. They form the base layer under every domain-specific
few-shot pack.
"""
from __future__ import annotations

from typing import Any

# --------------------------------------------------------------------------
# Generic Stage A examples — table archetypes
# --------------------------------------------------------------------------

GENERIC_STAGE_A: list[dict[str, Any]] = [
    {
        "input": {
            "table_name": "events",
            "columns": "event_id (STRING), actor_id (STRING), "
            "event_type (STRING), occurred_at (TIMESTAMP), "
            "payload (STRING)",
        },
        "output": {
            "primary_entity": "Event",
            "grain_hypothesis": "one row per event occurrence",
            "secondary_entity_hints": ["actor", "event type"],
            "ambiguity_flags": [],
            "confidence": 0.9,
        },
    },
    {
        "input": {
            "table_name": "orders",
            "columns": "order_id (STRING), customer_id (STRING), "
            "total_amount (DECIMAL), placed_at (TIMESTAMP), "
            "status (STRING)",
        },
        "output": {
            "primary_entity": "Order",
            "grain_hypothesis": "one row per order "
            "(multiple orders per customer)",
            "secondary_entity_hints": ["customer", "order status"],
            "ambiguity_flags": [],
            "confidence": 0.9,
        },
    },
    {
        "input": {
            "table_name": "products",
            "columns": "product_id (STRING), name (STRING), "
            "category (STRING), launched_on (DATE)",
        },
        "output": {
            "primary_entity": "Product",
            "grain_hypothesis": "one row per product",
            "secondary_entity_hints": ["product category"],
            "ambiguity_flags": [],
            "confidence": 0.9,
        },
    },
    {
        "input": {
            "table_name": "user_roles",
            "columns": "user_id (STRING), role_id (STRING), "
            "granted_at (TIMESTAMP)",
        },
        "output": {
            "primary_entity": "User-Role Assignment",
            "grain_hypothesis": "one row per (user, role) pair "
            "— bridge table",
            "secondary_entity_hints": ["user", "role"],
            "ambiguity_flags": [],
            "confidence": 0.85,
        },
    },
    {
        "input": {
            "table_name": "customer_profile",
            "columns": "customer_id (STRING), display_name (STRING), "
            "email (STRING), preferred_channel (STRING), "
            "segment (STRING), lifetime_value (DECIMAL)",
        },
        "output": {
            "primary_entity": "Customer",
            "grain_hypothesis": "one row per customer "
            "— wide attribute profile",
            "secondary_entity_hints": ["customer segment"],
            "ambiguity_flags": [],
            "confidence": 0.9,
        },
    },
]

# --------------------------------------------------------------------------
# Generic Stage B examples — column archetypes
# --------------------------------------------------------------------------

GENERIC_STAGE_B: list[dict[str, Any]] = [
    {
        "input": {
            "table_name": "users",
            "column": "user_id",
            "data_type": "STRING",
            "entity_context": "User",
        },
        "output": {
            "canonical_property_label": "user identifier",
            "semantic_type": "identifier",
            "synonyms": ["user id", "user key", "uid"],
            "candidate_vocab_families": [],
            "entity_role": "primary_key",
            "needs_stage_c": False,
        },
    },
    {
        "input": {
            "table_name": "orders",
            "column": "customer_id",
            "data_type": "STRING",
            "entity_context": "Order",
        },
        "output": {
            "canonical_property_label": "customer identifier",
            "semantic_type": "identifier",
            "synonyms": ["customer key", "customer ref"],
            "candidate_vocab_families": [],
            "entity_role": "foreign_key",
            "needs_stage_c": False,
        },
    },
    {
        "input": {
            "table_name": "events",
            "column": "occurred_at",
            "data_type": "TIMESTAMP",
            "entity_context": "Event",
        },
        "output": {
            "canonical_property_label": "event occurrence time",
            "semantic_type": "temporal",
            "synonyms": ["event time", "timestamp"],
            "candidate_vocab_families": [],
            "entity_role": "attribute",
            "needs_stage_c": False,
        },
    },
    {
        "input": {
            "table_name": "orders",
            "column": "total_amount",
            "data_type": "DECIMAL",
            "entity_context": "Order",
        },
        "output": {
            "canonical_property_label": "order total amount",
            "semantic_type": "numeric",
            "synonyms": ["total", "order total", "amount"],
            "candidate_vocab_families": ["monetary amount"],
            "entity_role": "attribute",
            "needs_stage_c": False,
        },
    },
    {
        "input": {
            "table_name": "orders",
            "column": "status",
            "data_type": "STRING",
            "top_values": "pending, active, completed, cancelled",
            "entity_context": "Order",
        },
        "output": {
            "canonical_property_label": "order status",
            "semantic_type": "categorical",
            "synonyms": ["state", "order state"],
            "candidate_vocab_families": ["order status codes"],
            "entity_role": "attribute",
            "needs_stage_c": True,
        },
    },
    {
        "input": {
            "table_name": "products",
            "column": "description",
            "data_type": "STRING",
            "entity_context": "Product",
        },
        "output": {
            "canonical_property_label": "product description",
            "semantic_type": "free_text",
            "synonyms": ["summary", "details"],
            "candidate_vocab_families": [],
            "entity_role": "attribute",
            "needs_stage_c": False,
        },
    },
    {
        "input": {
            "table_name": "users",
            "column": "is_active",
            "data_type": "BOOLEAN",
            "entity_context": "User",
        },
        "output": {
            "canonical_property_label": "active flag",
            "semantic_type": "boolean",
            "synonyms": ["active", "enabled"],
            "candidate_vocab_families": [],
            "entity_role": "attribute",
            "needs_stage_c": False,
        },
    },
    {
        "input": {
            "table_name": "tickets",
            "column": "priority",
            "data_type": "INT",
            "top_values": "1, 2, 3, 4, 5",
            "entity_context": "Ticket",
        },
        "output": {
            "canonical_property_label": "ticket priority",
            "semantic_type": "ordinal",
            "synonyms": ["priority level", "severity"],
            "candidate_vocab_families": ["priority ranking"],
            "entity_role": "attribute",
            "needs_stage_c": True,
        },
    },
]

# --------------------------------------------------------------------------
# Generic Stage C examples — value-decoding patterns
# --------------------------------------------------------------------------

GENERIC_STAGE_C: list[dict[str, Any]] = [
    {
        "input": {
            "table_name": "orders",
            "column": "status",
            "values": [
                "pending (25%)", "active (40%)",
                "completed (30%)", "cancelled (5%)",
            ],
        },
        "output": {
            "decoded_categories": [
                {"raw": "pending", "label": "order awaiting processing"},
                {"raw": "active", "label": "order in progress"},
                {"raw": "completed", "label": "order fulfilled"},
                {"raw": "cancelled", "label": "order cancelled"},
            ],
            "uncertainty": 0.05,
            "codebook_lookup_needed": False,
        },
    },
    {
        "input": {
            "table_name": "subscriptions",
            "column": "active_flag",
            "values": ["Y (70%)", "N (30%)"],
        },
        "output": {
            "decoded_categories": [
                {"raw": "Y", "label": "subscription is active"},
                {"raw": "N", "label": "subscription is inactive"},
            ],
            "uncertainty": 0.0,
            "codebook_lookup_needed": False,
        },
    },
    {
        "input": {
            "table_name": "jobs",
            "column": "outcome",
            "values": ["0:SUCCESS (85%)", "1:FAILED (15%)"],
        },
        "output": {
            "decoded_categories": [
                {"raw": "0:SUCCESS", "label": "job completed successfully"},
                {"raw": "1:FAILED", "label": "job failed"},
            ],
            "uncertainty": 0.0,
            "codebook_lookup_needed": False,
        },
    },
    {
        "input": {
            "table_name": "tickets",
            "column": "priority",
            "values": [
                "1 (10%)", "2 (20%)", "3 (40%)",
                "4 (20%)", "5 (10%)",
            ],
        },
        "output": {
            "decoded_categories": [
                {"raw": "1", "label": "highest priority"},
                {"raw": "2", "label": "high priority"},
                {"raw": "3", "label": "medium priority"},
                {"raw": "4", "label": "low priority"},
                {"raw": "5", "label": "lowest priority"},
            ],
            "uncertainty": 0.2,
            "codebook_lookup_needed": True,
        },
    },
]

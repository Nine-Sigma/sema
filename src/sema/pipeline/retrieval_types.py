"""TypedDict contracts for the retrieval pipeline.

Defines normalized shapes for seed hits (from vector/lexical search)
and expanded artifact candidates (from graph traversal). These provide
static type checking and IDE support without replacing the dict-based flow.
"""

from __future__ import annotations

from typing import Any, TypedDict


# ---------------------------------------------------------------------------
# Seed hit contracts — normalized shape for vector + lexical results
# ---------------------------------------------------------------------------


class SeedHit(TypedDict, total=False):
    """Common fields shared by all seed hits."""

    node_type: str
    match_type: str
    score: float
    confidence: float
    status: str

    # Identity fields — presence depends on node_type
    name: str
    code: str
    label: str
    vocabulary_name: str
    text: str
    target_name: str
    target_labels: list[str]

    # Scoped identity (when available in hit payload)
    datasource_id: str
    column_key: str
    table_key: str
    target_key: str

    # Entity-specific
    entity_name: str

    # Carried through from raw hit
    index: str
    final_score: float


# ---------------------------------------------------------------------------
# Expanded artifact contracts — shapes emitted by expansion functions
# ---------------------------------------------------------------------------


class EntityArtifact(TypedDict, total=False):
    """Expanded entity candidate."""

    type: str
    name: str
    catalog: str
    schema: str
    table: str
    description: str | None
    columns: list[dict[str, Any]]
    confidence: float
    source: str
    status: str
    confidence_policy: str


class PropertyArtifact(TypedDict, total=False):
    """Expanded property candidate."""

    type: str
    name: str
    entity_name: str
    physical_column: str
    physical_table: str
    semantic_type: str
    vocabulary: str | None
    score: float
    source: str
    status: str
    confidence: float
    confidence_policy: str


class ValueArtifact(TypedDict, total=False):
    """Expanded governed value candidate."""

    type: str
    property_name: str
    table: str
    column: str
    code: str
    label: str
    status: str
    confidence: float
    confidence_policy: str


class JoinArtifact(TypedDict, total=False):
    """Expanded join path candidate."""

    type: str
    from_table: str
    to_table: str
    join_predicates: Any
    sql_snippet: str | None
    hop_count: int
    cardinality_hint: str | None
    confidence: float
    source: str
    status: str
    confidence_policy: str


class MetricArtifact(TypedDict, total=False):
    """Expanded metric candidate."""

    type: str
    name: str
    description: str | None
    formula: str | None
    aggregates: list[str]
    filters: list[str]
    grains: list[str]
    confidence: float
    source: str
    status: str
    confidence_policy: str


class AncestryArtifact(TypedDict, total=False):
    """Expanded ancestry candidate."""

    type: str
    code: str
    label: str
    parent_code: str
    vocabulary: str | None
    source: str
    status: str
    confidence: float
    confidence_policy: str


class AliasArtifact(TypedDict, total=False):
    """Expanded alias candidate."""

    type: str
    text: str
    target: str
    target_type: str | None
    score: float
    source: str
    status: str
    confidence: float
    confidence_policy: str

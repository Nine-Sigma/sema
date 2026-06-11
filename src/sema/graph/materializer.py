"""Unified materializer: single assertion-to-graph path.

All helper functions live in materializer_utils.py.
This module exposes only the public entry point and constants.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING

from sema.models.assertions import Assertion
from sema.graph.materializer_utils import (
    apply_resolution_edges,
    upsert_column_nodes,
    upsert_physical_nodes,
    upsert_semantic_nodes,
)
from sema.graph.vocabulary_materializer import materialize_vocabulary_edges

if TYPE_CHECKING:
    from sema.graph.loader import GraphLoader

logger = logging.getLogger(__name__)

PROVENANCE_PREDICATES = frozenset({
    "has_entity_name", "has_property_name", "has_alias",
    "has_semantic_type", "has_decoded_value",
    "vocabulary_match", "parent_of", "has_join_evidence",
})


def materialize_unified(
    loader: GraphLoader,
    assertions: list[Assertion],
    source_schema: str | None = None,
) -> None:
    """Unified materializer: single assertion-to-graph path.

    Executes 4 ordered phases:
      1. Physical nodes (DataSource, Catalog, Schema, Table, Column)
      2. Semantic nodes (Entity, Property, Vocabulary, Term, ValueSet,
         Alias, JoinPath)
      3. Bridge edges (CLASSIFIED_AS, IN_VOCABULARY, PARENT_OF)
      4. Directed provenance (Assertion SUBJECT/OBJECT edges)

    Vocabulary lifecycle (deprecating stale vocabularies) is NOT run
    here: it runs once per build over the union of every table's active
    vocabularies (see sema.graph.lifecycle_utils), so a later table can
    never deprecate vocabularies an earlier table introduced.
    """
    by_subject: dict[str, list[Assertion]] = defaultdict(list)
    groups: dict[tuple[str, str], list[Assertion]] = defaultdict(list)
    for a in assertions:
        by_subject[a.subject_ref].append(a)
        groups[(a.subject_ref, a.predicate.value)].append(a)

    upsert_physical_nodes(loader, by_subject)
    upsert_column_nodes(loader, by_subject)
    upsert_semantic_nodes(
        loader, by_subject, groups, source_schema=source_schema,
    )
    apply_resolution_edges(
        loader, groups, source_schema=source_schema,
    )
    materialize_vocabulary_edges(loader, groups)
    loader.materialize_provenance_edges(assertions)

    logger.info(
        "Unified materialization complete: %d assertions processed",
        len(assertions),
    )

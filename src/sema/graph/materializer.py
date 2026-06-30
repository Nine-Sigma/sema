"""Unified materializer: single assertion-to-graph path.

All helper functions live in materializer_utils.py.
This module exposes only the public entry point and constants.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING

from sema.models.assertions import Assertion
from sema.models.physical_key import CanonicalRef
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

    ``source_schema`` scopes every relationship this writes so
    ``delete_study_scoped`` can later remove them. It is resolved BEFORE
    any write — explicit arg, else the single distinct schema across the
    assertions, else the single distinct schema parsed from their refs.
    If none can be resolved the call raises, rather than silently writing
    null-scoped edges that no study cleanup can ever delete.
    """
    if not assertions:
        return
    effective_schema = _resolve_source_schema(assertions, source_schema)

    by_subject: dict[str, list[Assertion]] = defaultdict(list)
    groups: dict[tuple[str, str], list[Assertion]] = defaultdict(list)
    for a in assertions:
        by_subject[a.subject_ref].append(a)
        groups[(a.subject_ref, a.predicate.value)].append(a)

    upsert_physical_nodes(loader, by_subject)
    upsert_column_nodes(loader, by_subject)
    upsert_semantic_nodes(
        loader, by_subject, groups, source_schema=effective_schema,
    )
    apply_resolution_edges(
        loader, groups, source_schema=effective_schema,
    )
    materialize_vocabulary_edges(
        loader, groups, source_schema=effective_schema,
    )
    loader.materialize_provenance_edges(assertions)

    logger.info(
        "Unified materialization complete: %d assertions processed",
        len(assertions),
    )


def _resolve_source_schema(
    assertions: list[Assertion], source_schema: str | None,
) -> str:
    """Resolve the study schema that scopes this materialization.

    Precedence: explicit arg -> single distinct ``Assertion.source_schema``
    -> single distinct schema parsed from assertion refs (covers the legacy
    ``materialize_table_graph`` delegate, whose assertions may not carry an
    explicit scope). Raises if the scope is ambiguous (mixed studies) or
    absent, so a study build can never emit undeletable null-scoped edges.
    """
    if source_schema is not None:
        return source_schema

    # Cross-check explicit assertion scopes AND the schemas parsed from refs:
    # a batch with one assertion stamped study_a plus an unstamped ref under
    # study_b must raise, not silently materialize everything as study_a.
    on_assertions = {a.source_schema for a in assertions if a.source_schema}
    combined = on_assertions | _schemas_from_refs(assertions)
    if len(combined) == 1:
        return combined.pop()
    if len(combined) > 1:
        raise ValueError(
            "materialize_unified: assertions span multiple source schemas "
            f"{sorted(combined)}; pass source_schema explicitly."
        )
    raise ValueError(
        "materialize_unified: cannot resolve source_schema (none on "
        "assertions, none parseable from refs, none passed). Study "
        "materialization must be scoped so delete_study_scoped can remove it."
    )


def _schemas_from_refs(assertions: list[Assertion]) -> set[str]:
    schemas: set[str] = set()
    for a in assertions:
        # Both endpoints matter: a join/FK assertion can carry the opposite
        # table/column in object_ref, in a different schema than subject_ref.
        for ref in (a.subject_ref, a.object_ref):
            if not ref:
                continue
            try:
                pk = CanonicalRef.parse(ref)
            except ValueError:
                continue
            if pk.schema:
                schemas.add(pk.schema)
    return schemas

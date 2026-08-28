"""Materialize a Column-anchored `:ValueSet` of concept `:Term`s (M3/M4).

Given a target column and the concept codes its source values resolve to, write
a value set in the exact source contract retrieval reads: a Column-anchored
``(:Column)-[:HAS_VALUE_SET]->(:ValueSet)`` (M3) whose members are canonical
``:Term {vocabulary_name, code, label}`` nodes (M4) — so they share the concept
spine with source terms and are found by lexical search — plus ``PARENT_OF``
ancestor context.

Generic: the ``:ValueSet`` primitive is domain-neutral; only *which* codes and
*how* they are resolved (the ``ConceptSource``) is target-specific, supplied by
the caller.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from sema.graph.concept_source import ConceptSource
from sema.graph.loader import GraphLoader

_SOURCE = "target-load"
_CONFIDENCE = 1.0


@dataclass(frozen=True)
class ValueSetColumn:
    """Physical location of the column a value set is anchored on."""

    catalog: str
    schema: str
    table: str
    column: str

    @property
    def ref(self) -> str:
        return f"{self.catalog}.{self.schema}.{self.table}.{self.column}"


def materialize_concept_value_set(
    loader: GraphLoader,
    *,
    column: ValueSetColumn,
    concept_vocabulary: str,
    concept_codes: Iterable[str],
    concepts: ConceptSource,
    source_schema: str,
    value_set_name: str | None = None,
    load_ancestors: bool = True,
) -> None:
    vs_name = value_set_name or f"{column.table}_{column.column}_values"
    loader.upsert_value_set(
        vs_name, column.column, column.table, column.schema, column.catalog,
        source_schema=source_schema, column_ref=column.ref,
    )
    for code in dict.fromkeys(concept_codes):
        _upsert_concept(
            loader, code, concept_vocabulary, concepts, source_schema
        )
        loader.add_term_to_value_set(
            code, vs_name, source_schema=source_schema,
            vocabulary_name=concept_vocabulary, value_set_ref=column.ref,
        )
        if load_ancestors:
            _upsert_ancestors(
                loader, code, concept_vocabulary, concepts, source_schema
            )


def _upsert_concept(
    loader: GraphLoader,
    code: str,
    vocabulary: str,
    concepts: ConceptSource,
    source_schema: str,
) -> None:
    detail = concepts.name_synonyms(code)
    label = detail.name if detail else code
    loader.upsert_term(
        code, label, source=_SOURCE, confidence=_CONFIDENCE,
        source_schema=source_schema, vocabulary_name=vocabulary,
    )
    if detail and detail.synonyms:
        _set_term_synonyms(loader, code, vocabulary, detail.synonyms)


def _upsert_ancestors(
    loader: GraphLoader,
    code: str,
    vocabulary: str,
    concepts: ConceptSource,
    source_schema: str,
) -> None:
    for ancestor in concepts.ancestors(code):
        loader.upsert_term(
            ancestor.code, ancestor.name, source=_SOURCE,
            confidence=_CONFIDENCE, source_schema=source_schema,
            vocabulary_name=vocabulary,
        )
        loader.add_term_hierarchy(
            parent_code=ancestor.code, child_code=code,
            source_schema=source_schema, vocabulary_name=vocabulary,
        )


def _set_term_synonyms(
    loader: GraphLoader,
    code: str,
    vocabulary: str,
    synonyms: tuple[str, ...],
) -> None:
    loader._run(
        "MATCH (t:Term {vocabulary_name: $vocabulary_name, code: $code}) "
        "SET t.synonyms = $synonyms",
        vocabulary_name=vocabulary, code=code, synonyms=list(synonyms),
    )


__all__ = ["ValueSetColumn", "materialize_concept_value_set"]

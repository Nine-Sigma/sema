"""SQLGlot AST builders and row shapes for the vocabulary query layer (US-003).

Every query is authored once as a SQLGlot expression and rendered per dialect
(DuckDB for dev, Databricks for prod) — no hand-concatenated SQL. The store is
vocabulary-agnostic: the physical table and column names arrive as a
:class:`VocabStoreSchema` (the OMOP instance lives behind the policy boundary in
:mod:`sema.resolve.policies.omop`), and the standardizing relationship name and
standard flag are bound as parameters by the caller (the US-004 policy), never
embedded here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import sqlglot.expressions as exp


@dataclass(frozen=True)
class VocabStoreSchema:
    """Physical table/column names for the concept-vocabulary tables.

    Field names are generic (role-based); the concrete OMOP values are supplied
    by the per-vocabulary policy module so no OMOP literal is hardcoded here.
    """

    concept_table: str
    relationship_table: str
    synonym_table: str
    id_col: str
    code_col: str
    name_col: str
    vocab_col: str
    domain_col: str
    standard_col: str
    invalid_col: str
    rel_from_col: str
    rel_to_col: str
    rel_id_col: str
    synonym_name_col: str


@dataclass(frozen=True)
class ConceptRow:
    """A concept row reduced to the fields the resolver needs."""

    id: str
    name: str
    domain: str | None
    vocabulary: str
    standard: str | None
    code: str
    invalid_reason: str | None


def concept_column_order(schema: VocabStoreSchema) -> tuple[str, ...]:
    """Fixed SELECT order matching :func:`row_to_concept` positions."""
    return (
        schema.id_col,
        schema.name_col,
        schema.domain_col,
        schema.vocab_col,
        schema.standard_col,
        schema.code_col,
        schema.invalid_col,
    )


def row_to_concept(row: Sequence[Any]) -> ConceptRow:
    """Map a positional result tuple (in :func:`concept_column_order`) to a row."""
    return ConceptRow(
        id=str(row[0]),
        name=str(row[1]),
        domain=None if row[2] is None else str(row[2]),
        vocabulary=str(row[3]),
        standard=None if row[4] is None else str(row[4]),
        code=str(row[5]),
        invalid_reason=None if row[6] is None else str(row[6]),
    )


def _table(namespace: str, name: str) -> exp.Table:
    return exp.to_table(f"{namespace}.{name}" if namespace else name)


def _concept_select(
    schema: VocabStoreSchema, namespace: str, alias: str | None
) -> exp.Select:
    cols = [exp.column(c, table=alias) for c in concept_column_order(schema)]
    return exp.select(*cols)


def concept_by_code_query(schema: VocabStoreSchema, namespace: str) -> exp.Select:
    """``SELECT <concept cols> WHERE vocab = ? AND code = ?``."""
    return (
        _concept_select(schema, namespace, None)
        .from_(_table(namespace, schema.concept_table))
        .where(exp.column(schema.vocab_col).eq(exp.Placeholder()))
        .where(exp.column(schema.code_col).eq(exp.Placeholder()))
    )


def maps_to_targets_query(
    schema: VocabStoreSchema,
    namespace: str,
    *,
    standard: bool = False,
    only_valid: bool = False,
) -> exp.Select:
    """Join the relationship table to its target concept; relationship bound as ?.

    ``standard`` adds ``standard_col = ?`` (the flag VALUE is the caller's, bound
    as a parameter); ``only_valid`` adds ``invalid_col IS NULL``.
    """
    rel = _table(namespace, schema.relationship_table).as_("r")
    concept = _table(namespace, schema.concept_table).as_("c")
    cols = [exp.column(c, table="c") for c in concept_column_order(schema)]
    query = (
        exp.select(*cols)
        .from_(rel)
        .join(
            concept,
            on=exp.column(schema.id_col, "c").eq(exp.column(schema.rel_to_col, "r")),
            join_type="inner",
        )
        .where(exp.column(schema.rel_from_col, "r").eq(exp.Placeholder()))
        .where(exp.column(schema.rel_id_col, "r").eq(exp.Placeholder()))
    )
    if standard:
        query = query.where(exp.column(schema.standard_col, "c").eq(exp.Placeholder()))
    if only_valid:
        query = query.where(exp.column(schema.invalid_col, "c").is_(exp.null()))
    return query


def concept_domain_query(schema: VocabStoreSchema, namespace: str) -> exp.Select:
    """``SELECT domain WHERE id = ?``."""
    return (
        exp.select(exp.column(schema.domain_col))
        .from_(_table(namespace, schema.concept_table))
        .where(exp.column(schema.id_col).eq(exp.Placeholder()))
    )

"""Term staging: vocabularies above the inline cap, in the vocabulary store shape.

The shape is the OMOP staging shape (``vocabulary_omop``) generalized to any
spec's external namespace, so ``VocabStore`` reads it unchanged. It lives in
``binding`` because the R29 guard does not scan it and the column names are
OMOP-shaped. The standard flag is written null: which value marks a standard
concept is a resolver policy choice.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date

import pyarrow as pa

from sema.binding.importer_terms_utils import TermRow
from sema.ingest.duckdb_staging import Staging
from sema.ingest.duckdb_staging_utils import quote_ident
from sema.resolve.vocab_store_utils import VocabStoreSchema

TERM_STAGING_SCHEMA = VocabStoreSchema(
    concept_table="concept",
    relationship_table="concept_relationship",
    synonym_table="concept_synonym",
    id_col="concept_id",
    code_col="concept_code",
    name_col="concept_name",
    vocab_col="vocabulary_id",
    domain_col="domain_id",
    standard_col="standard_concept",
    invalid_col="invalid_reason",
    rel_from_col="concept_id_1",
    rel_to_col="concept_id_2",
    rel_id_col="relationship_id",
    synonym_name_col="concept_synonym_name",
)
VALID_START_COL = "valid_start_date"
VALID_END_COL = "valid_end_date"
SYNONYM_LANG_COL = "language"
OPEN_START = date(1970, 1, 1)
OPEN_END = date(2099, 12, 31)
REPLACED_BY_RELATIONSHIP = "replaced_by"

_S = TERM_STAGING_SCHEMA
CONCEPT_TYPES = {
    _S.id_col: "VARCHAR",
    _S.name_col: "VARCHAR",
    _S.domain_col: "VARCHAR",
    _S.vocab_col: "VARCHAR",
    _S.standard_col: "VARCHAR",
    _S.code_col: "VARCHAR",
    VALID_START_COL: "DATE",
    VALID_END_COL: "DATE",
    _S.invalid_col: "VARCHAR",
}
RELATIONSHIP_TYPES = {
    _S.rel_from_col: "VARCHAR",
    _S.rel_to_col: "VARCHAR",
    _S.rel_id_col: "VARCHAR",
}
SYNONYM_TYPES = {_S.id_col: "VARCHAR", _S.synonym_name_col: "VARCHAR", SYNONYM_LANG_COL: "VARCHAR"}


def staged_concept_id(vocabulary: str, code: str) -> str:
    return f"{vocabulary}:{code}"


def staged_tables(terms: Iterable[TermRow]) -> tuple[pa.Table, pa.Table, pa.Table]:
    concepts: dict[str, list[object]] = {name: [] for name in CONCEPT_TYPES}
    relationships: dict[str, list[object]] = {name: [] for name in RELATIONSHIP_TYPES}
    synonyms: dict[str, list[object]] = {name: [] for name in SYNONYM_TYPES}
    for term in terms:
        concept_id = staged_concept_id(term.vocabulary, term.code)
        lifecycle = term.lifecycle
        _append(concepts, concept_id, term.display, None, term.vocabulary, None, term.code)
        concepts[VALID_START_COL].append(_date(lifecycle.valid_from, OPEN_START))
        concepts[VALID_END_COL].append(_date(lifecycle.valid_to, OPEN_END))
        concepts[_S.invalid_col].append(lifecycle.status if lifecycle.retired else None)
        if lifecycle.replaced_by:
            replacement = staged_concept_id(term.vocabulary, lifecycle.replaced_by)
            _append(relationships, concept_id, replacement, REPLACED_BY_RELATIONSHIP)
        for synonym in term.synonyms:
            _append(synonyms, concept_id, synonym, term.lang)
    return (
        _table(concepts, CONCEPT_TYPES),
        _table(relationships, RELATIONSHIP_TYPES),
        _table(synonyms, SYNONYM_TYPES),
    )


def write_term_staging(
    staging: Staging,
    namespace: str,
    terms: pa.Table,
    relationships: pa.Table,
    synonyms: pa.Table,
) -> None:
    """Replace the namespace's three tables in one transaction."""
    writes = (
        (_S.concept_table, terms, CONCEPT_TYPES),
        (_S.relationship_table, relationships, RELATIONSHIP_TYPES),
        (_S.synonym_table, synonyms, SYNONYM_TYPES),
    )
    staging.execute("BEGIN TRANSACTION")
    try:
        staging.execute(f"CREATE SCHEMA IF NOT EXISTS {quote_ident(namespace)}")
        for table, rows, types in writes:
            staging.write_table(namespace, table, rows, types, {}, None)
    except Exception:
        staging.execute("ROLLBACK")
        raise
    staging.execute("COMMIT")


def clear_term_staging(staging: Staging, namespace: str) -> None:
    """Drop the namespace's three tables: a run that stages nothing leaves none."""
    tables = (_S.concept_table, _S.relationship_table, _S.synonym_table)
    staging.execute("BEGIN TRANSACTION")
    try:
        for table in tables:
            staging.execute(
                f"DROP TABLE IF EXISTS {quote_ident(namespace)}.{quote_ident(table)}"
            )
    except Exception:
        staging.execute("ROLLBACK")
        raise
    staging.execute("COMMIT")


def _append(columns: dict[str, list[object]], *values: object) -> None:
    for name, value in zip(list(columns)[: len(values)], values, strict=True):
        columns[name].append(value)


def _date(text: str | None, default: date) -> date:
    if text is None:
        return default
    try:
        return date.fromisoformat(text[:10])
    except ValueError as exc:
        raise ValueError(f"lifecycle date {text!r} is not an ISO date") from exc


_ARROW = {"VARCHAR": pa.string(), "DATE": pa.date32()}


def _table(columns: dict[str, list[object]], types: dict[str, str]) -> pa.Table:
    return pa.table(
        {name: pa.array(values, _ARROW[types[name]]) for name, values in columns.items()}
    )

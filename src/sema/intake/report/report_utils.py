"""Helpers for the dry-run report: the terms an import would write, read back and counted."""

from __future__ import annotations

import random
import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pyarrow as pa

from sema.binding.importer import ConceptImport
from sema.binding.importer_utils import as_text, in_concept_scope, select_sql
from sema.binding.spec import ConceptSourceSpec
from sema.binding.spec_sections import CodeListRoles, RelationRef, ValueSetRoles
from sema.binding.term_staging import REPLACED_BY_RELATIONSHIP, TERM_STAGING_SCHEMA
from sema.ingest.duckdb_staging import Staging
from sema.ingest.duckdb_staging_utils import quote_ident
from sema.intake.calls.ledger import RunLedger
from sema.intake.land.profile_utils import TEXT_TYPE
from sema.intake.land.relations import RelationProvider
from sema.intake.land.table_schema import TableSchema
from sema.intake.verify.dry_run import TERM_KIND
from sema.models.target.refs import ExternalRef
from sema.targets.adapters.manifest_models import ParsedManifest

# The one role a value-set row carries beyond a code-list row: the concept it hangs off.
VALUE_SET_GROUP_ROLE = next(iter(set(ValueSetRoles.model_fields) - set(CodeListRoles.model_fields)))
EDGE_COLUMNS = ("child_key", "parent_key")
KEY_SEPARATOR = "\x1f"
NEAR_DUPLICATE_NOISE = re.compile(r"[\W_]+")
NO_PRIMARY_KEY = "no primary key stated"
NO_REQUIRED_FIELD = "no required field stated"

NullRows = dict[str, list[tuple[str | None, str | None]]]


@dataclass(frozen=True)
class TermEntry:
    vocabulary: str
    code: str
    display: str | None
    retired: bool
    replaced_by: str | None


@dataclass(frozen=True)
class EdgeSet:
    edges: list[tuple[str, str]]
    orphan_parents: int


def import_terms(parsed: ParsedManifest, imported: ConceptImport) -> list[TermEntry]:
    """Active terms from the manifest, retired inline terms from the sidecar retirements."""
    active = [TermEntry(t.vocabulary, t.code, t.display, False, None) for t in parsed.terms]
    retired = [
        TermEntry(r.vocabulary or "", r.code, None, True, r.replaced_by)
        for r in imported.authored.retirements
        if r.kind == TERM_KIND
    ]
    return active + retired


def staged_terms(staging: Staging, namespace: str) -> list[TermEntry]:
    schema = TERM_STAGING_SCHEMA
    if not _has_table(staging, namespace, schema.concept_table):
        return []
    quoted = quote_ident(namespace)
    rows = staging.execute(
        f"SELECT {schema.id_col}, {schema.vocab_col}, {schema.code_col}, {schema.name_col},"
        f" {schema.invalid_col} FROM {quoted}.{quote_ident(schema.concept_table)}"
    ).fetchall()
    replaced = dict(
        staging.execute(
            f"SELECT {schema.rel_from_col}, {schema.rel_to_col}"
            f" FROM {quoted}.{quote_ident(schema.relationship_table)}"
            f" WHERE {schema.rel_id_col} = ?",
            [REPLACED_BY_RELATIONSHIP],
        ).fetchall()
    )
    codes = {row[0]: row[2] for row in rows}
    return [
        TermEntry(vocabulary, code, name, invalid is not None, codes.get(replaced.get(key)))
        for key, vocabulary, code, name, invalid in rows
    ]


def _has_table(staging: Staging, namespace: str, table: str) -> bool:
    found = staging.execute(
        "SELECT COUNT(*) FROM information_schema.tables"
        " WHERE table_schema = ? AND table_name = ?",
        [namespace, table],
    ).fetchone()
    return bool(found and found[0])


def seeded_sample(terms: Sequence[TermEntry], size: int, seed: int) -> list[TermEntry]:
    ordered = sorted(terms, key=lambda term: term.code)
    picked = random.Random(seed).sample(ordered, min(size, len(ordered)))
    return sorted(picked, key=lambda term: term.code)


def longest_labels(terms: Sequence[TermEntry], size: int) -> list[TermEntry]:
    return sorted(terms, key=lambda term: (-len(term.display or ""), term.code))[:size]


def near_duplicates(terms: Sequence[TermEntry]) -> list[TermEntry]:
    """Distinct codes that coincide once case, spacing, and punctuation are removed."""
    groups: dict[str, set[str]] = defaultdict(set)
    for term in terms:
        groups[NEAR_DUPLICATE_NOISE.sub("", term.code.casefold())].add(term.code)
    near = {code for codes in groups.values() if len(codes) > 1 for code in codes}
    return sorted((term for term in terms if term.code in near), key=lambda term: term.code)


def duplicate_codes(terms: Sequence[TermEntry]) -> list[str]:
    counts = Counter(term.code for term in terms if not term.retired)
    return sorted(code for code, count in counts.items() if count > 1)


def lifecycle_counts(terms: Sequence[TermEntry]) -> tuple[int, int, int]:
    """Active, retired, and retired with an active replacement in its own vocabulary."""
    active = {(term.vocabulary, term.code) for term in terms if not term.retired}
    retired = [term for term in terms if term.retired]
    replaced = sum((term.vocabulary, term.replaced_by) in active for term in retired)
    return len(active), len(retired), replaced


def hierarchy_edges(parsed: ParsedManifest) -> EdgeSet:
    hierarchical = [r for r in parsed.relationships if r.hierarchical]
    edges = sorted({(_key(r.from_), _key(r.to)) for r in hierarchical})
    orphans = {_key(r.to) for r in hierarchical if not r.far_end_loaded}
    return EdgeSet(edges=edges, orphan_parents=len(orphans))


def _key(ref: ExternalRef) -> str:
    return f"{ref.system}{KEY_SEPARATOR}{ref.id}"


def write_edges(staging: Staging, namespace: str, table: str, edges: EdgeSet) -> None:
    child, parent = EDGE_COLUMNS
    rows = pa.table(
        {child: [e[0] for e in edges.edges], parent: [e[1] for e in edges.edges]},
        schema=pa.schema([(child, pa.string()), (parent, pa.string())]),
    )
    staging.execute(f"CREATE SCHEMA IF NOT EXISTS {quote_ident(namespace)}")
    staging.write_table(namespace, table, rows, {child: "VARCHAR", parent: "VARCHAR"}, {}, None)


def external_ref_codes(parsed: ParsedManifest) -> dict[str, list[str]]:
    refs = [ref for term in parsed.terms for ref in term.external_refs]
    refs += [ref for e in parsed.entities for p in e.properties for ref in p.external_refs]
    codes: dict[str, set[str]] = defaultdict(set)
    for ref in refs:
        codes[ref.system].add(ref.id)
    return {system: sorted(found) for system, found in sorted(codes.items())}


def concept_count(parsed: ParsedManifest) -> int:
    return sum(p.concept_ref is not None for e in parsed.entities for p in e.properties)


def entity_counts(schema: TableSchema) -> tuple[int, int, int, int]:
    """Properties, declared foreign keys, domain bindings, and inline value sets."""
    fields = schema.fields
    return (
        len(fields),
        len(schema.foreign_keys or []),
        sum(field.x_sema_domain is not None for field in fields),
        sum(bool(field.x_sema_values) for field in fields),
    )


def without_obligation(
    parsed: ParsedManifest, schemas: dict[str, TableSchema]
) -> dict[str, str]:
    """Every reason an entity carries no obligation, not only the first one checked."""
    obligated = {e.qualified_name for e in parsed.entities if e.obligation is not None}
    reasons: dict[str, str] = {}
    for name, schema in sorted(schemas.items()):
        if name in obligated:
            continue
        found = [NO_PRIMARY_KEY] if not schema.primary_key else []
        found += [] if any(field.required for field in schema.fields) else [NO_REQUIRED_FIELD]
        reasons[name] = "; ".join(found)
    return reasons


def vocabulary_of_concepts(parsed: ParsedManifest) -> dict[str, str]:
    return {
        p.concept_ref.id: p.vocabulary_binding.vocabulary
        for e in parsed.entities
        for p in e.properties
        if p.concept_ref is not None and p.vocabulary_binding is not None
    }


def null_required_rows(
    spec: ConceptSourceSpec,
    provider: RelationProvider,
    ledger: RunLedger,
    stage: str,
    vocabulary_of: dict[str, str],
) -> NullRows:
    """Term rows with a null in a role the spec models require, per vocabulary."""
    found: NullRows = defaultdict(list)
    if spec.value_sets is not None:
        value_sets = spec.value_sets
        group = getattr(value_sets.roles, VALUE_SET_GROUP_ROLE)
        scope = in_concept_scope(provider, value_sets.relation, group, spec)
        query = _NullQuery(provider, ledger, stage, value_sets.relation, value_sets.roles)
        for row in query.rows(scope):
            vocabulary = vocabulary_of.get(as_text(row[group]) or "")
            if vocabulary is not None:
                found[vocabulary].append(query.rendered(row))
    for entry in spec.code_lists or ():
        query = _NullQuery(provider, ledger, stage, entry.relation, entry.roles)
        found[entry.vocabulary_id] += [query.rendered(row) for row in query.rows(entry.filter)]
    return found


@dataclass(frozen=True)
class _NullQuery:
    provider: RelationProvider
    ledger: RunLedger
    stage: str
    relation: RelationRef
    roles: CodeListRoles

    def rows(self, where: str | None) -> list[dict[str, Any]]:
        model_fields = type(self.roles).model_fields
        columns = [getattr(self.roles, n) for n, f in model_fields.items() if f.is_required()]
        blank = " OR ".join(
            f"COALESCE(TRIM(CAST({self.provider.quote(self.relation, c)} AS {TEXT_TYPE})), '')"
            " = ''"
            for c in columns
        )
        scoped = f"({where}) AND ({blank})" if where else blank
        statement = select_sql(self.provider, self.relation, columns, scoped)
        self.ledger.sql(self.stage, self.relation.provider, statement, True, False)
        found = self.provider.query(self.relation)(statement)
        return sorted(
            (dict(zip(columns, row, strict=True)) for row in found),
            key=lambda row: [str(value) for value in row.values()],
        )

    def rendered(self, row: dict[str, Any]) -> tuple[str | None, str | None]:
        return as_text(row[self.roles.code]), as_text(row[self.roles.display])

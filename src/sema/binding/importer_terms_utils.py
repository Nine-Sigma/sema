"""Value-set and code-list terms: one ``TermRow`` per code, inline or staged."""

from __future__ import annotations

from dataclasses import dataclass, field

from sema.binding.importer_utils import (
    Details,
    Record,
    RefKey,
    fetch,
    first_text,
    group_by_key,
    matched_details,
    refs_payload,
    row_external_refs,
    synonyms_of,
)
from sema.binding.lifecycle_utils import (
    Lifecycle,
    lifecycle_columns,
    lifecycle_of,
    retirement_row,
)
from sema.binding.models import Retirement
from sema.binding.spec import (
    CodeListRoles,
    CodeListSpec,
    ConceptSourceSpec,
    ExternalRefRole,
)
from sema.ingest.duckdb_staging import Staging
from sema.intake.relations import RelationProvider


@dataclass(frozen=True)
class TermRow:
    vocabulary: str
    code: str
    display: str
    definition: str | None
    synonyms: tuple[str, ...]
    external_refs: frozenset[RefKey]
    lang: str | None
    lifecycle: Lifecycle

    def manifest_record(self) -> Record:
        return {
            "vocabulary": self.vocabulary,
            "code": self.code,
            "display": self.display,
            "definition": self.definition,
            "synonyms": list(self.synonyms),
            "external_refs": refs_payload(self.external_refs),
        }


@dataclass
class Collected:
    properties: dict[str, Record] = field(default_factory=dict)
    vocabularies: list[Record] = field(default_factory=list)
    terms: list[Record] = field(default_factory=list)
    staged: list[TermRow] = field(default_factory=list)
    retirements: list[Retirement] = field(default_factory=list)


def term_columns(roles: CodeListRoles) -> list[str | None]:
    return [
        roles.code,
        roles.display,
        roles.display_fallback,
        roles.vocabulary_id,
        roles.concept_code,
        roles.synonyms,
        *lifecycle_columns(roles),
    ]


def collect_terms(
    spec: ConceptSourceSpec,
    roles: CodeListRoles,
    relation: str,
    vocabulary: str,
    group: list[Record],
    details: dict[str, Details],
) -> list[TermRow]:
    own_refs = value_row_ref_roles(roles)
    # Code-list rows never carry the term_details join column, and pass no details.
    join_column = spec.term_details.on[0] if spec.term_details and details else None
    terms: list[TermRow] = []
    for code, rows in sorted(group_by_key(group, roles.code).items()):
        found = matched_details(rows, join_column, details)
        display = first_text(rows, roles.display) or first_text(
            rows, roles.display_fallback
        )
        display = display or code
        refs = {ref for d in found for ref in d.external_refs}
        refs |= {ref for row in rows for ref in row_external_refs(row, own_refs)}
        terms.append(
            TermRow(
                vocabulary=vocabulary,
                code=code,
                display=display,
                definition=next((d.definition for d in found if d.definition), None),
                synonyms=tuple(sorted(synonyms_of(rows, roles.synonyms, []) - {display})),
                external_refs=frozenset(refs),
                lang=first_text(rows, roles.lang),
                lifecycle=lifecycle_of(rows, roles, relation),
            )
        )
    return terms


def value_row_ref_roles(roles: CodeListRoles) -> list[ExternalRefRole]:
    if roles.vocabulary_id is None or roles.concept_code is None:
        return []
    return [
        ExternalRefRole(system=f"{{{roles.vocabulary_id}}}", column=roles.concept_code)
    ]



def store_terms(
    spec: ConceptSourceSpec,
    collected: Collected,
    vocabulary: str,
    terms: list[TermRow],
    staging: Staging | None,
) -> None:
    if _above_cap(spec, vocabulary, terms, staging):
        collected.vocabularies.append({"name": vocabulary, "source": "EXTERNAL"})
        collected.staged.extend(terms)
        return
    collected.vocabularies.append({"name": vocabulary, "source": "INLINE"})
    for term in terms:
        if term.lifecycle.retired:
            collected.retirements.append(
                retirement_row("term", vocabulary, term.code, term.lifecycle)
            )
        else:
            collected.terms.append(term.manifest_record())


def code_lists(
    spec: ConceptSourceSpec,
    provider: RelationProvider,
    collected: Collected,
    staging: Staging | None,
) -> None:
    """One vocabulary per entry named by its ``vocabulary_id``, with no concept relation."""
    for entry in spec.code_lists or []:
        rows = fetch(provider, entry.relation, term_columns(entry.roles), entry.filter)
        vocabulary = entry.vocabulary_id
        terms = collect_terms(spec, entry.roles, entry.relation.name, vocabulary, rows, {})
        store_terms(spec, collected, vocabulary, terms, staging)
        if entry.concept is not None:
            collected.properties[entry.concept.concept_id] = _code_list_property(
                spec, entry
            )


def _code_list_property(spec: ConceptSourceSpec, entry: CodeListSpec) -> Record:
    concept = entry.concept
    if concept is None:
        raise ValueError(f"code list {entry.vocabulary_id} declares no concept")
    return {
        "name": concept.name,
        "type": "string",
        "synonyms": [],
        "concept_ref": {
            "identity_system": spec.identity_system,
            "id": concept.concept_id,
            "version": spec.release,
        },
        "definition": concept.definition,
        "external_refs": [],
        "vocabulary_binding": {"vocabulary": entry.vocabulary_id},
    }


def _above_cap(
    spec: ConceptSourceSpec, vocabulary: str, terms: list[TermRow], staging: Staging | None
) -> bool:
    cap = spec.term_storage.inline_cap
    if spec.spec_version < 2 or len(terms) <= cap:
        return False
    if staging is None:
        raise ValueError(
            f"vocabulary {vocabulary} has {len(terms)} terms, above inline_cap {cap}; "
            "term staging needs a staging database (--staging)"
        )
    return True

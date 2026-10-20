"""Value-set terms: one ``TermRow`` per code, carrying lifecycle and language."""

from __future__ import annotations

from dataclasses import dataclass

from sema.binding.importer_utils import (
    Details,
    Record,
    RefKey,
    first_text,
    group_by_key,
    matched_details,
    refs_payload,
    row_external_refs,
    synonyms_of,
)
from sema.binding.lifecycle_utils import Lifecycle, lifecycle_of
from sema.binding.spec import ConceptSourceSpec, ExternalRefRole, ValueSetRoles


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


def collect_terms(
    spec: ConceptSourceSpec,
    roles: ValueSetRoles,
    relation: str,
    vocabulary: str,
    group: list[Record],
    details: dict[str, Details],
) -> list[TermRow]:
    own_refs = value_row_ref_roles(roles)
    join_column = spec.term_details.on[0] if spec.term_details else None
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


def value_row_ref_roles(roles: ValueSetRoles) -> list[ExternalRefRole]:
    if roles.vocabulary_id is None or roles.concept_code is None:
        return []
    return [
        ExternalRefRole(system=f"{{{roles.vocabulary_id}}}", column=roles.concept_code)
    ]


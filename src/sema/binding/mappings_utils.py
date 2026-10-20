"""SSSOM-shaped mapping rows: read, split CURIEs, project exact matches to refs.

Mappings are joined to loaded concepts and terms in memory by subject CURIE,
never by a scope subquery, so the mappings relation may live on any provider.
"""

from __future__ import annotations

from pydantic import BaseModel

from sema.binding.importer_utils import Record, RefKey, as_text, fetch, refs_payload
from sema.binding.spec_sections import MappingsSpec
from sema.intake.relations import RelationProvider
from sema.log import logger


class SssomMapping(BaseModel):
    subject_id: str
    predicate_id: str
    object_id: str
    mapping_justification: str | None
    subject_label: str | None = None
    object_label: str | None = None
    confidence: float | None = None


def split_curie(curie: str) -> RefKey | None:
    prefix, sep, local = curie.partition(":")
    return (prefix, local) if sep and prefix and local else None


def subject_index(identity_system: str, loaded_ids: set[str]) -> dict[str, str]:
    """Subject CURIE to loaded id: ``<system>:<id>``, or the id when it is a CURIE."""
    index = {f"{identity_system}:{loaded_id}": loaded_id for loaded_id in loaded_ids}
    index.update({i: i for i in loaded_ids if split_curie(i) is not None})
    return index


def read_mappings(
    mappings: MappingsSpec, provider: RelationProvider, subjects: dict[str, str]
) -> list[SssomMapping]:
    """Rows whose subject is a loaded concept or term (see ``subject_index``)."""
    roles = mappings.roles
    columns = [
        roles.subject_id,
        roles.predicate_id,
        roles.object_id,
        roles.mapping_justification,
        roles.subject_label,
        roles.object_label,
        roles.confidence,
    ]
    rows = fetch(provider, mappings.relation, columns, mappings.filter)
    kept = [
        mapping
        for row in rows
        if (mapping := _mapping(row, mappings)) is not None
        and mapping.subject_id in subjects
    ]
    if len(kept) < len(rows):
        logger.info(
            "Mappings {}: kept {} of {} rows; the rest name no loaded subject",
            mappings.relation.name,
            len(kept),
            len(rows),
        )
    return kept


def exact_refs(
    mappings: list[SssomMapping], exact_predicates: list[str], subjects: dict[str, str]
) -> dict[str, set[RefKey]]:
    """Object refs per loaded id, for exact-match predicates only."""
    refs: dict[str, set[RefKey]] = {}
    for mapping in mappings:
        target = split_curie(mapping.object_id)
        if mapping.predicate_id in exact_predicates and target:
            refs.setdefault(subjects[mapping.subject_id], set()).add(target)
    return refs


def add_refs(record: Record, extra: set[RefKey]) -> None:
    current = {(ref["system"], ref["id"]) for ref in record["external_refs"]}
    record["external_refs"] = refs_payload(current | extra)


def _mapping(row: Record, mappings: MappingsSpec) -> SssomMapping | None:
    roles = mappings.roles
    subject, predicate, target = (
        as_text(row[roles.subject_id]),
        as_text(row[roles.predicate_id]),
        as_text(row[roles.object_id]),
    )
    if subject is None or predicate is None or target is None:
        return None
    return SssomMapping(
        subject_id=subject,
        predicate_id=predicate,
        object_id=target,
        mapping_justification=as_text(row[roles.mapping_justification]),
        subject_label=_optional(row, roles.subject_label),
        object_label=_optional(row, roles.object_label),
        confidence=_confidence(row, roles.confidence, mappings.relation.name),
    )


def _optional(row: Record, column: str | None) -> str | None:
    return as_text(row[column]) if column else None


def _confidence(row: Record, column: str | None, relation: str) -> float | None:
    text = _optional(row, column)
    if text is None:
        return None
    try:
        return float(text)
    except ValueError as exc:
        raise ValueError(
            f"relation {relation} column {column} has confidence {text!r}, "
            "which is not a number"
        ) from exc

"""Authored-evidence sidecar rows: a library's own column bindings and value mappings."""

from __future__ import annotations

from sema.binding.importer_utils import as_text, expand_list, fetch, in_concept_scope
from sema.binding.models import AuthoredColumnBinding, AuthoredValueMapping
from sema.binding.spec import (
    AuthoredBindingsSpec,
    ConceptSourceSpec,
    ValueMappingsSpec,
)
from sema.intake.relations import RelationProvider


def column_bindings(
    spec: ConceptSourceSpec,
    bindings_spec: AuthoredBindingsSpec,
    provider: RelationProvider,
    dataset_map: dict[str, str],
) -> list[AuthoredColumnBinding]:
    roles = bindings_spec.roles
    rows = fetch(
        provider,
        bindings_spec.relation,
        [
            roles.concept_id,
            roles.dataset_label,
            roles.dataset_id,
            roles.columns,
            roles.project,
        ],
        in_concept_scope(provider, bindings_spec.relation, roles.concept_id, spec),
    )
    bindings: list[AuthoredColumnBinding] = []
    for row in rows:
        concept_id = as_text(row[roles.concept_id])
        if concept_id is None:
            continue
        label, dataset_id = (
            as_text(row[roles.dataset_label]),
            as_text(row[roles.dataset_id]),
        )
        physical = _resolve_dataset(dataset_map, label, dataset_id)
        bindings.append(
            AuthoredColumnBinding(
                concept_ref=f"{spec.identity_system}:{concept_id}",
                dataset_label=label,
                dataset_id=dataset_id,
                physical=physical,
                columns=expand_list(row[roles.columns]),
                resolved=physical is not None,
                project=as_text(row[roles.project]) if roles.project else None,
            )
        )
    return bindings


def _resolve_dataset(
    dataset_map: dict[str, str], label: str | None, dataset_id: str | None
) -> str | None:
    for key in (label, dataset_id):
        if key is not None and key in dataset_map:
            return dataset_map[key]
    return None


def value_mappings(
    spec: ConceptSourceSpec, mappings_spec: ValueMappingsSpec, provider: RelationProvider
) -> list[AuthoredValueMapping]:
    roles = mappings_spec.roles
    rows = fetch(
        provider,
        mappings_spec.relation,
        [roles.concept_id, roles.source_value, roles.code, roles.status],
        in_concept_scope(provider, mappings_spec.relation, roles.concept_id, spec),
    )
    approved_statuses = set(mappings_spec.approved_statuses)
    mappings: list[AuthoredValueMapping] = []
    for row in rows:
        concept_id = as_text(row[roles.concept_id])
        if concept_id is None:
            continue
        status = as_text(row[roles.status])
        mappings.append(
            AuthoredValueMapping(
                concept_ref=f"{spec.identity_system}:{concept_id}",
                source_value=as_text(row[roles.source_value]),
                codes=expand_list(row[roles.code]),
                status=status,
                approved=status in approved_statuses,
            )
        )
    return mappings

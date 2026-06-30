"""Helpers for the target-model normalizer.

Cross-reference resolution, endpoint-property synthesis, stable ordering.
"""

from __future__ import annotations

from sema.models.planner._enums import TargetArtifactKind
from sema.models.target.context_card import TargetContextCard
from sema.models.target.entity import TargetEntityDecl
from sema.models.target.obligation import TargetObligationDecl
from sema.models.target.properties import PropertyKind, TargetPropertyDecl
from sema.models.target.refs import VocabularySource
from sema.models.target.term import TargetTermDecl
from sema.models.target.vocab_binding import VocabularyBindingDecl
from sema.targets.exceptions import DanglingRefError


def synthesize_endpoint_properties(entity: TargetEntityDecl) -> list[TargetPropertyDecl]:
    if entity.endpoints is None:
        return []
    return [
        _build_endpoint_property(entity.endpoints.subject),
        _build_endpoint_property(entity.endpoints.object),
    ]


def _build_endpoint_property(endpoint: object) -> TargetPropertyDecl:
    role = endpoint.role  # type: ignore[attr-defined]
    return TargetPropertyDecl(
        name=role,
        type="endpoint",
        nullable=endpoint.nullable,  # type: ignore[attr-defined]
        property_kind=PropertyKind.ENDPOINT,
        endpoint_role=role,
        endpoint_target_entity_qualified_name=endpoint.target_entity.qualified_name,  # type: ignore[attr-defined]
        endpoint_cardinality=endpoint.cardinality,  # type: ignore[attr-defined]
        endpoint_nullable=endpoint.nullable,  # type: ignore[attr-defined]
        materialized_as_edge_property=False,
    )


def normalize_entity(entity: TargetEntityDecl) -> TargetEntityDecl:
    synthesized = synthesize_endpoint_properties(entity)
    if not synthesized:
        sorted_props = sorted(entity.properties, key=lambda p: p.name)
        if list(entity.properties) == sorted_props:
            return entity
        return entity.model_copy(update={"properties": sorted_props})
    combined = sorted(list(entity.properties) + synthesized, key=lambda p: p.name)
    return entity.model_copy(update={"properties": combined})


def sort_entities(entities: list[TargetEntityDecl]) -> list[TargetEntityDecl]:
    return sorted(entities, key=lambda e: (e.ref.target_model_id, e.ref.qualified_name))


def sort_obligations(obligations: list[TargetObligationDecl]) -> list[TargetObligationDecl]:
    return sorted(obligations, key=lambda o: o.target_entity)


def sort_bindings(
    bindings: list[VocabularyBindingDecl],
) -> list[VocabularyBindingDecl]:
    return sorted(
        bindings,
        key=lambda b: (b.entity_ref.qualified_name, b.property_name, b.vocabulary.name),
    )


def sort_terms(terms: list[TargetTermDecl]) -> list[TargetTermDecl]:
    return sorted(terms, key=lambda t: (t.vocabulary.name, t.code))


def sort_context_cards(cards: list[TargetContextCard]) -> list[TargetContextCard]:
    return sorted(cards, key=lambda c: c.entity_ref.qualified_name)


def validate_obligation_required_fields(
    obligation: TargetObligationDecl,
    properties_by_entity: dict[str, list[TargetPropertyDecl]],
) -> None:
    props = properties_by_entity.get(obligation.target_entity)
    if props is None:
        raise DanglingRefError(
            f"TargetObligationDecl(target_entity={obligation.target_entity!r}) "
            f"references no entity declared in the loaded model"
        )
    names = {p.name for p in props}
    missing = [field for field in obligation.required_fields if field not in names]
    if missing:
        raise DanglingRefError(
            f"TargetObligationDecl(target_entity={obligation.target_entity!r}) "
            f"required_fields {missing} not present on entity"
        )


def validate_foreign_keys(
    obligation: TargetObligationDecl,
    entities_by_qualified_name: dict[str, TargetEntityDecl],
) -> None:
    for fk in obligation.foreign_keys:
        if fk.referenced_entity not in entities_by_qualified_name:
            raise DanglingRefError(
                f"TargetObligationDecl(target_entity={obligation.target_entity!r}) "
                f"foreign_key references unknown entity {fk.referenced_entity!r}"
            )


def validate_endpoint_targets(
    entity: TargetEntityDecl,
    entities_by_qualified_name: dict[str, TargetEntityDecl],
) -> None:
    if entity.endpoints is None:
        return
    for role, endpoint in (("subject", entity.endpoints.subject), ("object", entity.endpoints.object)):
        target = entities_by_qualified_name.get(endpoint.target_entity.qualified_name)
        if target is None:
            raise DanglingRefError(
                f"GRAPH_EDGE {entity.ref.qualified_name} endpoint {role!r} targets "
                f"missing entity {endpoint.target_entity.qualified_name!r}"
            )
        if target.ref.kind is TargetArtifactKind.TABLE_ROW:
            raise DanglingRefError(
                f"GRAPH_EDGE {entity.ref.qualified_name} endpoint {role!r} targets "
                f"TABLE_ROW entity {endpoint.target_entity.qualified_name!r}; "
                f"endpoints MUST reference GRAPH_NODE or GRAPH_EDGE entities"
            )


def validate_vocabulary_bindings(
    bindings: list[VocabularyBindingDecl],
    terms: list[TargetTermDecl],
) -> None:
    inline_term_names = {
        t.vocabulary.name for t in terms if t.vocabulary.source is VocabularySource.INLINE
    }
    for binding in bindings:
        vocab = binding.vocabulary
        if vocab.source is VocabularySource.EXTERNAL:
            continue
        if vocab.name not in inline_term_names:
            raise DanglingRefError(
                f"VocabularyBindingDecl({binding.entity_ref.qualified_name}.{binding.property_name}) "
                f"references INLINE vocabulary {vocab.name!r} but no inline terms found; "
                f"declare source=EXTERNAL or supply matching TargetTermDecl"
            )


__all__ = [
    "synthesize_endpoint_properties",
    "normalize_entity",
    "sort_entities",
    "sort_obligations",
    "sort_bindings",
    "sort_terms",
    "sort_context_cards",
    "validate_obligation_required_fields",
    "validate_foreign_keys",
    "validate_endpoint_targets",
    "validate_vocabulary_bindings",
]

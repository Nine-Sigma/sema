"""Cypher template helpers for `Neo4jGraphWriter`.

Centralises the MERGE/SET shapes so the writer stays focused on
session orchestration. None of these helpers import `sema.graph`;
they consume only DTOs and emit Cypher strings.
"""

from __future__ import annotations

from typing import Any

from sema.models.planner._enums import ModelRole
from sema.targets.materializer_ops import (
    ConstraintOp,
    ContextCardOp,
    EnrichmentDecisionOp,
    EntityOp,
    PropertyOp,
    RelationshipOp,
    TargetObligationOp,
    TermOp,
    VocabularyBindingOp,
)


_TARGET_ROLE = ModelRole.TARGET.value


def entity_merge(op: EntityOp) -> tuple[str, dict[str, Any]]:
    cypher = (
        "MERGE (n:Entity {"
        "target_model_id: $target_model_id, "
        "target_model_version: $target_model_version, "
        "target_schema_snapshot_hash: $target_schema_snapshot_hash, "
        "qualified_name: $qualified_name"
        "}) "
        "SET n.kind = $kind, n.is_current = $is_current, "
        "n.model_role = $model_role, "
        "n.id = $id, "
        f"{_enrichment_status_assign()}"
    )
    params = {
        "target_model_id": op.target_model_id,
        "target_model_version": op.target_model_version,
        "target_schema_snapshot_hash": op.target_schema_snapshot_hash,
        "qualified_name": op.qualified_name,
        "kind": op.kind,
        "is_current": op.is_current,
        "model_role": _TARGET_ROLE,
        "id": _entity_id(op),
        **_enrichment_status_params(op.enrichment_status),
    }
    return cypher, params


def _enrichment_status_assign() -> str:
    fields = (
        "structure",
        "obligations",
        "vocabulary_bindings",
        "semantic_aliases",
        "terms",
    )
    return ", ".join(
        f"n.enrichment_{f}_status = $enrichment_{f}_status" for f in fields
    )


def _enrichment_status_params(status: dict[str, str]) -> dict[str, Any]:
    fields = (
        "structure",
        "obligations",
        "vocabulary_bindings",
        "semantic_aliases",
        "terms",
    )
    return {f"enrichment_{f}_status": status.get(f) for f in fields}


def _entity_id(op: EntityOp) -> str:
    return (
        f"{op.target_model_id}|{op.target_model_version}|"
        f"{op.target_schema_snapshot_hash}|{op.qualified_name}"
    )


def property_merge(op: PropertyOp) -> tuple[str, dict[str, Any]]:
    cypher = (
        "MERGE (n:Property {"
        "target_model_id: $target_model_id, "
        "target_model_version: $target_model_version, "
        "target_schema_snapshot_hash: $target_schema_snapshot_hash, "
        "parent_entity_qualified_name: $parent_entity_qualified_name, "
        "name: $name"
        "}) "
        "SET n.type = $type, n.nullable = $nullable, "
        "n.synonyms = $synonyms, n.decoded_values_json = $decoded_values_json, "
        "n.property_kind = $property_kind, "
        "n.endpoint_role = $endpoint_role, "
        "n.endpoint_target_entity_qualified_name = "
        "$endpoint_target_entity_qualified_name, "
        "n.endpoint_cardinality = $endpoint_cardinality, "
        "n.endpoint_nullable = $endpoint_nullable, "
        "n.materialized_as_edge_property = $materialized_as_edge_property, "
        "n.is_current = $is_current, "
        "n.model_role = $model_role, n.id = $id"
    )
    params = {
        "target_model_id": op.target_model_id,
        "target_model_version": op.target_model_version,
        "target_schema_snapshot_hash": op.target_schema_snapshot_hash,
        "parent_entity_qualified_name": op.parent_entity_qualified_name,
        "name": op.name,
        "type": op.type,
        "nullable": op.nullable,
        "synonyms": list(op.synonyms),
        "decoded_values_json": _json_dumps(op.decoded_values),
        "property_kind": op.property_kind,
        "endpoint_role": op.endpoint_role,
        "endpoint_target_entity_qualified_name": op.endpoint_target_entity_qualified_name,
        "endpoint_cardinality": op.endpoint_cardinality,
        "endpoint_nullable": op.endpoint_nullable,
        "materialized_as_edge_property": op.materialized_as_edge_property,
        "is_current": op.is_current,
        "model_role": _TARGET_ROLE,
        "id": _property_id(op),
    }
    return cypher, params


def _property_id(op: PropertyOp) -> str:
    return (
        f"{op.target_model_id}|{op.target_model_version}|"
        f"{op.target_schema_snapshot_hash}|"
        f"{op.parent_entity_qualified_name}.{op.name}"
    )


def _stringify(payload: dict[str, Any]) -> dict[str, str]:
    return {str(k): str(v) for k, v in payload.items()}


def _json_dumps(payload: dict[str, Any]) -> str:
    import json as _json

    return _json.dumps(payload, sort_keys=True)


def term_merge(op: TermOp) -> tuple[str, dict[str, Any]]:
    cypher = (
        "MERGE (n:Term {"
        "target_model_id: $target_model_id, "
        "target_model_version: $target_model_version, "
        "target_schema_snapshot_hash: $target_schema_snapshot_hash, "
        "vocabulary_name: $vocabulary_name, "
        "code: $code"
        "}) "
        "SET n.display = $display, n.is_current = $is_current, "
        "n.model_role = $model_role, n.id = $id"
    )
    return cypher, {
        "target_model_id": op.target_model_id,
        "target_model_version": op.target_model_version,
        "target_schema_snapshot_hash": op.target_schema_snapshot_hash,
        "vocabulary_name": op.vocabulary_name,
        "code": op.code,
        "display": op.display,
        "is_current": True,
        "model_role": _TARGET_ROLE,
        "id": (
            f"{op.target_model_id}|{op.target_model_version}|"
            f"{op.target_schema_snapshot_hash}|{op.vocabulary_name}|{op.code}"
        ),
    }


def constraint_merge(op: ConstraintOp) -> tuple[str, dict[str, Any]]:
    cypher = (
        "MERGE (n:Constraint {"
        "target_model_id: $target_model_id, "
        "target_model_version: $target_model_version, "
        "target_schema_snapshot_hash: $target_schema_snapshot_hash, "
        "attached_property_id: $attached_property_id, "
        "constraint_kind: $constraint_kind, "
        "payload_hash: $payload_hash"
        "}) "
        "SET n.payload_json = $payload_json, n.is_current = $is_current, "
        "n.model_role = $model_role, n.id = $id"
    )
    import json as _json

    return cypher, {
        "target_model_id": op.target_model_id,
        "target_model_version": op.target_model_version,
        "target_schema_snapshot_hash": op.target_schema_snapshot_hash,
        "attached_property_id": op.attached_property_id,
        "constraint_kind": op.constraint_kind,
        "payload_hash": op.payload_hash,
        "payload_json": _json.dumps(op.payload, sort_keys=True),
        "is_current": op.is_current,
        "model_role": _TARGET_ROLE,
        "id": (
            f"{op.target_model_id}|{op.target_model_version}|"
            f"{op.target_schema_snapshot_hash}|{op.attached_property_id}|"
            f"{op.constraint_kind}|{op.payload_hash}"
        ),
    }


def target_obligation_merge(op: TargetObligationOp) -> tuple[str, dict[str, Any]]:
    import json as _json

    cypher = (
        "MERGE (n:TargetObligation {"
        "target_model_id: $target_model_id, "
        "target_model_version: $target_model_version, "
        "target_schema_snapshot_hash: $target_schema_snapshot_hash, "
        "target_entity: $target_entity"
        "}) "
        "SET n.payload_json = $payload_json, n.is_current = $is_current, "
        "n.model_role = $model_role, n.id = $id"
    )
    return cypher, {
        "target_model_id": op.target_model_id,
        "target_model_version": op.target_model_version,
        "target_schema_snapshot_hash": op.target_schema_snapshot_hash,
        "target_entity": op.target_entity,
        "payload_json": _json.dumps(op.payload, sort_keys=True, default=str),
        "is_current": True,
        "model_role": _TARGET_ROLE,
        "id": (
            f"{op.target_model_id}|{op.target_model_version}|"
            f"{op.target_schema_snapshot_hash}|{op.target_entity}"
        ),
    }


def enrichment_decision_merge(op: EnrichmentDecisionOp) -> tuple[str, dict[str, Any]]:
    cypher = (
        "MERGE (n:EnrichmentDecision {"
        "target_model_id: $target_model_id, "
        "target_model_version: $target_model_version, "
        "target_schema_snapshot_hash: $target_schema_snapshot_hash, "
        "entity_ref: $entity_ref"
        "}) "
        "SET n.decisions_json = $decisions_json, n.decided_at = $decided_at, "
        "n.is_current = $is_current, n.model_role = $model_role, n.id = $id"
    )
    return cypher, {
        "target_model_id": op.target_model_id,
        "target_model_version": op.target_model_version,
        "target_schema_snapshot_hash": op.target_schema_snapshot_hash,
        "entity_ref": op.entity_ref,
        "decisions_json": op.decisions_json,
        "decided_at": op.decided_at.isoformat(),
        "is_current": op.is_current,
        "model_role": _TARGET_ROLE,
        "id": (
            f"{op.target_model_id}|{op.target_model_version}|"
            f"{op.target_schema_snapshot_hash}|{op.entity_ref}"
        ),
    }


def vocabulary_binding_merge(op: VocabularyBindingOp) -> tuple[str, dict[str, Any]]:
    cypher = (
        "MERGE (n:VocabularyBinding {"
        "target_model_id: $target_model_id, "
        "target_model_version: $target_model_version, "
        "target_schema_snapshot_hash: $target_schema_snapshot_hash, "
        "parent_entity_qualified_name: $parent_entity_qualified_name, "
        "property_name: $property_name, "
        "vocabulary_name: $vocabulary_name"
        "}) "
        "SET n.vocabulary_source = $vocabulary_source, n.domain = $domain, "
        "n.require_standard = $require_standard, "
        "n.allow_zero_default = $allow_zero_default, "
        "n.standard_domain_governed = $standard_domain_governed, "
        "n.effective_date_ref = $effective_date_ref, "
        "n.resolver_policy_ref = $resolver_policy_ref, "
        "n.is_current = $is_current, n.model_role = $model_role, n.id = $id"
    )
    return cypher, {
        "target_model_id": op.target_model_id,
        "target_model_version": op.target_model_version,
        "target_schema_snapshot_hash": op.target_schema_snapshot_hash,
        "parent_entity_qualified_name": op.parent_entity_qualified_name,
        "property_name": op.property_name,
        "vocabulary_name": op.vocabulary_name,
        "vocabulary_source": op.vocabulary_source,
        "domain": op.domain,
        "require_standard": op.require_standard,
        "allow_zero_default": op.allow_zero_default,
        "standard_domain_governed": op.standard_domain_governed,
        "effective_date_ref": op.effective_date_ref,
        "resolver_policy_ref": op.resolver_policy_ref,
        "is_current": op.is_current,
        "model_role": _TARGET_ROLE,
        "id": (
            f"{op.target_model_id}|{op.target_model_version}|"
            f"{op.target_schema_snapshot_hash}|"
            f"{op.parent_entity_qualified_name}.{op.property_name}|"
            f"{op.vocabulary_name}"
        ),
    }


def context_card_merge(op: ContextCardOp) -> tuple[str, dict[str, Any]]:
    cypher = (
        "MERGE (n:ContextCard {"
        "target_model_id: $target_model_id, "
        "target_model_version: $target_model_version, "
        "target_schema_snapshot_hash: $target_schema_snapshot_hash, "
        "entity_qualified_name: $entity_qualified_name, "
        "card_version: $card_version"
        "}) "
        "SET n.card_hash = $card_hash, "
        "n.description = $description, n.examples = $examples, "
        "n.obligation_summary = $obligation_summary, "
        "n.curated_synonyms = $curated_synonyms, "
        "n.is_current = $is_current, n.model_role = $model_role, n.id = $id"
    )
    return cypher, {
        "target_model_id": op.target_model_id,
        "target_model_version": op.target_model_version,
        "target_schema_snapshot_hash": op.target_schema_snapshot_hash,
        "entity_qualified_name": op.entity_qualified_name,
        "card_version": op.card_version,
        "card_hash": op.card_hash,
        "description": op.description,
        "examples": list(op.examples),
        "obligation_summary": op.obligation_summary,
        "curated_synonyms": list(op.curated_synonyms),
        "is_current": op.is_current,
        "model_role": _TARGET_ROLE,
        "id": (
            f"{op.target_model_id}|{op.target_model_version}|"
            f"{op.target_schema_snapshot_hash}|{op.entity_qualified_name}|"
            f"{op.card_version}"
        ),
    }


def relationship_merge(op: RelationshipOp) -> tuple[str, dict[str, Any]]:
    from_match = _key_predicate("a", "from", op.from_keys)
    to_match = _key_predicate("b", "to", op.to_keys)
    cypher = (
        f"MATCH (a:{op.from_label}) WHERE {from_match} "
        f"MATCH (b:{op.to_label}) WHERE {to_match} "
        f"MERGE (a)-[:{op.rel_type} "
        "{target_schema_snapshot_hash: $rel_snapshot_hash}]->(b)"
    )
    params: dict[str, Any] = {"rel_snapshot_hash": op.target_schema_snapshot_hash}
    for k, v in op.from_keys.items():
        params[f"from_{k}"] = v
    for k, v in op.to_keys.items():
        params[f"to_{k}"] = v
    return cypher, params


def _key_predicate(var: str, prefix: str, keys: dict[str, str]) -> str:
    parts = [f"{var}.{k} = ${prefix}_{k}" for k in keys]
    return " AND ".join(parts)


from sema.targets.neo4j_writer_flip_utils import flip_statements

__all__ = [
    "constraint_merge",
    "context_card_merge",
    "enrichment_decision_merge",
    "entity_merge",
    "flip_statements",
    "property_merge",
    "relationship_merge",
    "target_obligation_merge",
    "term_merge",
    "vocabulary_binding_merge",
]

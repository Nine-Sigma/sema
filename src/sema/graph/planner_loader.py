"""Round-trip helpers between Pydantic planner models and Neo4j properties.

This module owns the native-property storage layout for `Provenance`:
prefix `prov_run_*` for `RunProvenance`, `prov_source_*` for `SourceScope`,
and `prov_timestamp` for the per-call timestamp. Same layout applies to
`MappingAssertion`, `ResolutionPlan`, `RiskFlag`, and
`HumanPin.confirmed_under`.

It also owns the app-layer guard for relationship-target model_role rules
declared in planner-graph-storage spec 8.5. Neo4j Community lacks
relationship-target property constraints; APOC triggers fill the gap when
available (see `planner_migrations.cypher_up(apoc=True)`), and these write
helpers enforce the same rules from Python regardless of edition.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from sema.models.planner._enums import ModelRole
from sema.models.planner._role_validation import (
    require_property_role_for_relationship,
)
from sema.models.planner.field_map import FieldMap, RowIdentity
from sema.models.planner.lifecycle import HumanPin, PinState, Status
from sema.models.planner.mapping_plan import MappingAssertion, MappingPlan
from sema.models.planner.patterns import (
    MappingPattern,
    PatternPayload,
    expected_payload_type,
)
from sema.models.planner.provenance import (
    Provenance,
    RunProvenance,
    SourceScope,
)
from sema.models.planner.resolution import ResolutionPlan
from sema.models.planner.risk import RiskFlag
from sema.models.planner.target_model import TargetObligation


_RUN_FIELDS = (
    "run_id",
    "target_model_version",
    "target_schema_snapshot_hash",
    "vocab_release",
    "context_card_version",
    "prompt_template_version",
    "few_shot_set_version",
    "constraint_version",
    "llm_model",
    "embedding_model",
)
_SOURCE_FIELDS = ("source_id", "source_schema_hash", "source_profile_hash")


def provenance_to_properties(prov: Provenance) -> dict[str, Any]:
    props: dict[str, Any] = {}
    for f in _RUN_FIELDS:
        props[f"prov_run_{f}"] = getattr(prov.run, f)
    for f in _SOURCE_FIELDS:
        props[f"prov_source_{f}"] = getattr(prov.source, f)
    props["prov_timestamp"] = prov.timestamp.isoformat()
    return props


def properties_to_provenance(props: dict[str, Any]) -> Provenance:
    run = RunProvenance(**{f: props[f"prov_run_{f}"] for f in _RUN_FIELDS})
    source = SourceScope(**{f: props[f"prov_source_{f}"] for f in _SOURCE_FIELDS})
    timestamp = datetime.fromisoformat(props["prov_timestamp"])
    return Provenance(run=run, source=source, timestamp=timestamp)


def confirmed_under_to_properties(
    run: RunProvenance, source: SourceScope
) -> dict[str, Any]:
    props: dict[str, Any] = {}
    for f in _RUN_FIELDS:
        props[f"prov_run_{f}"] = getattr(run, f)
    for f in _SOURCE_FIELDS:
        props[f"prov_source_{f}"] = getattr(source, f)
    return props


def properties_to_confirmed_under(
    props: dict[str, Any],
) -> tuple[RunProvenance, SourceScope]:
    run = RunProvenance(**{f: props[f"prov_run_{f}"] for f in _RUN_FIELDS})
    source = SourceScope(**{f: props[f"prov_source_{f}"] for f in _SOURCE_FIELDS})
    return run, source


def cypher_create_field_map_maps_to(
    field_map_id: str, target_property_id: str, target_role: ModelRole
) -> tuple[str, dict[str, str]]:
    require_property_role_for_relationship("MAPS_TO", target_role)
    stmt = (
        "MATCH (f:FieldMap {id: $fm_id}), (p:Property {id: $p_id}) "
        "MERGE (f)-[:MAPS_TO]->(p)"
    )
    return stmt, {"fm_id": field_map_id, "p_id": target_property_id}


def cypher_create_field_map_derived_from(
    field_map_id: str, source_property_id: str, source_role: ModelRole
) -> tuple[str, dict[str, str]]:
    require_property_role_for_relationship("DERIVED_FROM", source_role)
    stmt = (
        "MATCH (f:FieldMap {id: $fm_id}), (p:Property {id: $p_id}) "
        "MERGE (f)-[:DERIVED_FROM]->(p)"
    )
    return stmt, {"fm_id": field_map_id, "p_id": source_property_id}


def cypher_create_plan_has_lineage(
    plan_id: str, source_property_id: str, source_role: ModelRole
) -> tuple[str, dict[str, str]]:
    require_property_role_for_relationship("HAS_LINEAGE", source_role)
    stmt = (
        "MATCH (m:MappingPlan {id: $plan_id}), (p:Property {id: $p_id}) "
        "MERGE (m)-[:HAS_LINEAGE]->(p)"
    )
    return stmt, {"plan_id": plan_id, "p_id": source_property_id}


def cypher_create_resolution_input(
    resolution_plan_id: str, source_property_id: str, source_role: ModelRole
) -> tuple[str, dict[str, str]]:
    require_property_role_for_relationship("RESOLUTION_INPUT", source_role)
    stmt = (
        "MATCH (r:ResolutionPlan {id: $rp_id}), (p:Property {id: $p_id}) "
        "MERGE (r)-[:RESOLUTION_INPUT]->(p)"
    )
    return stmt, {"rp_id": resolution_plan_id, "p_id": source_property_id}


def mapping_assertion_to_properties(a: MappingAssertion) -> dict[str, Any]:
    props = provenance_to_properties(a.provenance)
    props.update(
        id=a.id,
        source_field_ref=a.source_field_ref,
        target_property_ref=a.target_property_ref,
        pattern=a.pattern.value,
        payload_json=a.payload.model_dump_json(),
        confidence=a.confidence,
        status=a.status.value,
        risk_flags_json=json.dumps(
            [rf.model_dump(mode="json") for rf in a.risk_flags]
        ),
        concerns_text=a.concerns_text,
    )
    return props


def properties_to_mapping_assertion(props: dict[str, Any]) -> MappingAssertion:
    from typing import cast

    pattern = MappingPattern(props["pattern"])
    payload_cls = expected_payload_type(pattern)
    payload = cast(
        PatternPayload, payload_cls.model_validate_json(props["payload_json"])
    )
    risk_flags = [
        RiskFlag.model_validate(rf)
        for rf in json.loads(props.get("risk_flags_json") or "[]")
    ]
    return MappingAssertion(
        id=props["id"],
        source_field_ref=props["source_field_ref"],
        target_property_ref=props["target_property_ref"],
        pattern=pattern,
        payload=payload,
        confidence=props["confidence"],
        status=Status(props["status"]),
        risk_flags=risk_flags,
        provenance=properties_to_provenance(props),
        concerns_text=props.get("concerns_text"),
    )


def field_map_to_properties(fm: FieldMap) -> dict[str, Any]:
    return {
        "target_field_ref": fm.target_field_ref,
        "pattern": fm.pattern.value,
        "payload_json": fm.payload.model_dump_json(),
    }


def properties_to_field_map(props: dict[str, Any]) -> FieldMap:
    return FieldMap.model_validate(
        {
            "target_field_ref": props["target_field_ref"],
            "pattern": props["pattern"],
            "payload": json.loads(props["payload_json"]),
        }
    )


def target_obligation_to_properties(o: TargetObligation) -> dict[str, Any]:
    return {
        "target_entity": o.target_entity,
        "primary_key": o.primary_key.value,
        "obligation_json": o.model_dump_json(),
    }


def properties_to_target_obligation(props: dict[str, Any]) -> TargetObligation:
    return TargetObligation.model_validate_json(props["obligation_json"])


def mapping_plan_to_properties(p: MappingPlan) -> dict[str, Any]:
    return {
        "id": p.id,
        "source_scope_ref": p.source_scope_ref,
        "plan_verdict": p.derive_verdict().value,
        "obligation_json": p.obligation.model_dump_json(),
        "row_identity_json": p.row_identity.model_dump_json(),
        "field_maps_json": json.dumps(
            [fm.model_dump(mode="json") for fm in p.field_maps]
        ),
        "risk_flags_json": json.dumps(
            [rf.model_dump(mode="json") for rf in p.risk_flags]
        ),
        "lineage_json": json.dumps(p.lineage),
    }


def properties_to_mapping_plan(props: dict[str, Any]) -> MappingPlan:
    return MappingPlan(
        id=props["id"],
        source_scope_ref=props["source_scope_ref"],
        obligation=TargetObligation.model_validate_json(props["obligation_json"]),
        row_identity=RowIdentity.model_validate_json(props["row_identity_json"]),
        field_maps=[
            FieldMap.model_validate(fm)
            for fm in json.loads(props.get("field_maps_json") or "[]")
        ],
        risk_flags=[
            RiskFlag.model_validate(rf)
            for rf in json.loads(props.get("risk_flags_json") or "[]")
        ],
        lineage=json.loads(props.get("lineage_json") or "[]"),
    )


def resolution_plan_to_properties(rp: ResolutionPlan) -> dict[str, Any]:
    props: dict[str, Any] = {
        f"prov_run_{f}": getattr(rp.provenance_run, f) for f in _RUN_FIELDS
    }
    props.update(
        id=rp.id,
        target_identity_ref=rp.target_identity_ref,
        strategy=rp.strategy.value,
        confidence=rp.confidence,
        status=rp.status.value,
        plan_json=rp.model_dump_json(),
        prov_timestamp=rp.timestamp.isoformat(),
    )
    return props


def properties_to_resolution_plan(props: dict[str, Any]) -> ResolutionPlan:
    return ResolutionPlan.model_validate_json(props["plan_json"])


def risk_flag_to_properties(rf: RiskFlag) -> dict[str, Any]:
    return {
        "code": rf.code.value,
        "severity": rf.severity.value,
        "source_stage": rf.source_stage.value,
        "suggested_action": rf.suggested_action.value,
        "evidence_json": json.dumps(
            [e.model_dump(mode="json") for e in rf.evidence]
        ),
        "flag_json": rf.model_dump_json(),
    }


def properties_to_risk_flag(props: dict[str, Any]) -> RiskFlag:
    return RiskFlag.model_validate_json(props["flag_json"])


def human_pin_to_properties(pin: HumanPin) -> dict[str, Any]:
    props = confirmed_under_to_properties(
        pin.confirmed_under_run, pin.confirmed_under_source
    )
    props.update(
        id=pin.pin_id,
        pin_id=pin.pin_id,
        assertion_id=pin.assertion_id,
        resolution_plan_id=pin.resolution_plan_id,
        pinned_at=pin.pinned_at.isoformat(),
        pinned_by=pin.pinned_by,
        pin_state=pin.pin_state.value,
        expires_on_change_of=list(pin.expires_on_change_of),
    )
    return props


def properties_to_human_pin(props: dict[str, Any]) -> HumanPin:
    run, source = properties_to_confirmed_under(props)
    return HumanPin(
        pin_id=props["pin_id"],
        assertion_id=props.get("assertion_id"),
        resolution_plan_id=props.get("resolution_plan_id"),
        pinned_at=datetime.fromisoformat(props["pinned_at"]),
        pinned_by=props["pinned_by"],
        confirmed_under_run=run,
        confirmed_under_source=source,
        pin_state=PinState(props["pin_state"]),
        expires_on_change_of=list(props["expires_on_change_of"]),
    )


# --- Cypher write/read helpers (driver-aware wrappers around serializers).
#
# These wrap a Neo4j ``Session`` so callers don't have to know labels or
# write Cypher inline. Each ``write_*`` / ``read_*`` pair satisfies spec 8.8
# "write helpers, read-back helpers, and round-trip serialization for each
# planner node kind."


def write_mapping_assertion(session: Any, a: MappingAssertion) -> None:
    session.run(
        "MERGE (n:MappingAssertion {id: $props.id}) SET n = $props",
        props=mapping_assertion_to_properties(a),
    )


def read_mapping_assertion(session: Any, assertion_id: str) -> MappingAssertion:
    row = session.run(
        "MATCH (n:MappingAssertion {id: $id}) RETURN properties(n) AS p",
        id=assertion_id,
    ).single()
    if row is None:
        raise LookupError(f"MappingAssertion id={assertion_id!r} not found")
    return properties_to_mapping_assertion(row["p"])


def write_mapping_plan(session: Any, p: MappingPlan) -> None:
    session.run(
        "MERGE (n:MappingPlan {id: $props.id}) SET n = $props",
        props=mapping_plan_to_properties(p),
    )


def read_mapping_plan(session: Any, plan_id: str) -> MappingPlan:
    row = session.run(
        "MATCH (n:MappingPlan {id: $id}) RETURN properties(n) AS p",
        id=plan_id,
    ).single()
    if row is None:
        raise LookupError(f"MappingPlan id={plan_id!r} not found")
    return properties_to_mapping_plan(row["p"])


def write_resolution_plan(session: Any, rp: ResolutionPlan) -> None:
    session.run(
        "MERGE (n:ResolutionPlan {id: $props.id}) SET n = $props",
        props=resolution_plan_to_properties(rp),
    )


def read_resolution_plan(session: Any, plan_id: str) -> ResolutionPlan:
    row = session.run(
        "MATCH (n:ResolutionPlan {id: $id}) RETURN properties(n) AS p",
        id=plan_id,
    ).single()
    if row is None:
        raise LookupError(f"ResolutionPlan id={plan_id!r} not found")
    return properties_to_resolution_plan(row["p"])


def write_human_pin(session: Any, pin: HumanPin) -> None:
    session.run(
        "MERGE (n:HumanPin {id: $props.id}) SET n = $props",
        props=human_pin_to_properties(pin),
    )


def read_human_pin(session: Any, pin_id: str) -> HumanPin:
    row = session.run(
        "MATCH (n:HumanPin {id: $id}) RETURN properties(n) AS p",
        id=pin_id,
    ).single()
    if row is None:
        raise LookupError(f"HumanPin id={pin_id!r} not found")
    return properties_to_human_pin(row["p"])


def write_target_obligation(
    session: Any, obligation: TargetObligation, *, obligation_id: str
) -> None:
    props = target_obligation_to_properties(obligation)
    props["id"] = obligation_id
    session.run(
        "MERGE (n:TargetObligation {id: $props.id}) SET n = $props",
        props=props,
    )


def read_target_obligation(
    session: Any, obligation_id: str
) -> TargetObligation:
    row = session.run(
        "MATCH (n:TargetObligation {id: $id}) RETURN properties(n) AS p",
        id=obligation_id,
    ).single()
    if row is None:
        raise LookupError(f"TargetObligation id={obligation_id!r} not found")
    return properties_to_target_obligation(row["p"])


def write_risk_flag(session: Any, rf: RiskFlag, *, flag_id: str) -> None:
    props = risk_flag_to_properties(rf)
    props["id"] = flag_id
    session.run(
        "MERGE (n:RiskFlag {id: $props.id}) SET n = $props",
        props=props,
    )


def read_risk_flag(session: Any, flag_id: str) -> RiskFlag:
    row = session.run(
        "MATCH (n:RiskFlag {id: $id}) RETURN properties(n) AS p",
        id=flag_id,
    ).single()
    if row is None:
        raise LookupError(f"RiskFlag id={flag_id!r} not found")
    return properties_to_risk_flag(row["p"])


def write_field_map(session: Any, fm: FieldMap, *, field_map_id: str) -> None:
    props = field_map_to_properties(fm)
    props["id"] = field_map_id
    session.run(
        "MERGE (n:FieldMap {id: $props.id}) SET n = $props",
        props=props,
    )


def read_field_map(session: Any, field_map_id: str) -> FieldMap:
    row = session.run(
        "MATCH (n:FieldMap {id: $id}) RETURN properties(n) AS p",
        id=field_map_id,
    ).single()
    if row is None:
        raise LookupError(f"FieldMap id={field_map_id!r} not found")
    return properties_to_field_map(row["p"])

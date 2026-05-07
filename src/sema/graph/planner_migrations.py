"""Cypher migrations for the generic-mapping-planner-contract."""

from __future__ import annotations


PLANNER_NODE_LABELS = (
    "MappingAssertion",
    "MappingPlan",
    "FieldMap",
    "ResolutionPlan",
    "TargetObligation",
    "RiskFlag",
    "HumanPin",
)

PLANNER_RELATIONSHIPS = (
    "HAS_OBLIGATION",
    "ASSEMBLED_INTO",
    "FIELD_MAP_OF",
    "MAPS_TO",
    "DERIVED_FROM",
    "RESOLVED_BY",
    "HAS_LINEAGE",
    "RAISED_FLAG",
    "PINNED",
    "CONFLICT_LOSER",
    "RESOLUTION_INPUT",
)


def cypher_up(*, enterprise: bool = False, apoc: bool = False) -> list[str]:
    """Forward migration: add labels, relationships, indexes, model_role backfill.

    Property-existence constraints require Neo4j Enterprise; they are emitted
    only when ``enterprise=True``. On Community editions, model_role presence
    is enforced at the application layer (Pydantic validators) plus the
    backfill statements below.

    Relationship-target role rules from spec 8.5
    (`MAPS_TO`→TARGET, `DERIVED_FROM`/`HAS_LINEAGE`/`RESOLUTION_INPUT`→SOURCE)
    cannot be expressed as native Cypher constraints. When ``apoc=True`` the
    migration emits APOC `before` triggers that abort transactions creating
    role-mismatched relationships. When APOC is unavailable, the same rules
    are enforced from Python via `planner_loader.cypher_create_*` helpers.
    """
    statements: list[str] = []
    statements.extend(_role_backfill())
    statements.extend(_uniqueness_constraints())
    if enterprise:
        statements.extend(_role_existence_constraints())
    statements.extend(_indexes())
    if apoc:
        statements.extend(_apoc_relationship_role_triggers())
        statements.extend(_apoc_scoping_id_triggers())
    return statements


def cypher_down(*, apoc: bool = False) -> list[str]:
    """Reverse migration: drop planner labels, relationships, indexes, triggers.

    When ``apoc=True`` the migration also removes the planner APOC triggers
    (the symmetric inverse of ``cypher_up(apoc=True)``). Pass ``apoc=False``
    when the upgrade did not install triggers; otherwise the
    ``apoc.trigger.remove`` calls fail on a vanilla Neo4j Community node.
    """
    drops: list[str] = []
    if apoc:
        all_triggers = _APOC_TRIGGER_NAMES + _SCOPING_TRIGGER_NAMES
        for trigger_name in all_triggers:
            drops.append(
                f"CALL apoc.trigger.remove('{trigger_name}') YIELD name "
                "RETURN name"
            )
    for label in PLANNER_NODE_LABELS:
        drops.append(f"DROP CONSTRAINT {label}_id_unique IF EXISTS")
        drops.append(
            f"MATCH (n:{label}) DETACH DELETE n"
        )
    drops.extend(
        [
            "DROP INDEX mapping_assertion_run_id IF EXISTS",
            "DROP INDEX mapping_assertion_source_id IF EXISTS",
            "DROP INDEX mapping_assertion_status IF EXISTS",
            "DROP INDEX mapping_plan_verdict IF EXISTS",
            "DROP INDEX resolution_plan_status IF EXISTS",
            "DROP INDEX resolution_plan_verdict IF EXISTS",
            "DROP INDEX risk_flag_code IF EXISTS",
            "DROP INDEX human_pin_state IF EXISTS",
            "DROP INDEX human_pin_assertion_id IF EXISTS",
            "DROP INDEX human_pin_resolution_plan_id IF EXISTS",
            "DROP INDEX entity_model_role IF EXISTS",
            "DROP INDEX property_model_role IF EXISTS",
        ]
    )
    return drops


def _role_backfill() -> list[str]:
    """Backfill model_role + source_id on pre-planner nodes.

    `model_role` defaults to SOURCE. `source_id` is backfilled from any
    `source_schema` stamped on the node's incident edges (one representative
    edge per node) so the discriminator rule "SOURCE node has source_id"
    holds for already-loaded graphs. Nodes without any source_schema-stamped
    edge fall back to the legacy `source` field.
    """
    backfills: list[str] = []
    for label in ("Entity", "Property", "Term", "Constraint"):
        backfills.append(
            f"MATCH (n:{label}) WHERE n.model_role IS NULL "
            "SET n.model_role = 'SOURCE'"
        )
        backfills.append(
            f"MATCH (n:{label}) WHERE n.source_id IS NULL "
            "AND n.model_role = 'SOURCE' "
            "OPTIONAL MATCH (n)-[r]-() WHERE r.source_schema IS NOT NULL "
            "WITH n, head(collect(r.source_schema)) AS scope "
            "WITH n, coalesce(scope, n.source) AS resolved "
            "WHERE resolved IS NOT NULL "
            "SET n.source_id = resolved"
        )
    return backfills


def _uniqueness_constraints() -> list[str]:
    return [
        f"CREATE CONSTRAINT {label}_id_unique IF NOT EXISTS "
        f"FOR (n:{label}) REQUIRE n.id IS UNIQUE"
        for label in PLANNER_NODE_LABELS
    ]


def _role_existence_constraints() -> list[str]:
    return [
        "CREATE CONSTRAINT entity_model_role_exists IF NOT EXISTS "
        "FOR (n:Entity) REQUIRE n.model_role IS NOT NULL",
        "CREATE CONSTRAINT property_model_role_exists IF NOT EXISTS "
        "FOR (n:Property) REQUIRE n.model_role IS NOT NULL",
        "CREATE CONSTRAINT term_model_role_exists IF NOT EXISTS "
        "FOR (n:Term) REQUIRE n.model_role IS NOT NULL",
        "CREATE CONSTRAINT constraint_model_role_exists IF NOT EXISTS "
        "FOR (n:Constraint) REQUIRE n.model_role IS NOT NULL",
    ]


_SCOPING_TRIGGER_NAMES = (
    "planner_no_role_id_collision",
    "planner_source_role_requires_source_id",
    "planner_target_role_requires_target_model_id",
)


def _apoc_scoping_id_triggers() -> list[str]:
    """APOC triggers enforcing the source_id/target_model_id discriminator.

    Spec 2.2 + 2.1: a node MUST NOT carry both `source_id` and
    `target_model_id`; SOURCE-role nodes MUST carry source_id; TARGET-role
    nodes MUST carry target_model_id. The Pydantic models enforce this at
    construction; these triggers enforce it for nodes written through any
    path, including legacy loaders.
    """
    labels = "['Entity','Property','Term','Constraint']"
    return [
        f"CALL apoc.trigger.add('planner_no_role_id_collision', \""
        f"UNWIND $createdNodes AS n "
        f"WITH n WHERE any(l IN labels(n) WHERE l IN {labels}) "
        f"AND n.source_id IS NOT NULL AND n.target_model_id IS NOT NULL "
        f"CALL apoc.util.validate(true, "
        f"'node MUST NOT carry both source_id and target_model_id', []) "
        f"RETURN 1\", {{phase: 'before'}})",
        f"CALL apoc.trigger.add('planner_source_role_requires_source_id', \""
        f"UNWIND $createdNodes AS n "
        f"WITH n WHERE any(l IN labels(n) WHERE l IN {labels}) "
        f"AND coalesce(n.model_role, 'SOURCE') = 'SOURCE' "
        f"AND n.source_id IS NULL "
        f"CALL apoc.util.validate(true, "
        f"'model_role=SOURCE requires source_id', []) "
        f"RETURN 1\", {{phase: 'before'}})",
        f"CALL apoc.trigger.add('planner_target_role_requires_target_model_id', "
        f"\"UNWIND $createdNodes AS n "
        f"WITH n WHERE any(l IN labels(n) WHERE l IN {labels}) "
        f"AND n.model_role = 'TARGET' "
        f"AND n.target_model_id IS NULL "
        f"CALL apoc.util.validate(true, "
        f"'model_role=TARGET requires target_model_id', []) "
        f"RETURN 1\", {{phase: 'before'}})",
    ]


_APOC_TRIGGER_NAMES = (
    "planner_maps_to_requires_target_property",
    "planner_derived_from_requires_source_property",
    "planner_has_lineage_requires_source_property",
    "planner_resolution_input_requires_source_property",
)


def _apoc_relationship_role_triggers() -> list[str]:
    """APOC `before` triggers enforcing spec 8.5 relationship-target role rules."""
    return [
        _apoc_role_trigger(
            name="planner_maps_to_requires_target_property",
            rel_type="MAPS_TO",
            required_role="TARGET",
        ),
        _apoc_role_trigger(
            name="planner_derived_from_requires_source_property",
            rel_type="DERIVED_FROM",
            required_role="SOURCE",
        ),
        _apoc_role_trigger(
            name="planner_has_lineage_requires_source_property",
            rel_type="HAS_LINEAGE",
            required_role="SOURCE",
        ),
        _apoc_role_trigger(
            name="planner_resolution_input_requires_source_property",
            rel_type="RESOLUTION_INPUT",
            required_role="SOURCE",
        ),
    ]


def _apoc_role_trigger(*, name: str, rel_type: str, required_role: str) -> str:
    body = (
        "UNWIND $createdRelationships AS r "
        f"WITH r WHERE type(r) = '{rel_type}' "
        "WITH r, endNode(r) AS p "
        "WHERE p:Property AND coalesce(p.model_role, 'SOURCE') <> "
        f"'{required_role}' "
        "CALL apoc.util.validate(true, "
        f"'{rel_type} requires Property.model_role={required_role}', []) "
        "RETURN 1"
    )
    return (
        f"CALL apoc.trigger.add('{name}', \"{body}\", {{phase: 'before'}})"
    )


def _indexes() -> list[str]:
    return [
        "CREATE INDEX mapping_assertion_run_id IF NOT EXISTS "
        "FOR (n:MappingAssertion) ON (n.prov_run_run_id)",
        "CREATE INDEX mapping_assertion_source_id IF NOT EXISTS "
        "FOR (n:MappingAssertion) ON (n.prov_source_source_id)",
        "CREATE INDEX mapping_assertion_status IF NOT EXISTS "
        "FOR (n:MappingAssertion) ON (n.status)",
        "CREATE INDEX mapping_plan_verdict IF NOT EXISTS "
        "FOR (n:MappingPlan) ON (n.plan_verdict)",
        "CREATE INDEX resolution_plan_status IF NOT EXISTS "
        "FOR (n:ResolutionPlan) ON (n.status)",
        "CREATE INDEX resolution_plan_verdict IF NOT EXISTS "
        "FOR (n:ResolutionPlan) ON (n.resolution_verdict)",
        "CREATE INDEX risk_flag_code IF NOT EXISTS "
        "FOR (n:RiskFlag) ON (n.code)",
        "CREATE INDEX human_pin_state IF NOT EXISTS "
        "FOR (n:HumanPin) ON (n.pin_state)",
        "CREATE INDEX human_pin_assertion_id IF NOT EXISTS "
        "FOR (n:HumanPin) ON (n.assertion_id)",
        "CREATE INDEX human_pin_resolution_plan_id IF NOT EXISTS "
        "FOR (n:HumanPin) ON (n.resolution_plan_id)",
        "CREATE INDEX entity_model_role IF NOT EXISTS "
        "FOR (n:Entity) ON (n.model_role)",
        "CREATE INDEX property_model_role IF NOT EXISTS "
        "FOR (n:Property) ON (n.model_role)",
    ]

"""Validation helpers for the model_role discriminator on graph nodes."""

from __future__ import annotations

from sema.models.planner._enums import ModelRole, TargetArtifactKind


_REL_REQUIRED_PROPERTY_ROLE: dict[str, ModelRole] = {
    "MAPS_TO": ModelRole.TARGET,
    "DERIVED_FROM": ModelRole.SOURCE,
    "HAS_LINEAGE": ModelRole.SOURCE,
    "RESOLUTION_INPUT": ModelRole.SOURCE,
}


def require_role_identifier(
    role: ModelRole, source_id: str | None, target_model_id: str | None
) -> None:
    if source_id is not None and target_model_id is not None:
        raise ValueError(
            "node MUST NOT carry both source_id and target_model_id"
        )
    if role is ModelRole.TARGET:
        if target_model_id is None:
            raise ValueError("model_role=TARGET requires target_model_id")
        if source_id is not None:
            raise ValueError("model_role=TARGET rejects source_id")
    if role is ModelRole.SOURCE:
        if target_model_id is not None:
            raise ValueError("model_role=SOURCE rejects target_model_id")
        if source_id is None:
            raise ValueError("model_role=SOURCE requires source_id")


def require_kind_matches_role(
    role: ModelRole, kind: TargetArtifactKind | None
) -> None:
    if role is ModelRole.TARGET and kind is None:
        raise ValueError("model_role=TARGET Entity requires kind")
    if role is ModelRole.SOURCE and kind is not None:
        raise ValueError("model_role=SOURCE Entity rejects kind")


def required_property_role(rel_type: str) -> ModelRole | None:
    """Return the Property.model_role this relationship type must point at."""
    return _REL_REQUIRED_PROPERTY_ROLE.get(rel_type)


def require_property_role_for_relationship(
    rel_type: str, property_role: ModelRole
) -> None:
    """Reject a relationship whose target Property.model_role mismatches the contract.

    Per planner-graph-storage spec 8.5: MAPS_TO targets MUST be TARGET-role;
    DERIVED_FROM, HAS_LINEAGE, RESOLUTION_INPUT targets MUST be SOURCE-role.
    """
    required = _REL_REQUIRED_PROPERTY_ROLE.get(rel_type)
    if required is None:
        return
    if property_role is not required:
        raise ValueError(
            f"{rel_type} requires Property.model_role={required.value}, "
            f"got {property_role.value}"
        )

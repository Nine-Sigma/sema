"""Manifest file → ParsedManifest with file-format and structural validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from sema.models.planner._enums import TargetArtifactKind
from sema.targets.adapters.manifest_exceptions import (
    ManifestContextCardError,
    ManifestEndpointError,
    ManifestReservedNameError,
    ManifestSchemaError,
    UnsupportedManifestExtensionError,
    UnsupportedManifestVersionError,
)
from sema.targets.adapters.manifest_models import (
    ManifestEntity,
    ParsedManifest,
)

SUPPORTED_MANIFEST_VERSIONS: frozenset[int] = frozenset({1})
RESERVED_ENDPOINT_PROPERTY_NAMES: frozenset[str] = frozenset({"subject", "object"})


def parse_manifest_file(path: Path) -> ParsedManifest:
    raw = _load_raw(path)
    return parse_manifest_raw(raw)


def parse_manifest_raw(raw: dict[str, Any]) -> ParsedManifest:
    _reject_card_hash_anywhere(raw)
    version = raw.get("manifest_version")
    if version not in SUPPORTED_MANIFEST_VERSIONS:
        raise UnsupportedManifestVersionError(
            f"manifest_version={version!r} is not supported; "
            f"supported={sorted(SUPPORTED_MANIFEST_VERSIONS)}"
        )
    try:
        parsed = ParsedManifest.model_validate(raw)
    except ValidationError as exc:
        raise ManifestSchemaError(str(exc)) from exc
    _validate_entities(parsed)
    _validate_endpoint_kinds(parsed)
    return parsed


def _load_raw(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    text = path.read_text()
    if suffix in (".yaml", ".yml"):
        loaded = yaml.safe_load(text)
    elif suffix == ".json":
        loaded = json.loads(text)
    else:
        raise UnsupportedManifestExtensionError(
            f"manifest file extension {suffix!r} is not supported; "
            f"use .yaml, .yml, or .json"
        )
    if not isinstance(loaded, dict):
        raise ManifestSchemaError(
            f"manifest root must be a mapping; got {type(loaded).__name__}"
        )
    return loaded


def _reject_card_hash_anywhere(raw: Any, path: str = "$") -> None:
    if isinstance(raw, dict):
        if "card_hash" in raw:
            raise ManifestContextCardError(
                f"manifest at {path} carries forbidden field 'card_hash'; "
                f"card_hash is computed by Sema"
            )
        for key, value in raw.items():
            _reject_card_hash_anywhere(value, f"{path}.{key}")
    elif isinstance(raw, list):
        for i, item in enumerate(raw):
            _reject_card_hash_anywhere(item, f"{path}[{i}]")


def _validate_entities(parsed: ParsedManifest) -> None:
    for entity in parsed.entities:
        _validate_reserved_names(entity)
        _validate_endpoints_kind_invariant(entity)
        _validate_context_card(entity)


def _validate_reserved_names(entity: ManifestEntity) -> None:
    for prop in entity.properties:
        if prop.name in RESERVED_ENDPOINT_PROPERTY_NAMES:
            raise ManifestReservedNameError(
                f"entity {entity.qualified_name!r} declares property {prop.name!r}; "
                f"'subject' and 'object' are reserved for synthesized endpoint properties"
            )


def _validate_endpoints_kind_invariant(entity: ManifestEntity) -> None:
    if entity.kind is TargetArtifactKind.GRAPH_EDGE:
        if entity.endpoints is None:
            raise ManifestEndpointError(
                f"entity {entity.qualified_name!r} kind=GRAPH_EDGE requires endpoints"
            )
    else:
        if entity.endpoints is not None:
            raise ManifestEndpointError(
                f"entity {entity.qualified_name!r} kind={entity.kind.value} "
                f"forbids endpoints"
            )


def _validate_context_card(entity: ManifestEntity) -> None:
    if entity.context_card is None:
        return
    if not entity.context_card.description.strip():
        raise ManifestContextCardError(
            f"entity {entity.qualified_name!r} context_card has empty description"
        )


def _validate_endpoint_kinds(parsed: ParsedManifest) -> None:
    by_qname = {e.qualified_name: e for e in parsed.entities}
    for entity in parsed.entities:
        if entity.endpoints is None:
            continue
        for role, endpoint in (
            ("subject", entity.endpoints.subject),
            ("object", entity.endpoints.object),
        ):
            target = by_qname.get(endpoint.target_entity)
            if target is None:
                continue
            if target.kind is TargetArtifactKind.TABLE_ROW:
                raise ManifestEndpointError(
                    f"entity {entity.qualified_name!r} endpoint {role!r} targets "
                    f"{endpoint.target_entity!r} of kind=TABLE_ROW; endpoints MUST "
                    f"reference GRAPH_NODE or GRAPH_EDGE entities"
                )


__all__ = [
    "parse_manifest_file",
    "parse_manifest_raw",
    "SUPPORTED_MANIFEST_VERSIONS",
    "RESERVED_ENDPOINT_PROPERTY_NAMES",
]

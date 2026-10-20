"""Concept source spec: which relations hold a concept library and their column roles.

Version 1 is the original all-sections spec. Version 2 adds per-relation
providers, optional sections, hierarchy forms, lifecycle and language roles,
SSSOM mappings, term storage, an entity model, and a recorded shape. A version 1
spec cannot use a version 2 field, so the version marker can be trusted.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, field_validator, model_validator

from sema.binding.spec_sections import (
    AuthoredBindingRoles,
    AuthoredBindingsSpec,
    ConceptRoles,
    ConceptsSpec,
    DetailRoles,
    DetailsSpec,
    EndpointFallback,
    EndpointRole,
    EntityModelRoles,
    EntityModelSpec,
    ExternalRefRole,
    HierarchySpec,
    LevelColumnsHierarchy,
    LifecycleRoles,
    MappingRoles,
    MappingsSpec,
    ParentColumnHierarchy,
    PathCodesHierarchy,
    Provider,
    RelationRef,
    RelationshipRoles,
    RelationshipsSpec,
    SchemeSpec,
    ShapeEntry,
    StatusRole,
    TermStorageSpec,
    ValueMappingRoles,
    ValueMappingsSpec,
    ValueSetRoles,
    ValueSetsSpec,
    _SpecModel,
)

SUPPORTED_SPEC_VERSIONS = (1, 2)
_LIFECYCLE_FIELDS = ("status", "valid_from", "valid_to", "replaced_by", "lang")
VERSION_2_FIELDS: tuple[tuple[str, str], ...] = (
    ("", "default_provider"),
    ("", "shape"),
    ("", "hierarchy"),
    ("", "mappings"),
    ("", "term_storage"),
    ("", "entity_model"),
    *(("concepts.roles", name) for name in _LIFECYCLE_FIELDS),
    *(("value_sets.roles", name) for name in _LIFECYCLE_FIELDS),
)
SCOPED_SECTIONS = (
    "value_sets",
    "concept_details",
    "term_details",
    "relationships",
    "authored_bindings",
    "value_mappings",
)
VERSION_1_SECTIONS = ("scheme", "concepts", *SCOPED_SECTIONS)
RELATION_SECTIONS = (
    "concepts",
    *SCOPED_SECTIONS,
    "hierarchy",
    "mappings",
    "entity_model",
)


class ConceptSourceSpec(_SpecModel):
    spec_version: int = 1
    id: str = Field(min_length=1)
    identity_system: str = Field(min_length=1)
    release: str = Field(min_length=1)
    default_provider: Provider = "warehouse"
    shape: list[ShapeEntry] | None = None
    scheme: SchemeSpec | None = None
    concepts: ConceptsSpec | None = None
    value_sets: ValueSetsSpec | None = None
    concept_details: DetailsSpec | None = None
    term_details: DetailsSpec | None = None
    relationships: RelationshipsSpec | None = None
    hierarchy: HierarchySpec | None = None
    authored_bindings: AuthoredBindingsSpec | None = None
    value_mappings: ValueMappingsSpec | None = None
    mappings: MappingsSpec | None = None
    term_storage: TermStorageSpec = Field(default_factory=TermStorageSpec)
    entity_model: EntityModelSpec | None = None

    @property
    def external_namespace(self) -> str:
        explicit = self.term_storage.external_namespace
        return explicit or "vocabulary_" + re.sub(r"\W", "_", self.id)

    @field_validator("spec_version")
    @classmethod
    def _supported_version(cls, value: int) -> int:
        if value not in SUPPORTED_SPEC_VERSIONS:
            supported = ", ".join(str(v) for v in SUPPORTED_SPEC_VERSIONS)
            raise ValueError(
                f"spec_version {value} is not supported; supported versions are "
                f"{{{supported}}}"
            )
        return value

    @model_validator(mode="before")
    @classmethod
    def _resolve_relations(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        version = data.get("spec_version", 1)
        provider = data.get("default_provider", "warehouse")
        resolved = dict(data)
        for section in RELATION_SECTIONS:
            body = resolved.get(section)
            if isinstance(body, dict) and "relation" in body:
                relation = _relation_ref(section, body["relation"], provider, version)
                resolved[section] = {**body, "relation": relation}
        return resolved

    @model_validator(mode="after")
    def _check_sections(self) -> ConceptSourceSpec:
        if self.spec_version == 1:
            _gate_version_1(self)
        if self.concepts is None and self.entity_model is None:
            raise ValueError("a spec needs `concepts` or `entity_model`; it has neither")
        if self.concepts is not None and self.scheme is None:
            raise ValueError("`scheme` is required when `concepts` is present")
        _check_single_hierarchy(self)
        _check_scope_providers(self)
        return self


def _relation_ref(section: str, value: Any, provider: str, version: Any) -> Any:
    if isinstance(value, str):
        return {"provider": provider, "name": value}
    if version == 1:
        raise ValueError(
            f"{section}.relation as an object requires spec_version: 2"
        )
    return value


def _gate_version_1(spec: ConceptSourceSpec) -> None:
    for path, name in VERSION_2_FIELDS:
        owner = _at_path(spec, path)
        if owner is not None and name in owner.model_fields_set:
            where = f"{path}.{name}" if path else name
            raise ValueError(f"field `{where}` requires spec_version: 2")
    missing = [name for name in VERSION_1_SECTIONS if getattr(spec, name) is None]
    if missing:
        raise ValueError(
            f"spec_version 1 requires every section; missing {', '.join(missing)}"
        )


def _at_path(spec: ConceptSourceSpec, path: str) -> Any:
    owner: Any = spec
    for part in filter(None, path.split(".")):
        owner = getattr(owner, part, None)
        if owner is None:
            return None
    return owner


def _check_single_hierarchy(spec: ConceptSourceSpec) -> None:
    relationships = spec.relationships
    if spec.hierarchy is None or relationships is None:
        return
    if relationships.roles.hierarchical is not None:
        raise ValueError(
            "hierarchy is declared twice: `hierarchy` and "
            "`relationships.roles.hierarchical`; keep one"
        )


def _check_scope_providers(spec: ConceptSourceSpec) -> None:
    present = [name for name in SCOPED_SECTIONS if getattr(spec, name) is not None]
    if spec.concepts is None:
        if present:
            raise ValueError(f"{', '.join(present)} need a `concepts` section")
        return
    expected = spec.concepts.relation.provider
    for name in present:
        provider = getattr(spec, name).relation.provider
        if provider != expected:
            raise ValueError(
                f"{name} resolves to provider {provider!r} but concepts resolves to "
                f"{expected!r}; sections filtered to the concept scope must use the "
                "concepts provider"
            )


def load_concept_source_spec(path: Path) -> ConceptSourceSpec:
    raw: Any = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict) or "concept_source" not in raw:
        raise ValueError(f"{path}: expected a top-level 'concept_source' mapping")
    return ConceptSourceSpec.model_validate(
        raw["concept_source"], context={"base_dir": Path(path).parent}
    )


def load_dataset_map(path: Path) -> dict[str, str]:
    raw: Any = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a mapping of dataset label or id to table")
    return {str(key): str(value) for key, value in raw.items()}


__all__ = [
    "AuthoredBindingRoles",
    "AuthoredBindingsSpec",
    "ConceptRoles",
    "ConceptSourceSpec",
    "ConceptsSpec",
    "DetailRoles",
    "DetailsSpec",
    "EndpointFallback",
    "EndpointRole",
    "EntityModelRoles",
    "EntityModelSpec",
    "ExternalRefRole",
    "HierarchySpec",
    "LevelColumnsHierarchy",
    "LifecycleRoles",
    "MappingRoles",
    "MappingsSpec",
    "ParentColumnHierarchy",
    "PathCodesHierarchy",
    "Provider",
    "RelationRef",
    "RelationshipRoles",
    "RelationshipsSpec",
    "SchemeSpec",
    "ShapeEntry",
    "StatusRole",
    "TermStorageSpec",
    "ValueMappingRoles",
    "ValueMappingsSpec",
    "ValueSetRoles",
    "ValueSetsSpec",
    "SUPPORTED_SPEC_VERSIONS",
    "VERSION_2_FIELDS",
    "load_concept_source_spec",
    "load_dataset_map",
]

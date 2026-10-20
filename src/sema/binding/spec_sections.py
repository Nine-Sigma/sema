"""Section and role models of the concept source spec.

Role values are column names in the named relation. ``filter`` strings are SQL
``WHERE`` fragments written by the operator and appended verbatim.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

Provider = Literal["warehouse", "staging"]
DEFAULT_RELATIONSHIP_TYPE = "BROADER"


class _SpecModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RelationRef(_SpecModel):
    provider: Provider
    name: str = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def _bare_name(cls, data: Any) -> Any:
        # The spec-level validator resolves strings with default_provider first.
        return {"provider": "warehouse", "name": data} if isinstance(data, str) else data


class SchemeSpec(_SpecModel):
    qualified_name: str = Field(min_length=1)
    description: str = Field(min_length=1)


class StatusRole(_SpecModel):
    column: str
    active_values: list[str]
    retired_values: list[str]


class LifecycleRoles(_SpecModel):
    status: StatusRole | None = None
    valid_from: str | None = None
    valid_to: str | None = None
    replaced_by: str | None = None
    lang: str | None = None


class ConceptRoles(LifecycleRoles):
    concept_id: str
    name: str
    title: str | None = None
    definition: str | None = None
    data_type: str | None = None
    sensitive: str | None = None
    synonyms: str | None = None


class ConceptsSpec(_SpecModel):
    relation: RelationRef
    group_by: str
    roles: ConceptRoles
    filter: str | None = None


class ValueSetRoles(LifecycleRoles):
    concept_id: str
    code: str
    display: str
    display_fallback: str | None = None
    vocabulary_id: str | None = None
    concept_code: str | None = None
    synonyms: str | None = None


class ValueSetsSpec(_SpecModel):
    relation: RelationRef
    roles: ValueSetRoles


class ExternalRefRole(_SpecModel):
    """``system`` is a literal, or ``{column}`` to read the system per row."""

    system: str
    column: str


class DetailRoles(_SpecModel):
    definition: str | None = None
    external_refs: list[ExternalRefRole] = Field(default_factory=list)


class DetailsSpec(_SpecModel):
    """``on`` is [column in the source relation, column in this relation]."""

    relation: RelationRef
    on: list[str] = Field(min_length=2, max_length=2)
    roles: DetailRoles

    @model_validator(mode="before")
    @classmethod
    def _yaml_on_key(cls, data: Any) -> Any:
        # YAML 1.1 reads the bare key `on` as boolean True.
        if isinstance(data, dict) and True in data:
            data = {
                ("on" if key is True else key): value for key, value in data.items()
            }
        return data


class EndpointFallback(_SpecModel):
    system: str
    id: str


class EndpointRole(_SpecModel):
    system: str
    id: str
    fallback: EndpointFallback | None = None


class RelationshipRoles(_SpecModel):
    from_: EndpointRole = Field(alias="from")
    to: EndpointRole
    type: str
    hierarchical: str | None = None


class RelationshipsSpec(_SpecModel):
    relation: RelationRef
    roles: RelationshipRoles
    filter: str | None = None


class ParentColumnHierarchy(_SpecModel):
    """``relation`` and ``child_column`` default to the concepts relation and id."""

    parent_column: str
    child_column: str | None = None
    relation: RelationRef | None = None
    relationship_type: str = DEFAULT_RELATIONSHIP_TYPE


class LevelColumnsHierarchy(_SpecModel):
    level_columns: list[str] = Field(min_length=2)
    relation: RelationRef | None = None
    relationship_type: str = DEFAULT_RELATIONSHIP_TYPE


class PathCodesHierarchy(_SpecModel):
    path_column: str
    separator: str = Field(min_length=1)
    relation: RelationRef | None = None
    relationship_type: str = DEFAULT_RELATIONSHIP_TYPE


HierarchySpec = ParentColumnHierarchy | LevelColumnsHierarchy | PathCodesHierarchy


class AuthoredBindingRoles(_SpecModel):
    concept_id: str
    dataset_label: str
    dataset_id: str
    columns: str
    project: str | None = None


class AuthoredBindingsSpec(_SpecModel):
    relation: RelationRef
    roles: AuthoredBindingRoles
    dataset_map: Path

    @field_validator("dataset_map")
    @classmethod
    def _relative_to_spec(cls, value: Path, info: ValidationInfo) -> Path:
        base_dir = (info.context or {}).get("base_dir")
        return value if base_dir is None else Path(base_dir) / value


class ValueMappingRoles(_SpecModel):
    concept_id: str
    source_value: str
    code: str
    status: str


class ValueMappingsSpec(_SpecModel):
    relation: RelationRef
    roles: ValueMappingRoles
    approved_statuses: list[str] = Field(default_factory=list)


class MappingRoles(_SpecModel):
    subject_id: str
    predicate_id: str
    object_id: str
    mapping_justification: str
    subject_label: str | None = None
    object_label: str | None = None
    confidence: str | None = None


class MappingsSpec(_SpecModel):
    """SSSOM-shaped rows; subject and object values are CURIEs."""

    relation: RelationRef
    roles: MappingRoles
    exact_predicates: list[str] = Field(default_factory=lambda: ["skos:exactMatch"])
    filter: str | None = None


class TermStorageSpec(_SpecModel):
    inline_cap: int = Field(default=5000, ge=0)
    external_namespace: str | None = None


class EntityModelRoles(_SpecModel):
    entity: str
    property: str
    type: str
    required: str | None = None
    primary_key: str | None = None
    fk_entity: str | None = None
    fk_property: str | None = None
    domain: str | None = None
    description: str | None = None
    entity_description: str | None = None


class EntityModelSpec(_SpecModel):
    """``entity_case: lower`` lowercases entity, property, and foreign key names."""

    relation: RelationRef
    roles: EntityModelRoles
    entity_case: Literal["preserve", "lower"] = "preserve"
    filter: str | None = None


class ShapeEntry(_SpecModel):
    section: str
    template: str

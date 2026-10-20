"""Concept source spec: which relations hold a concept library and their column roles.

Role values are column names in the named relation. ``filter`` strings are SQL
``WHERE`` fragments written by the operator and appended verbatim.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)


class _SpecModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SchemeSpec(_SpecModel):
    qualified_name: str = Field(min_length=1)
    description: str = Field(min_length=1)


class ConceptRoles(_SpecModel):
    concept_id: str
    name: str
    title: str | None = None
    definition: str | None = None
    data_type: str | None = None
    sensitive: str | None = None
    synonyms: str | None = None


class ConceptsSpec(_SpecModel):
    relation: str
    group_by: str
    roles: ConceptRoles
    filter: str | None = None


class ValueSetRoles(_SpecModel):
    concept_id: str
    code: str
    display: str
    display_fallback: str | None = None
    vocabulary_id: str | None = None
    concept_code: str | None = None
    synonyms: str | None = None


class ValueSetsSpec(_SpecModel):
    relation: str
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

    relation: str
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
    relation: str
    roles: RelationshipRoles
    filter: str | None = None


class AuthoredBindingRoles(_SpecModel):
    concept_id: str
    dataset_label: str
    dataset_id: str
    columns: str
    project: str | None = None


class AuthoredBindingsSpec(_SpecModel):
    relation: str
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
    relation: str
    roles: ValueMappingRoles
    approved_statuses: list[str] = Field(default_factory=list)


class ConceptSourceSpec(_SpecModel):
    id: str = Field(min_length=1)
    identity_system: str = Field(min_length=1)
    release: str = Field(min_length=1)
    scheme: SchemeSpec
    concepts: ConceptsSpec
    value_sets: ValueSetsSpec
    concept_details: DetailsSpec
    term_details: DetailsSpec
    relationships: RelationshipsSpec
    authored_bindings: AuthoredBindingsSpec
    value_mappings: ValueMappingsSpec


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

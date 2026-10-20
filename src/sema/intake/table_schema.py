"""A Frictionless Table Schema subset, and its projection to a manifest entity.

The models carry the Frictionless field names (``primaryKey``, ``foreignKeys``)
plus the ``x-sema-*`` extension keys, so ``descriptor()`` validates against the
published Table Schema JSON Schema. The projection states only what a
dictionary states: no vocabulary binding, synonym, resolver policy, or
completeness annotation, and only natural or compound key strategies.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict
from pydantic import Field as ModelField

from sema.log import logger

FrictionlessType = Literal[
    "string", "number", "integer", "boolean", "date", "datetime", "any"
]
TABLE_ROW_KIND = "TABLE_ROW"


def _aliased(name: str, key: str) -> Any:
    """Constructed by field name; dumped and parsed under the Frictionless key."""
    return ModelField(
        default=None,
        validation_alias=AliasChoices(name, key),
        serialization_alias=key,
    )


class _Descriptor(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Constraints(_Descriptor):
    required: bool | None = None


class Reference(_Descriptor):
    resource: str
    fields: list[str] = ModelField(min_length=1)


class ForeignKey(_Descriptor):
    fields: list[str] = ModelField(min_length=1)
    reference: Reference


class Field(_Descriptor):
    name: str = ModelField(min_length=1)
    type: FrictionlessType
    description: str | None = None
    constraints: Constraints | None = None
    x_sema_raw_type: str | None = _aliased("x_sema_raw_type", "x-sema-raw-type")
    x_sema_domain: str | None = _aliased("x_sema_domain", "x-sema-domain")
    x_sema_sensitive: bool | None = _aliased("x_sema_sensitive", "x-sema-sensitive")

    @property
    def required(self) -> bool:
        return bool(self.constraints and self.constraints.required)


class TableSchema(_Descriptor):
    fields: list[Field] = ModelField(min_length=1)
    primary_key: list[str] | None = _aliased("primary_key", "primaryKey")
    foreign_keys: list[ForeignKey] | None = _aliased("foreign_keys", "foreignKeys")
    description: str | None = None

    def descriptor(self) -> dict[str, Any]:
        dumped: dict[str, Any] = self.model_dump(by_alias=True, exclude_none=True)
        return dumped


def to_manifest_entity(
    schema: TableSchema,
    qualified_name: str,
    *,
    namespace: str,
    card_version: str = "1",
) -> dict[str, Any]:
    entity: dict[str, Any] = {
        "qualified_name": qualified_name,
        "kind": TABLE_ROW_KIND,
        "properties": [_property(field) for field in schema.fields],
    }
    obligation = _obligation(schema, qualified_name, namespace)
    if obligation is not None:
        entity["obligation"] = obligation
    if schema.description:
        entity["context_card"] = {
            "card_version": card_version,
            "description": schema.description,
        }
    return entity


def _property(field: Field) -> dict[str, Any]:
    prop: dict[str, Any] = {
        "name": field.name,
        "type": field.type,
        "nullable": not field.required,
    }
    if field.description:
        prop["definition"] = field.description
    return prop


def _obligation(
    schema: TableSchema, qualified_name: str, namespace: str
) -> dict[str, Any] | None:
    required = [field.name for field in schema.fields if field.required]
    if not required:
        logger.info("Entity {} has no required field; no obligation", qualified_name)
        return None
    keys = schema.primary_key or []
    foreign_keys = schema.foreign_keys or []
    if not keys:
        logger.info(
            "Entity {} has no primary key stated; no obligation, its {} required "
            "fields stay non-nullable and its {} foreign keys are not carried",
            qualified_name,
            len(required),
            len(foreign_keys),
        )
        return None
    obligation: dict[str, Any] = {
        "required_fields": required,
        "primary_key": "NATURAL_KEY" if len(keys) == 1 else "COMPOUND",
    }
    if foreign_keys:
        obligation["foreign_keys"] = [_foreign_key(fk, namespace) for fk in foreign_keys]
    domains = [
        {"property_name": field.name, "domain_id": field.x_sema_domain}
        for field in schema.fields
        if field.x_sema_domain
    ]
    if domains:
        obligation["domain_constraints"] = domains
    return obligation


def _foreign_key(foreign_key: ForeignKey, namespace: str) -> dict[str, Any]:
    pairs = zip(foreign_key.fields, foreign_key.reference.fields, strict=True)
    return {
        "referenced_entity": f"{namespace}.{foreign_key.reference.resource}",
        "join_keys": [[local, remote] for local, remote in pairs],
    }

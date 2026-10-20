"""Entity model section: dictionary rows to Table Schemas to manifest entities."""

from __future__ import annotations

from sema.binding.importer_utils import Record, as_text, fetch, group_by_key
from sema.binding.spec import ConceptSourceSpec, EntityModelSpec
from sema.log import logger
from sema.intake.relations import RelationProvider
from sema.intake.table_schema import to_manifest_entity
from sema.intake.table_schema_utils import (
    DictionaryRow,
    dictionary_flag,
    table_schema_from_rows,
)


def entity_model_entities(
    spec: ConceptSourceSpec, model: EntityModelSpec, provider: RelationProvider
) -> list[Record]:
    roles = model.roles
    rows = fetch(
        provider,
        model.relation,
        [
            roles.entity,
            roles.property,
            roles.type,
            roles.required,
            roles.primary_key,
            roles.fk_entity,
            roles.fk_property,
            roles.domain,
            roles.description,
            roles.entity_description,
        ],
        model.filter,
    )
    entities: list[Record] = []
    for entity, group in sorted(group_by_key(rows, roles.entity).items()):
        name = _cased(entity, model)
        dictionary = [row for record in group if (row := _row(record, model))]
        schema = table_schema_from_rows(
            name, dictionary, description=_entity_description(name, group, model)
        )
        entities.append(
            to_manifest_entity(
                schema,
                f"{spec.identity_system}.{name}",
                namespace=spec.identity_system,
                card_version=spec.release,
            )
        )
    return entities


def _row(record: Record, model: EntityModelSpec) -> DictionaryRow | None:
    roles = model.roles
    prop = as_text(record[roles.property])
    if prop is None:
        return None
    fk_entity = _text(record, roles.fk_entity)
    fk_property = _text(record, roles.fk_property)
    return DictionaryRow(
        property=_cased(prop, model),
        raw_type=as_text(record[roles.type]) or "",
        required=dictionary_flag(record[roles.required]) if roles.required else False,
        primary_key=dictionary_flag(record[roles.primary_key])
        if roles.primary_key
        else False,
        fk_entity=_cased(fk_entity, model) if fk_entity else None,
        fk_property=_cased(fk_property, model) if fk_property else None,
        domain=_text(record, roles.domain),
        description=_text(record, roles.description),
    )


def _entity_description(
    entity: str, group: list[Record], model: EntityModelSpec
) -> str | None:
    column = model.roles.entity_description
    if column is None:
        return None
    found = sorted({text for record in group if (text := as_text(record[column]))})
    if len(found) > 1:
        logger.info(
            "Entity {} has {} distinct {} values; the first is its description",
            entity,
            len(found),
            column,
        )
    return found[0] if found else None


def _text(record: Record, column: str | None) -> str | None:
    return as_text(record[column]) if column else None


def _cased(name: str, model: EntityModelSpec) -> str:
    return name.lower() if model.entity_case == "lower" else name

"""Entity model section: dictionary rows to Table Schemas to manifest entities."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from sema.binding.identifier_utils import card_version, entity_identifier
from sema.binding.importer_utils import (
    Record,
    as_text,
    fetch,
    fetch_in_order,
    group_by_key,
)
from sema.binding.spec import (
    ConceptSourceSpec,
    EntityModelSpec,
    HeaderKeys,
    PerFileEntry,
    RelationRef,
)
from sema.intake.landing import LOCATOR_COLUMN, SHEET_SEPARATOR
from sema.intake.lexicon import HEADER_KEY_TOKENS, normalize
from sema.intake.relations import RelationProvider
from sema.intake.table_schema import ForeignKey, Reference, TableSchema, to_manifest_entity
from sema.intake.table_schema_utils import (
    DictionaryRow,
    dictionary_flag,
    inline_values,
    table_schema_from_rows,
)
from sema.log import logger
from sema.models.target.identity import EntityIdentity

HeaderPair = tuple[str, str | None]
SourceNames = dict[str, str]


@dataclass(frozen=True)
class Header:
    pairs: list[HeaderPair]
    name: str | None
    description: str | None
    prose: dict[str, str | None]


@dataclass(frozen=True)
class EntityTables:
    """Keyed by qualified name; ``labels`` holds the source name of each converted one."""

    schemas: dict[str, TableSchema] = field(default_factory=dict)
    labels: dict[str, str] = field(default_factory=dict)
    identities: dict[str, EntityIdentity] = field(default_factory=dict)


def entity_table_schemas(
    spec: ConceptSourceSpec, model: EntityModelSpec, provider: RelationProvider
) -> EntityTables:
    if model.layout == "per_file":
        tables = _per_file_schemas(spec, model, provider)
    else:
        tables = _single_relation_schemas(spec, model, provider)
    if model.roles.type is None:
        for qualified_name, schema in tables.schemas.items():
            logger.info(
                "Entity {}: no type role, so its {} fields are typed any",
                qualified_name,
                len(schema.fields),
            )
    return tables


def entity_records(spec: ConceptSourceSpec, tables: EntityTables) -> list[Record]:
    version = card_version(spec.release)
    records = []
    for qualified_name, schema in tables.schemas.items():
        record = to_manifest_entity(
            schema, qualified_name, namespace=spec.identity_system, card_version=version
        )
        label = tables.labels.get(qualified_name)
        if label is not None:
            _add_label(record, label, version)
        identity = tables.identities.get(qualified_name)
        if identity is not None:
            record["identity"] = identity.model_dump(mode="json", exclude_defaults=True)
        records.append(record)
    return records


def read_header(
    provider: RelationProvider, ref: RelationRef, keys: HeaderKeys | None
) -> Header:
    """Key-value pairs: the two column names first, then each row in landed order.

    A key not stated in ``keys`` is the first key in that order holding one of
    the lexicon's header tokens; keys that lose that tie stay prose.
    """
    columns = [name for name in provider.columns(ref) if name != LOCATOR_COLUMN]
    if len(columns) < 2:
        raise ValueError(f"header relation {ref.name} needs a key and a value column")
    key_column, value_column = columns[:2]
    rows = fetch_in_order(provider, ref, [key_column, value_column])
    return header_of(
        [(key_column, value_column)],
        ((record[key_column], record[value_column]) for record in rows),
        keys,
    )


def header_of(
    first: list[HeaderPair], rows: Iterable[tuple[Any, Any]], keys: HeaderKeys | None
) -> Header:
    """The reading rule over pairs already read in landed order; blank keys are skipped."""
    pairs = list(first)
    for raw_key, raw_value in rows:
        key = as_text(raw_key)
        if key is not None:
            pairs.append((key, as_text(raw_value)))
    stated = keys or HeaderKeys()
    name_key = _header_key(pairs, stated.name, HEADER_KEY_TOKENS["name"], set())
    taken = {name_key} - {None}
    description_key = _header_key(
        pairs, stated.description, HEADER_KEY_TOKENS["description"], taken
    )
    bound = {name_key, description_key}
    listed = {normalize(key) for key in stated.text}
    prose = {
        key: value
        for key, value in pairs
        if (normalize(key) in listed if listed else key not in bound)
    }
    return Header(
        pairs=pairs,
        name=_value_of(pairs, name_key),
        description=_value_of(pairs, description_key),
        prose=prose,
    )


def file_stem(relation: str) -> str:
    return relation.rsplit(".", 1)[-1].rsplit(SHEET_SEPARATOR, 1)[0]


def _header_key(
    pairs: list[HeaderPair],
    stated: str | None,
    tokens: frozenset[str],
    taken: set[str | None],
) -> str | None:
    if stated is not None:
        target = normalize(stated)
        return next((key for key, _ in pairs if normalize(key) == target), None)
    wanted = {normalize(token) for token in tokens}
    return next(
        (
            key
            for key, _ in pairs
            if key not in taken and wanted & set(normalize(key).split())
        ),
        None,
    )


def _value_of(pairs: list[HeaderPair], key: str | None) -> str | None:
    return next((value for found, value in pairs if found == key), None)


def _single_relation_schemas(
    spec: ConceptSourceSpec, model: EntityModelSpec, provider: RelationProvider
) -> EntityTables:
    roles = model.roles
    if model.relation is None or roles.entity is None:
        raise ValueError(f"spec {spec.id}: single_relation needs a relation and entity")
    rows = fetch(provider, model.relation, _role_columns(model), model.filter)
    tables = EntityTables()
    sources: SourceNames = {}
    for entity, group in sorted(group_by_key(rows, roles.entity).items()):
        name = _entity_name(entity, model)
        qualified_name = _claim(spec, tables, sources, entity, model)
        dictionary = [row for record in group if (row := _row(record, model, {}))]
        tables.schemas[qualified_name] = table_schema_from_rows(
            name, dictionary, description=_entity_description(name, group, model)
        )
    return tables


def _per_file_schemas(
    spec: ConceptSourceSpec, model: EntityModelSpec, provider: RelationProvider
) -> EntityTables:
    tables = EntityTables()
    sources: SourceNames = {}
    for entry in model.entities or []:
        source, schema = _per_file_schema(model, entry, provider)
        qualified_name = _claim(spec, tables, sources, source, model)
        tables.schemas[qualified_name] = schema
        if entry.identity is not None:
            tables.identities[qualified_name] = entry.identity
    return tables


def _claim(
    spec: ConceptSourceSpec,
    tables: EntityTables,
    sources: SourceNames,
    source: str,
    model: EntityModelSpec,
) -> str:
    """The qualified name ``source`` takes; two sources taking one name fail the import."""
    cased = _cased(source, model)
    name = entity_identifier(cased)
    qualified_name = f"{spec.identity_system}.{name}"
    taken = sources.get(qualified_name)
    if taken is not None:
        raise ValueError(
            f"spec {spec.id}: entity names {taken!r} and {source!r} both become "
            f"entity {qualified_name!r}; one import needs distinct entity names"
        )
    sources[qualified_name] = source
    if name != cased:
        tables.labels[qualified_name] = source
    return qualified_name


def _add_label(record: Record, label: str, version: str) -> None:
    """A renamed entity always keeps its source name; with no description, that name is it."""
    card = record.get("context_card")
    if card is None:
        logger.info(
            "Entity {} states no description; its card describes it by its source name {!r}",
            record["qualified_name"],
            label,
        )
        card = record["context_card"] = {"card_version": version, "description": label}
    card["curated_synonyms"] = [label]


def _per_file_schema(
    model: EntityModelSpec, entry: PerFileEntry, provider: RelationProvider
) -> tuple[str, TableSchema]:
    """The entity's source name, and its Table Schema."""
    header = read_header(provider, entry.header, model.header_keys) if entry.header else None
    named = entry.entity or (header.name if header else None)
    source = named or file_stem(entry.relation.name)
    name = _entity_name(source, model)
    rows = fetch_in_order(provider, entry.relation, _role_columns(model), model.filter)
    dictionary = [row for record in rows if (row := _row(record, model, entry.domains))]
    description = header.description if header else None
    declared = [
        ForeignKey(
            fields=[_cased(column, model) for column in fk.columns],
            reference=Reference(
                resource=_entity_name(fk.entity, model),
                fields=[_cased(column, model) for column in fk.reference_columns],
            ),
        )
        for fk in entry.foreign_keys
    ]
    schema = table_schema_from_rows(
        name,
        dictionary,
        description=description or _entity_description(name, rows, model),
        foreign_keys=declared,
    )
    return source, schema


def _role_columns(model: EntityModelSpec) -> list[str | None]:
    roles = model.roles
    return [
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
        *roles.notes,
    ]


def _row(
    record: Record, model: EntityModelSpec, domains: dict[str, str]
) -> DictionaryRow | None:
    roles = model.roles
    prop = as_text(record[roles.property])
    if prop is None:
        return None
    fk_entity = _text(record, roles.fk_entity)
    fk_property = _text(record, roles.fk_property)
    return DictionaryRow(
        property=_cased(prop, model),
        raw_type=(as_text(record[roles.type]) or "") if roles.type else None,
        required=dictionary_flag(record[roles.required]) if roles.required else False,
        primary_key=dictionary_flag(record[roles.primary_key])
        if roles.primary_key
        else False,
        fk_entity=_entity_name(fk_entity, model) if fk_entity else None,
        fk_property=_cased(fk_property, model) if fk_property else None,
        domain=domains.get(prop) or _text(record, roles.domain),
        description=_text(record, roles.description),
        values=_inline_values(record, [roles.description, *roles.notes]),
    )


def _inline_values(record: Record, columns: list[str | None]) -> dict[str, str] | None:
    """The first text, description before notes, that states a code list."""
    for column in columns:
        text = _text(record, column)
        values = inline_values(text) if text else None
        if values:
            return values
    return None


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


def _entity_name(name: str, model: EntityModelSpec) -> str:
    return entity_identifier(_cased(name, model))

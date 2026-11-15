"""Enrichment helpers: subjects from bound relations, the engine's client, and the entries.

A subject is what the engine sees as one table: an imported entity (its dictionary rows as
columns) or any other bound relation (its landed columns). The engine sanitizes the column
names it reads back, which turns a spaced name into an empty one, so the client keeps each
result's raw name and its call key, and the entries are written under the real names.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict

from sema.binding.importer_entity_utils import entity_table_schemas
from sema.binding.spec import ConceptSourceSpec, RelationRef
from sema.engine.stage_utils import sanitize_column_name
from sema.intake.roles.eligibility_spec_utils import RolesResult
from sema.intake.land.landing import LOCATOR_COLUMN
from sema.intake.roles.lexicon import KEY_ROLE
from sema.intake.calls.model_calls import ModelCalls
from sema.intake.land.profile import SAMPLE_CAP, ProfileTools
from sema.intake.land.profile_utils import ValueSample
from sema.log import logger
from sema.models.stages import StageBColumnResult, StageBResult, StageCResult

ENRICH_STAGE = "enrich"
ADDED: Final = "added"
ALTERNATIVE: Final = "alternative"
DESCRIPTION: Final = "description"
SEMANTIC_TYPE: Final = "semantic_type"
SYNONYMS: Final = "synonyms"
DECODED_VALUES: Final = "decoded_values"
PERCENT = 100.0

EnrichedField = Literal["description", "semantic_type", "synonyms", "decoded_values"]
EntryKind = Literal["added", "alternative"]
EntryValue = str | list[str] | dict[str, str]
Versions = Mapping[type[BaseModel], tuple[type[BaseModel], str]]


class EnrichmentEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    entity: str
    property: str
    field: EnrichedField
    value: EntryValue
    source: Literal["model"] = "model"
    confidence: float
    call_key: str
    kind: EntryKind


class Enrichment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    entries: tuple[EnrichmentEntry, ...] = ()


@dataclass(frozen=True)
class Subject:
    """``stated`` maps each property to the enriched fields its source already states."""

    name: str
    key: tuple[str, ...]
    columns: tuple[dict[str, Any], ...]
    stated: dict[str, frozenset[str]] = field(default_factory=dict)

    def table_metadata(self) -> dict[str, Any]:
        return {"table_name": self.name, "table_ref": self.name, "columns": list(self.columns)}

    def property_of(self, raw: str) -> str | None:
        """The real property a model-returned name stands for; None when it names none."""
        names = [column["name"] for column in self.columns]
        if raw in names:
            return raw
        clean = sanitize_column_name(raw)
        matches = [name for name in names if clean and sanitize_column_name(name) == clean]
        return matches[0] if len(matches) == 1 else None


class EngineClient:
    """The engine's client: each invoke is one intake call under a versioned prompt and model.

    Every returned column result is remembered with its raw name and call key, because the
    engine rewrites the name in place after the call returns.
    """

    def __init__(self, calls: ModelCalls, versions: Versions) -> None:
        self._calls = calls
        self._versions = versions
        self._origins: list[tuple[BaseModel, str, str]] = []

    def invoke(self, prompt: str, output_model: type[BaseModel], **_context: Any) -> BaseModel:
        versioned_model, version_line = self._versions[output_model]
        versioned = f"{version_line}\n{prompt}"
        result = self._calls.call(ENRICH_STAGE, versioned, versioned_model)
        key = self._calls.call_key(versioned, versioned_model)
        columns = getattr(result, "columns", None)
        items: list[Any] = [result] if columns is None else columns
        self._origins += [(item, key, item.column) for item in items]
        return result

    def origin(self, item: BaseModel) -> tuple[str, str] | None:
        """The call key and raw column name of a result this client returned."""
        found = (o for o in self._origins if o[0] is item)
        match = next(found, None)
        return (match[1], match[2]) if match else None


def enrichment_subjects(
    spec: ConceptSourceSpec, roles: RolesResult, tools: ProfileTools
) -> list[Subject]:
    return entity_subjects(spec, tools) + relation_subjects(spec, roles, tools)


def entity_subjects(spec: ConceptSourceSpec, tools: ProfileTools) -> list[Subject]:
    """Each imported entity, named as the manifest names it, its dictionary rows as columns."""
    model = spec.entity_model
    if model is None:
        return []
    tables = entity_table_schemas(spec, model, tools.provider)
    subjects = []
    for name, schema in tables.schemas.items():
        columns = tuple(
            _column(f.name, f.x_sema_raw_type or f.type, f.description) for f in schema.fields
        )
        stated = {f.name: _stated(f.description, f.x_sema_values) for f in schema.fields}
        subjects.append(Subject(name, tuple(schema.primary_key or ()), columns, stated))
    return subjects


def relation_subjects(
    spec: ConceptSourceSpec, roles: RolesResult, tools: ProfileTools
) -> list[Subject]:
    """Every bound relation the entity model does not read, keyed by its key-role column."""
    taken = entity_model_relations(spec)
    keys = {
        e.relation: e.choice
        for e in roles.eligibilities
        if e.role == KEY_ROLE and e.subject is None and e.choice
    }
    subjects: list[Subject] = []
    for ref in spec.relation_refs():
        if ref.name in taken or any(s.name == ref.name for s in subjects):
            continue
        key = keys.get(ref.name)
        subjects.append(_relation_subject(ref, (key,) if key else (), tools))
    return subjects


def entity_model_relations(spec: ConceptSourceSpec) -> set[str]:
    model = spec.entity_model
    if model is None:
        return set()
    refs = [model.relation] + [r for e in model.entities or [] for r in (e.relation, e.header)]
    return {ref.name for ref in refs if ref is not None}


def _relation_subject(ref: RelationRef, key: tuple[str, ...], tools: ProfileTools) -> Subject:
    profile = tools.describe_relation(ref)
    columns: list[dict[str, Any]] = []
    for column in profile.columns:
        if column.name == LOCATOR_COLUMN:
            continue
        described = _column(column.name, column.type, column.comment)
        described["null_pct"] = round(column.null_rate * PERCENT, 1)
        described["distinct_count"] = column.distinct_count
        values = _top_values(ref, column.name, tools)
        if values:
            described["top_values"] = values
        columns.append(described)
    stated = {c["name"]: _stated(c.get("comment"), None) for c in columns}
    return Subject(ref.name, key, tuple(columns), stated)


def _top_values(ref: RelationRef, column: str, tools: ProfileTools) -> list[dict[str, Any]]:
    """Values only where the evidence policy allows them; never for a sensitive column."""
    if (ref.name, column) in tools.policy.sensitive:
        return []
    sample = tools.sample_values(ref, column, SAMPLE_CAP)
    if not isinstance(sample, ValueSample):
        return []
    return [{"value": value} for value in sample.values if value is not None]


def _column(name: str, data_type: str, comment: str | None) -> dict[str, Any]:
    described: dict[str, Any] = {"name": name, "data_type": data_type}
    if comment:
        described["comment"] = comment
    return described


def _stated(description: str | None, values: Mapping[str, str] | None) -> frozenset[str]:
    fields = {DESCRIPTION} if description else set()
    return frozenset(fields | ({DECODED_VALUES} if values else set()))


def restore_names(stage_b: StageBResult, subject: Subject, client: EngineClient) -> None:
    """Put the real property names back on the engine's results before stage C reads them."""
    for batch in stage_b.batch_results:
        for result in batch.columns:
            origin = client.origin(result)
            real = subject.property_of(origin[1]) if origin else None
            if real is not None:
                result.column = real


def subject_entries(
    subject: Subject,
    stage_b: StageBResult,
    stage_c: Mapping[str, StageCResult],
    client: EngineClient,
) -> list[EnrichmentEntry]:
    entries: dict[tuple[str, str], EnrichmentEntry] = {}
    results: list[StageBColumnResult | StageCResult] = [
        result for batch in stage_b.batch_results for result in batch.columns
    ]
    for result in [*results, *stage_c.values()]:
        origin = client.origin(result)
        prop = subject.property_of(result.column)
        if origin is None or prop is None:
            logger.warning(f"Enrichment of {subject.name}: no property {result.column!r}")
            continue
        for found in _entries(subject, prop, origin[0], result):
            entries.setdefault((found.property, found.field), found)
    return list(entries.values())


def _entries(
    subject: Subject, prop: str, key: str, result: StageBColumnResult | StageCResult
) -> Iterable[EnrichmentEntry]:
    if isinstance(result, StageCResult):
        decoded = {c["raw"]: c["label"] for c in result.decoded_categories if "raw" in c}
        values: list[tuple[EnrichedField, EntryValue]] = [(DECODED_VALUES, decoded)]
        confidence = 1.0 - result.uncertainty
    else:
        values = [
            (DESCRIPTION, result.canonical_property_label),
            (SEMANTIC_TYPE, result.semantic_type),
            (SYNONYMS, list(result.synonyms)),
        ]
        confidence = result.confidence
    stated = subject.stated.get(prop, frozenset())
    for name, value in values:
        if value:
            kind: EntryKind = ALTERNATIVE if name in stated else ADDED
            yield EnrichmentEntry(
                entity=subject.name,
                property=prop,
                field=name,
                value=value,
                confidence=confidence,
                call_key=key,
                kind=kind,
            )

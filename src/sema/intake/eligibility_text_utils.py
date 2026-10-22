"""Roles over bound dictionary text: headers, entity mentions, declared joins, sections.

Every read goes through ``read_schema_text``, so it is issued only for a relation the
structure output binds as a dictionary or header and is logged as a schema relation
(design Decision 18). Matching is over landed names only (design Decision 7).
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, get_args

from pydantic import BaseModel, ConfigDict

from sema.binding.importer_entity_utils import header_of
from sema.binding.importer_utils import as_text, locator_row
from sema.binding.spec import ConceptSourceSpec
from sema.binding.spec_sections import CodeListSpec, ConceptsSpec, HeaderKeys
from sema.intake import lexicon
from sema.intake.eligibility_utils import (
    HUMAN,
    SOURCE_CONFIDENCE,
    DeclaredJoin,
    Hint,
    RoleEligibility,
    StatedChoice,
    contains,
    mentions,
)
from sema.intake.landing import LOCATOR_COLUMN
from sema.intake.loaders.tabular_utils import EMPTY_COLUMN_PREFIX
from sema.intake.profile import ProfileTools
from sema.intake.shapes import concept_list, entity_model
from sema.intake.shapes.base import RelationProfile, RelationSignature, SlotBinding, is_text
from sema.intake.shapes.base_utils import section_of
from sema.intake.structure import StructureResult
from sema.intake.structure_hint_utils import SECTION_SEPARATOR
from sema.intake.verify.base import Ambiguity
from sema.log import logger

STAGE = "roles"
CONCEPTS = section_of(ConceptsSpec)
CODE_LISTS = next(
    name
    for name, field in ConceptSourceSpec.model_fields.items()
    if list[CodeListSpec] in get_args(field.annotation)
)
SECTION_MEMBERS = (CONCEPTS, CODE_LISTS)
SECTION_SUBJECT = "section"
NO_REFERENCE_HINT = "no_reference"
JOIN_WITHOUT_COLUMNS = "declared_join_without_columns"
SENTENCE_BREAK = re.compile(r"[.?!]\s+|[\r\n]+")
VOCABULARY_ROLE, ENTITY_ROLE = (
    next(name for name, kind in lexicon.REFERENCE_ROLES if kind == wanted)
    for wanted in ("vocabulary", "entity")
)


class HeaderReading(BaseModel):
    """``values`` is keyed by the landed key text; ``prose`` names the unbound keys."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    relation: str
    name: str | None
    description: str | None
    values: dict[str, str | None]
    prose: tuple[str, ...]

    @property
    def sentences(self) -> tuple[str, ...]:
        """Each prose key is its own text, since a key boundary is a structural break."""
        texts = (text for key in self.prose if (text := self.values.get(key)))
        return tuple(
            sentence for text in texts for sentence in SENTENCE_BREAK.split(text)
            if sentence.strip()
        )


@dataclass(frozen=True)
class Dictionary:
    relation: str
    header: str | None
    entity: str
    aliases: tuple[str, ...]
    properties: tuple[tuple[str, str], ...]

    @property
    def property_names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.properties)


def schema_rows(
    tools: ProfileTools, profile: RelationProfile, columns: Sequence[str]
) -> list[tuple[Any, ...]]:
    """Rows in landed order: the locator orders them when the loader wrote one."""
    located = any(name == LOCATOR_COLUMN for name, _ in profile.landed.columns)
    wanted = [*columns, LOCATOR_COLUMN] if located else list(columns)
    rows = tools.read_schema_text(profile.ref, wanted)
    if not located:
        return rows
    return [row[:-1] for row in sorted(rows, key=lambda row: locator_row(row[-1]))]


def read_header(
    tools: ProfileTools, profile: RelationProfile, keys: HeaderKeys | None
) -> HeaderReading:
    key_column, value_column = (column.name for column in profile.columns[:2])
    rows = schema_rows(tools, profile, [key_column, value_column])
    header = header_of([(key_column, value_column)], rows, keys)
    values: dict[str, str | None] = {}
    for key, value in header.pairs:
        values.setdefault(key, value)
    return HeaderReading(
        relation=profile.ref.name,
        name=header.name,
        description=header.description,
        values=values,
        prose=tuple(header.prose),
    )


def note_columns(profile: RelationProfile, bound: Iterable[str]) -> tuple[str, ...]:
    """Every text column no other role took; a header cell the loader named is not one."""
    taken = set(bound)
    return tuple(
        column.name
        for column in profile.columns
        if is_text(column)
        and column.name not in taken
        and not column.name.startswith(EMPTY_COLUMN_PREFIX)
    )


def read_dictionary(
    tools: ProfileTools,
    profile: RelationProfile,
    header: HeaderReading | None,
    choices: Mapping[str, str],
    notes: Sequence[str],
) -> Dictionary:
    entity = header.name if header is not None and header.name else profile.stem
    found = Dictionary(profile.ref.name, header.relation if header else None, entity,
                       tuple(dict.fromkeys((entity, profile.stem))), ())
    prop = choices.get(lexicon.ENTITY_PROPERTY_ROLE)
    if prop is None:
        logger.info("dictionary {} has no settled property column; its text is unread",
                    profile.ref.name)
        return found
    texts = [choices[r] for r in lexicon.PROPERTY_TEXT_ROLES if r in choices] + list(notes)
    properties = []
    for row in schema_rows(tools, profile, [prop, *texts]):
        name = as_text(row[0])
        if name is not None:
            properties.append((name, " ".join(filter(None, map(as_text, row[1:])))))
    return Dictionary(found.relation, found.header, entity, found.aliases, tuple(properties))


def vocabulary_of(binding: SlotBinding, profile: RelationProfile) -> str:
    head, _, tail = (binding.section or "").partition(SECTION_SEPARATOR)
    return tail if head == CODE_LISTS and tail else profile.stem


def vocabularies(
    structure: StructureResult, by_name: Mapping[str, RelationProfile]
) -> dict[str, str]:
    """Vocabulary id to its relation, for concept lists that are or may be code lists."""
    found = {}
    for entry in structure.bound:
        binding = entry.binding
        head = (binding.section or "").partition(SECTION_SEPARATOR)[0]
        if binding.template == concept_list.ID and head in ("", CODE_LISTS):
            relation = binding.relations[0]
            found[vocabulary_of(binding, by_name[relation])] = relation
    return found


def reference_roles(
    dictionaries: Sequence[Dictionary], vocabulary_ids: Sequence[str]
) -> list[RoleEligibility]:
    found = []
    for dictionary in dictionaries:
        others = {
            alias: other.entity
            for other in dictionaries
            if other.relation != dictionary.relation
            for alias in other.aliases
        }
        for prop, text in dictionary.properties:
            found.append(_domain(dictionary.relation, prop, text, vocabulary_ids))
            found.append(_fk_entity(dictionary.relation, prop, text, others))
    return found


def _domain(relation: str, prop: str, text: str, names: Sequence[str]) -> RoleEligibility:
    members = (*mentions(text, names), lexicon.NO_REFERENCE)
    base = RoleEligibility(relation=relation, role=VOCABULARY_ROLE, subject=prop, members=members)
    phrases = [
        match.group("name") for pattern in lexicon.REFERENCE_PHRASES
        for match in pattern.finditer(text)
    ]
    named = tuple(dict.fromkeys(name for phrase in phrases for name in mentions(phrase, names)))
    unresolved = tuple(phrase for phrase in phrases if not mentions(phrase, names))
    if unresolved:
        return base.hinted(Hint(source="lexicon", matches=named, unresolved=unresolved))
    if named:
        base = base.hinted(Hint(source="lexicon", matches=named))
    if len(named) == 1:
        return base.settled(named[0], "lexicon")
    return base.settled(lexicon.NO_REFERENCE, "code") if len(members) == 1 else base


def _fk_entity(
    relation: str, prop: str, text: str, aliases: Mapping[str, str]
) -> RoleEligibility:
    named = tuple(dict.fromkeys(aliases[alias] for alias in mentions(text, aliases)))
    members = (*named, lexicon.NO_REFERENCE)
    base = RoleEligibility(relation=relation, role=ENTITY_ROLE, subject=prop, members=members)
    return base.settled(lexicon.NO_REFERENCE, "code") if not named else base


def declared_joins(
    dictionaries: Sequence[Dictionary], headers: Mapping[str, HeaderReading]
) -> tuple[list[DeclaredJoin], list[Ambiguity]]:
    joins: list[DeclaredJoin] = []
    open_joins: list[Ambiguity] = []
    for dictionary in dictionaries:
        header = headers.get(dictionary.header) if dictionary.header else None
        if header is None:
            continue
        for other in dictionaries:
            naming = [s for s in header.sentences if mentions(s, other.aliases)]
            if other.relation == dictionary.relation or not naming:
                continue
            shared = [p for p in dictionary.property_names if p in other.property_names]
            named = tuple(p for p in shared if any(contains(s, p) for s in naming))
            if named:
                joins.append(DeclaredJoin(
                    from_relation=dictionary.relation, from_entity=dictionary.entity,
                    to_relation=other.relation, to_entity=other.entity, columns=named,
                ))
            else:
                open_joins.append(Ambiguity(
                    stage=STAGE, subject=f"{dictionary.relation}.join.{other.entity}",
                    members=tuple(shared), hints=(JOIN_WITHOUT_COLUMNS,),
                ))
    return joins, open_joins


def section_subject(relation: str) -> str:
    return f"{relation}.{SECTION_SUBJECT}"


def resolve_sections(
    structure: StructureResult,
    references: Sequence[RoleEligibility],
    by_name: Mapping[str, RelationProfile],
    choices: Mapping[str, StatedChoice] | None = None,
) -> tuple[StructureResult, list[Ambiguity]]:
    """A concept list is a code list once any property's domain resolves to it; otherwise
    a stated choice for its ``<relation>.section`` subject settles it, and a later concept
    list with the same signature inherits that choice with ``source: memory``."""
    referenced: dict[str, list[str]] = {}
    for role in references:
        if role.role == VOCABULARY_ROLE and role.choice not in (None, lexicon.NO_REFERENCE):
            referenced.setdefault(str(role.choice), []).append(f"{role.relation}.{role.subject}")
    remembered: dict[RelationSignature, StatedChoice] = {}
    bound, open_sections = [], []
    for entry in structure.bound:
        binding = entry.binding
        if binding.template != concept_list.ID or binding.section is not None:
            bound.append(entry)
            continue
        relation = binding.relations[0]
        vocabulary = vocabulary_of(binding, by_name[relation])
        if vocabulary in referenced:
            evidence = {**binding.evidence, "references": float(len(referenced[vocabulary]))}
            section = f"{CODE_LISTS}{SECTION_SEPARATOR}{vocabulary}"
            binding = binding.model_copy(update={"section": section, "evidence": evidence})
            bound.append(entry.model_copy(update={"binding": binding}))
            continue
        listed = Ambiguity(
            stage=STAGE, subject=section_subject(relation), members=SECTION_MEMBERS,
            hints=(NO_REFERENCE_HINT,),
        )
        signature = by_name[relation].signature
        stated = (choices or {}).get(listed.subject) or remembered.get(signature)
        decided = _stated_section(stated, vocabulary)
        if stated is not None and decided is not None:
            remembered.setdefault(signature, _recalled(stated))
            binding = binding.model_copy(update={"section": decided})
            entry = entry.model_copy(update={"binding": binding})
        bound.append(entry)
        open_sections.append(_with_stated_section(listed, stated))
    return structure.model_copy(update={"bound": tuple(bound)}), open_sections


def _recalled(stated: StatedChoice) -> StatedChoice:
    return StatedChoice(
        choice=stated.choice, source="memory", confidence=SOURCE_CONFIDENCE["memory"]
    )


def _stated_section(stated: StatedChoice | None, vocabulary: str) -> str | None:
    if stated is None or stated.choice not in SECTION_MEMBERS:
        return None
    if stated.choice == CODE_LISTS:
        return f"{CODE_LISTS}{SECTION_SEPARATOR}{vocabulary}"
    return stated.choice


def _with_stated_section(listed: Ambiguity, stated: StatedChoice | None) -> Ambiguity:
    """As ``with_stated``: a member settles the entry, a human ``none`` closes it unset."""
    if stated is None:
        return listed
    fields = dict(source=stated.source, confidence=stated.confidence)
    if stated.choice in listed.members:
        return listed.model_copy(update=dict(fields, choice=stated.choice))
    if stated.source == HUMAN and stated.choice == lexicon.NO_REFERENCE:
        return listed.model_copy(update=fields)
    return listed


def is_dictionary_binding(binding: SlotBinding) -> bool:
    return (
        binding.template == entity_model.ID
        and binding.slots.get(entity_model.LAYOUT) == entity_model.PER_FILE
    )

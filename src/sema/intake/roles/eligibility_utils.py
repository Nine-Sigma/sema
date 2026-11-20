"""Role eligibility shapes and the pure rules under them (design Decisions 6, 7, 20).

A role's members are the columns its data fact admits (S5 decisions 2, 3, 18): a key is
unique and non-null, a date or array role takes a date, array, or text column, and every
other role admits every column. Fixture-measured readings are findings on a column, never
removals. Lexicon and mention matching compare whole normalized token sequences. The
embedding rank is a fact only; it settles no role (S5 decision 18b).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from sema.binding.spec_sections import SchemeSpec, TermStorageSpec
from sema.intake.roles import lexicon
from sema.intake.roles.lexicon import RoleEntry, normalize
from sema.intake.structure.shapes.base import RelationSignature, is_text
from sema.intake.land.profile_utils import ColumnProfile
from sema.intake.verify.base import Finding

__all__ = [
    "DETERMINISTIC_SOURCES",
    "DeclaredJoin",
    "Embedder",
    "Held",
    "Hint",
    "RoleEligibility",
    "RoleMemory",
    "RunDescriptor",
    "StatedChoice",
    "claimed_columns",
    "lexicon_hits",
    "mentions",
    "normalize",
    "with_claims",
]

EligibilitySource = Literal["code", "memory", "lexicon", "human", "model"]
HintSource = Literal["memory", "lexicon", "embedding"]
HUMAN: EligibilitySource = "human"
KEY_UNIQUE_NON_NULL = "key_unique_non_null"
DATE_SHAPE = "date_shape"
ARRAY_SHAPE = "array_shape"
CONSTANT = "constant"
DISPLAY_NULL_RATE = "display_null_rate"
DEFINITION_MEAN_LENGTH = "definition_mean_length"
STATUS_CARDINALITY = "status_cardinality"
DETERMINISTIC_SOURCES: tuple[EligibilitySource, ...] = ("code", "lexicon")
STATED_SOURCES: tuple[EligibilitySource, ...] = ("human", "model")
# Ordinal, not calibrated: S3 calibrates confidence against the gold set.
SOURCE_CONFIDENCE: dict[str, float] = {
    "code": 1.0,
    "human": 1.0,
    "memory": 0.95,
    "lexicon": 0.9,
}
ROLE_READINGS: dict[str, frozenset[str]] = {
    "text": frozenset({CONSTANT, DISPLAY_NULL_RATE}),
    "long_text": frozenset({CONSTANT, DEFINITION_MEAN_LENGTH}),
    "low_cardinality": frozenset({STATUS_CARDINALITY}),
    "any": frozenset({CONSTANT}),
}
DATE_TYPE_PREFIXES = ("DATE", "TIMESTAMP")
ARRAY_TYPE_MARKERS = ("[]", "ARRAY", "LIST")
MemoryKey = tuple[str, RelationSignature]


class Hint(BaseModel):
    """One narrowing hint; ``rank`` pairs are ``(column, score)``, best first."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: HintSource
    matches: tuple[str, ...] = ()
    rank: tuple[tuple[str, float], ...] = ()
    margin: float | None = None
    unresolved: tuple[str, ...] = ()


class RoleEligibility(BaseModel):
    """``members`` is the data-fact set; ``exclusions`` maps a member another role took to
    that role. A model or human choice may lie outside the members, for the verifier.
    ``findings`` are the measured readings on the members that the role's kind reads."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    relation: str
    section: str | None = None
    role: str
    subject: str | None = None
    members: tuple[str, ...]
    findings: tuple[Finding, ...] = ()
    hints: tuple[Hint, ...] = ()
    excluded_by: str | None = None
    exclusions: dict[str, str] = {}
    choice: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    source: EligibilitySource | None = None

    def settled(
        self, choice: str, source: EligibilitySource, confidence: float | None = None
    ) -> RoleEligibility:
        """``confidence`` is the stated one; otherwise the source's ordinal value."""
        if confidence is None:
            confidence = SOURCE_CONFIDENCE[source]
        return self.model_copy(update=dict(choice=choice, source=source, confidence=confidence))

    @property
    def column(self) -> str | None:
        """The bound column; a ``none`` choice answers the role with no column."""
        return None if self.choice == lexicon.NO_REFERENCE else self.choice

    def hinted(self, *hints: Hint) -> RoleEligibility:
        return self.model_copy(update={"hints": (*self.hints, *hints)})


class StatedChoice(BaseModel):
    """A choice made outside the predicates (a model answer or a human decision), keyed
    by its ambiguity subject; it settles its role before any hint and is verified."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    choice: str
    source: EligibilitySource
    confidence: float = Field(ge=0.0, le=1.0)


def scope_of(relation: str, section: str | None) -> str:
    """A relation, qualified by the section when one relation feeds two sections."""
    return relation if section is None else f"{relation}.{section}"


def subject_of(entry: RoleEligibility) -> str:
    parts = (scope_of(entry.relation, entry.section), entry.role, entry.subject)
    return ".".join(part for part in parts if part is not None)


def with_stated(
    entry: RoleEligibility, choices: Mapping[str, StatedChoice]
) -> RoleEligibility:
    """A human ``none`` over an entry offering no ``none`` closes it with no choice; any
    other human or model choice settles the entry even outside its members, for the
    verifier to judge. A model ``none`` the entry does not offer, or a rule choice outside
    the members, is dropped."""
    stated = choices.get(subject_of(entry))
    if stated is None:
        return entry
    if stated.choice in entry.members:
        return entry.settled(stated.choice, stated.source, stated.confidence)
    if stated.choice == lexicon.NO_REFERENCE:
        if stated.source != HUMAN:
            return entry
        return entry.model_copy(update=dict(source=stated.source, confidence=stated.confidence))
    if stated.source in STATED_SOURCES:
        return entry.settled(stated.choice, stated.source, stated.confidence)
    return entry


class DeclaredJoin(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    from_relation: str
    from_entity: str
    to_relation: str
    to_entity: str
    columns: tuple[str, ...]
    evidence: Literal["declared"] = "declared"


class RunDescriptor(BaseModel):
    """What no predicate can derive; fixed rules supply the rest. An unstated ``scheme``
    defaults, when a concepts section needs one, to ``<identity_system>.<id>`` described by
    id and release; ``relations`` is the warehouse Land allowlist, each entry required."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    identity_system: str = Field(min_length=1)
    release: str = Field(min_length=1)
    scheme: SchemeSpec | None = None
    term_storage: TermStorageSpec | None = None
    entity_case: Literal["preserve", "lower"] | None = None
    relations: tuple[str, ...] | None = Field(default=None, min_length=1)


class DocumentEmbedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...


class Encoder(Protocol):
    def encode(self, texts: list[str]) -> Any: ...


Embedder = DocumentEmbedder | Encoder


class RoleMemory:
    """Role maps already settled in this run, keyed by template and relation signature."""

    def __init__(self) -> None:
        self._maps: dict[MemoryKey, dict[str, str]] = {}

    def recall(self, key: MemoryKey) -> dict[str, str] | None:
        return self._maps.get(key)

    def remember(self, key: MemoryKey, choices: dict[str, str]) -> None:
        self._maps.setdefault(key, choices)


def claimed_columns(entries: Iterable[RoleEligibility]) -> dict[str, str]:
    """Columns code or the lexicon settled, each mapped to the first role settled on it."""
    claims: dict[str, str] = {}
    for entry in entries:
        if entry.source in DETERMINISTIC_SOURCES and entry.column is not None:
            claims.setdefault(entry.column, entry.role)
    return claims


def with_claims(entry: RoleEligibility, claims: Mapping[str, str]) -> RoleEligibility:
    """An open role lists each column another role claims as excluded, so it is never
    offered; the claim does not narrow the role, which stays open for a choice."""
    if entry.choice is not None or entry.source is not None or entry.subject is not None:
        return entry
    claimed = {
        column: role
        for column, role in claims.items()
        if column in entry.members and column not in entry.exclusions
        and role != entry.role and not lexicon.coincide(role, entry.role)
    }
    if not claimed:
        return entry
    return entry.model_copy(update={"exclusions": {**entry.exclusions, **claimed}})


@dataclass(frozen=True)
class Held:
    """Columns roles took on one side of a relation, in settle order. ``claims`` are the
    columns code or the lexicon settle with no stated choice; a stated choice that takes
    one leaves it to its claimant too, so the verifier sees both roles on it."""

    claims: Mapping[str, str]
    taken: dict[str, str] = field(default_factory=dict)
    stated: set[str] = field(default_factory=set)

    def take(self, entry: RoleEligibility) -> None:
        column = entry.column
        if column is None or column in self.taken:
            return
        self.taken[column] = entry.role
        if entry.source in STATED_SOURCES:
            self.stated.add(column)

    def excluded(self, members: Sequence[str], role: str) -> dict[str, str]:
        """Each member another role took, mapped to that role, unless the two coincide."""
        return {
            column: holder
            for column, holder in self.taken.items()
            if column in members
            and not lexicon.coincide(holder, role)
            and not (column in self.stated and self.claims.get(column) == role)
        }


def _key(column: ColumnProfile, row_count: int) -> bool:
    return row_count > 0 and column.null_rate == 0.0 and column.is_unique


def _date(column: ColumnProfile, row_count: int) -> bool:
    return column.type.upper().startswith(DATE_TYPE_PREFIXES) or is_text(column)


def _array(column: ColumnProfile, row_count: int) -> bool:
    kind = column.type.upper()
    return any(marker in kind for marker in ARRAY_TYPE_MARKERS) or is_text(column)


# The data facts a role's members must satisfy, by predicate kind, with the check each one
# names; every other kind admits every column and its readings are findings.
FACTS: dict[str, tuple[str, Callable[[ColumnProfile, int], bool]]] = {
    "key": (KEY_UNIQUE_NON_NULL, _key),
    "date": (DATE_SHAPE, _date),
    "array": (ARRAY_SHAPE, _array),
}


def fact_of(role: str) -> str | None:
    """The data fact ``role``'s members satisfy, or ``None`` when every column is one."""
    entry = next((e for e in lexicon.ROLES if e.name == role), None)
    found = FACTS.get(entry.predicate) if entry is not None else None
    return None if found is None else found[0]


def eligible(entry: RoleEntry, columns: Iterable[ColumnProfile], row_count: int) -> tuple[str, ...]:
    fact = FACTS.get(entry.predicate)
    return tuple(c.name for c in columns if fact is None or fact[1](c, row_count))


def column_findings(column: ColumnProfile) -> tuple[tuple[str, float | None], ...]:
    """The fixture-measured readings a column fails, each with its value (S5 decision 18)."""
    found: list[tuple[str, float | None]] = []
    if column.is_constant:
        found.append((CONSTANT, None))
    if column.null_rate >= lexicon.DISPLAY_NULL_RATE_FLOOR:
        found.append((DISPLAY_NULL_RATE, column.null_rate))
    mean = column.mean_length
    if is_text(column) and mean is not None and mean < lexicon.DEFINITION_MEAN_LENGTH_FLOOR:
        found.append((DEFINITION_MEAN_LENGTH, mean))
    if column.distinct_count > lexicon.STATUS_CARDINALITY_CAP:
        found.append((STATUS_CARDINALITY, float(column.distinct_count)))
    return tuple(found)


def role_findings(
    entry: RoleEntry, relation: str, columns: Iterable[ColumnProfile], members: Sequence[str]
) -> tuple[Finding, ...]:
    """Each member's readings its role's predicate kind tested before S5, as findings."""
    read = ROLE_READINGS.get(entry.predicate, frozenset())
    return tuple(
        Finding(name=name, subject=f"{relation}.{column.name}", value=value)
        for column in columns if column.name in members
        for name, value in column_findings(column) if name in read
    )


def _padded(text: str) -> str:
    return f" {normalize(text)} "


def contains(text: str, phrase: str) -> bool:
    target = normalize(phrase)
    return bool(target) and f" {target} " in _padded(text)


def lexicon_hits(role: str, names: Iterable[str]) -> tuple[str, ...]:
    tokens = lexicon.role(role).tokens
    return tuple(name for name in names if any(contains(name, token) for token in tokens))


def mentions(text: str, names: Iterable[str]) -> tuple[str, ...]:
    """The names whose normalized token sequence occurs in the normalized text."""
    padded = _padded(text)
    found = []
    for name in names:
        target = normalize(name)
        if target and f" {target} " in padded:
            found.append(name)
    return tuple(dict.fromkeys(found))


def status_classes(values: Sequence[Any]) -> tuple[list[str], list[str]] | None:
    """Active and retired values by the lexicon's status words; None if any is unknown."""
    classes: dict[str, list[str]] = {name: [] for name in lexicon.STATUS_VALUE_WORDS}
    for value in values:
        text = str(value)
        found = [
            name
            for name, words in lexicon.STATUS_VALUE_WORDS.items()
            if any(contains(text, word) for word in words)
        ]
        if len(found) != 1:
            return None
        classes[found[0]].append(text)
    active, retired = (classes[name] for name in lexicon.STATUS_VALUE_WORDS)
    return active, retired


def embed(embedder: Embedder, texts: Sequence[str]) -> list[list[float]]:
    batch = list(texts)
    vectors = (
        embedder.embed_documents(batch) if hasattr(embedder, "embed_documents")
        else embedder.encode(batch)
    )
    return [[float(value) for value in vector] for vector in vectors]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    norm = math.hypot(*a) * math.hypot(*b)
    return sum(x * y for x, y in zip(a, b, strict=True)) / norm if norm else 0.0


def embedding_rank(
    embedder: Embedder, entry: RoleEntry, columns: Sequence[ColumnProfile]
) -> Hint:
    """Column text is name, type, and comment; values are not read here."""
    texts = [" ".join(filter(None, (c.name, c.type, c.comment))) for c in columns]
    role_vector, *vectors = embed(embedder, [entry.description, *texts])
    scored = sorted(
        ((column.name, cosine(vector, role_vector)) for column, vector in zip(
            columns, vectors, strict=True
        )),
        key=lambda pair: -pair[1],
    )
    margin = scored[0][1] - scored[1][1] if len(scored) > 1 else scored[0][1]
    return Hint(source="embedding", rank=tuple(scored), margin=margin)

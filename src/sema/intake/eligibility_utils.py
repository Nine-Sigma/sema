"""Role eligibility shapes and the pure rules under them (design Decisions 6, 7, 20).

Predicates read column aggregates only. Lexicon and mention matching compare whole
normalized token sequences. The embedding rank is optional and settles a role only when
its margin reaches ``EMBEDDING_MARGIN_FLOOR``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from sema.binding.spec_sections import SchemeSpec, TermStorageSpec
from sema.intake import lexicon
from sema.intake.lexicon import RoleEntry, normalize
from sema.intake.shapes.base import RelationSignature, is_text
from sema.intake.profile_utils import ColumnProfile

__all__ = [
    "EMBEDDING_MARGIN_FLOOR",
    "DeclaredJoin",
    "Embedder",
    "Hint",
    "RoleEligibility",
    "RoleMemory",
    "RunDescriptor",
    "StatedChoice",
    "lexicon_hits",
    "mentions",
    "normalize",
]

# Measured 2026-10-21 by scripts/measure_embedding_margin.py over A2, A3, A8, ACRIS (38 role
# entries): one wrong top choice at 0.0893 (A2 display); correct margins 0.0274 to 0.0785.
# Floor = ceil to 0.01 of the widest wrong margin, plus 0.01.
EMBEDDING_MARGIN_FLOOR: float = 0.10
EligibilitySource = Literal["code", "memory", "lexicon", "embedding", "human", "model"]
HintSource = Literal["memory", "lexicon", "embedding"]
HUMAN: EligibilitySource = "human"
# Ordinal, not calibrated: S3 calibrates confidence against the gold set.
SOURCE_CONFIDENCE: dict[str, float] = {
    "code": 1.0,
    "human": 1.0,
    "memory": 0.95,
    "lexicon": 0.9,
    "embedding": 0.8,
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
    """``members`` is the predicate set before narrowing; ``exclusions`` maps a removed
    column to the earlier role that took it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    relation: str
    section: str | None = None
    role: str
    subject: str | None = None
    members: tuple[str, ...]
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
    """A human ``none`` over an entry offering no ``none`` closes it with no choice."""
    stated = choices.get(subject_of(entry))
    if stated is None:
        return entry
    if stated.choice in entry.members:
        return entry.settled(stated.choice, stated.source, stated.confidence)
    if stated.source == HUMAN and stated.choice == lexicon.NO_REFERENCE:
        return entry.model_copy(update=dict(source=stated.source, confidence=stated.confidence))
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
    """What no predicate can derive; everything else follows from fixed rules."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    identity_system: str = Field(min_length=1)
    release: str = Field(min_length=1)
    scheme: SchemeSpec | None = None
    term_storage: TermStorageSpec | None = None
    entity_case: Literal["preserve", "lower"] | None = None


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


def _key(column: ColumnProfile, row_count: int) -> bool:
    return row_count > 0 and column.null_rate == 0.0 and column.is_unique


def _text(column: ColumnProfile, row_count: int) -> bool:
    filled = column.null_rate < lexicon.DISPLAY_NULL_RATE_FLOOR
    return row_count > 0 and is_text(column) and filled and not column.is_constant


def _long_text(column: ColumnProfile, row_count: int) -> bool:
    mean = column.mean_length
    long_enough = mean is not None and mean >= lexicon.DEFINITION_MEAN_LENGTH_FLOOR
    return is_text(column) and long_enough and not column.is_constant


def _low_cardinality(column: ColumnProfile, row_count: int) -> bool:
    return 0 < column.distinct_count <= lexicon.STATUS_CARDINALITY_CAP


def _date(column: ColumnProfile, row_count: int) -> bool:
    return column.type.upper().startswith(DATE_TYPE_PREFIXES)


def _array(column: ColumnProfile, row_count: int) -> bool:
    kind = column.type.upper()
    return any(marker in kind for marker in ARRAY_TYPE_MARKERS)


def _any(column: ColumnProfile, row_count: int) -> bool:
    return not column.is_constant


PREDICATES: dict[str, Callable[[ColumnProfile, int], bool]] = {
    "key": _key,
    "text_key": lambda column, rows: _key(column, rows) and is_text(column),
    "text": _text,
    "long_text": _long_text,
    "low_cardinality": _low_cardinality,
    "date": _date,
    "array": _array,
    "any": _any,
    "filled": lambda column, rows: column.null_rate < 1.0,
}


def eligible(entry: RoleEntry, columns: Iterable[ColumnProfile], row_count: int) -> tuple[str, ...]:
    test = PREDICATES[entry.predicate]
    passing = tuple(column.name for column in columns if test(column, row_count))
    return lexicon_hits(entry.name, passing) if entry.name_gated else passing


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

"""Domain-neutral words the intake rules match, and how text is normalized first.

Normalization is English-only: case fold after splitting case changes,
non-alphanumerics become spaces, whitespace collapses, and a trailing plural
``s`` is stripped from each token.

This is the one intake module that spells role names (design Decision 5). Each role
names its predicate kind, which is also the check id the roles verifier reports. Tokens
are written as one space-separated word list per role and reviewed with every change.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from sema.binding.spec_sections import (
    CodeListRoles,
    ConceptRoles,
    EntityModelRoles,
    ValueSetRoles,
)

LEXICON_VERSION = "s2-2"

PredicateKind = Literal[
    "key", "text_key", "text", "long_text", "low_cardinality", "date", "array", "any"
]
ReferenceKind = Literal["vocabulary", "entity"]

HEADER_KEY_TOKENS: dict[str, frozenset[str]] = {
    "name": frozenset({"name", "title"}),
    "description": frozenset({"description", "summary"}),
}

REFERENCE_PHRASES: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bsee\s+(?P<name>.+?)\s+dataset\b", re.IGNORECASE | re.DOTALL),
)

PATTERNS_VERSION = "inline-values-1"
PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?P<code>[^\s=]+)\s*=\s*(?P<display>[^=\n]+?)(?=\s+[^\s=]+\s*=|\n|$)"),
)

# Measured: A2 Values.label and A3 rdfs:label hold no nulls (0.0).
DISPLAY_NULL_RATE_FLOOR = 0.05
# Measured: the longest S1 label mean is A3 rdfs:label 7.7 (A2 concept_name 6.4); the
# shortest S1 definition mean is syn-library concept.definition 16.6 (A2 definition
# 18.6, A3 IAO:0000115 24.6, syn-library data_element_definition 32.8). The design's
# starting value 40 sat above every definition it named, so the floor sits between.
DEFINITION_MEAN_LENGTH_FLOOR = 12.0
# Measured: A2 Values.status 2 distinct, A3 owl:deprecated 1 distinct.
STATUS_CARDINALITY_CAP = 10


@dataclass(frozen=True)
class RoleEntry:
    name: str
    words: str
    description: str
    predicate: PredicateKind

    @property
    def tokens(self) -> tuple[str, ...]:
        return tuple(self.words.split())

    @property
    def check(self) -> str:
        return self.predicate


# Settled in this order; code precedes the identifier because its words are narrower,
# and a property's description precedes the property because dictionary headers name
# both with the same leading word.
ROLES: tuple[RoleEntry, ...] = (
    RoleEntry("code", "code value", "the short code that identifies a term in its list", "key"),
    RoleEntry(
        "concept_id", "id concept_id identifier", "the stable identifier of a concept", "key"
    ),
    RoleEntry("name", "name label title", "the preferred human readable name", "text"),
    RoleEntry("display", "display label name", "the text shown to people for a code", "text"),
    RoleEntry(
        "definition",
        "definition description desc meaning",
        "a sentence or more explaining what something means",
        "long_text",
    ),
    RoleEntry(
        "description",
        "description definition desc meaning",
        "prose explaining one attribute of an entity",
        "text",
    ),
    RoleEntry(
        "property",
        "property field column attribute variable element name",
        "the name of one attribute an entity holds",
        "text_key",
    ),
    RoleEntry(
        "status",
        "status active deprecated retired invalid",
        "whether a row is in use or withdrawn",
        "low_cardinality",
    ),
    RoleEntry("parent", "parent parent_id broader", "the identifier of the broader item", "any"),
    RoleEntry("valid_from", "valid_from effective start", "when a row starts to apply", "date"),
    RoleEntry("valid_to", "valid_to expiry end", "when a row stops applying", "date"),
    RoleEntry("synonyms", "synonyms alias", "other names for the same thing", "array"),
)

COINCIDING_ROLES: frozenset[frozenset[str]] = frozenset({frozenset({"display", "name"})})

STATUS_ROLE = "status"
STATUS_VALUE_WORDS: dict[str, str] = {"active": "active", "retired": "deprecated retired invalid"}
ENTITY_PROPERTY_ROLE = "property"
PROPERTY_TEXT_ROLES = ("description",)
NOTES_ROLE = "notes"
REFERENCE_ROLES: tuple[tuple[str, ReferenceKind], ...] = (
    ("domain", "vocabulary"),
    ("fk_entity", "entity"),
)
NO_REFERENCE = "none"

_ROLE_MODELS = (ConceptRoles, CodeListRoles, ValueSetRoles, EntityModelRoles)
# The hierarchy section binds its parent column directly, so no role model declares it.
_UNMODELLED_ROLES = frozenset({"parent"})

_CASE_CHANGE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_NON_ALPHANUMERIC = re.compile(r"[^0-9a-z]+")
_MIN_PLURAL_LENGTH = 4


def normalize(text: str) -> str:
    split = _CASE_CHANGE.sub(" ", text).casefold()
    words = _NON_ALPHANUMERIC.sub(" ", split).split()
    return " ".join(_singular(word) for word in words)


def role(name: str) -> RoleEntry:
    return next(entry for entry in ROLES if entry.name == name)


def coincide(first: str, second: str) -> bool:
    return frozenset({first, second}) in COINCIDING_ROLES


def _check_role_names() -> None:
    known = {name for model in _ROLE_MODELS for name in model.model_fields}
    named = {entry.name for entry in ROLES} - _UNMODELLED_ROLES
    named |= {STATUS_ROLE, ENTITY_PROPERTY_ROLE, NOTES_ROLE, *PROPERTY_TEXT_ROLES}
    named |= {name for name, _ in REFERENCE_ROLES}
    if unknown := sorted(named - known):
        raise ValueError(f"lexicon names roles no spec role model declares: {unknown}")


def _singular(word: str) -> str:
    plural = len(word) >= _MIN_PLURAL_LENGTH and word.endswith("s")
    return word[:-1] if plural and not word.endswith("ss") else word


_check_role_names()

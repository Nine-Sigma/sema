"""Coded columns: which candidate value set a column binds to, and where its labels live.

A candidate is a code source (a code relation's key column, or a dictionary vocabulary) with
the nominating evidence that proposed it (`declared_key`, `stated_domain`, `name_match`) and
its containment numbers. Among the candidates whose containment holds, the one with the most
nominating evidence links; the link is verified when it has any. Containment alone links
unverified; holding candidates with equal evidence tie, and none links. Every candidate the
evidence nominates, and every data-shape candidate that holds, is recorded as
`has_value_set` assertions with its numbers, its alternatives and the reason.

A label column is a text column the code determines: its non-NULL texts fill at least
`COVERAGE_THRESHOLD` of the coded rows, and its most common text per code agrees with at
least that share of them. Exactly one such column labels; several, and no metadata naming
one, leave the label unset. A declared key resolves by its reference; a key to a relation
the run did not read is recorded with the reason.
"""
from __future__ import annotations

import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from sema.engine.join_detector_utils import (
    COVERAGE_THRESHOLD,
    Containment,
    containment,
    normalized_name,
)
from sema.models.assertions import Assertion, AssertionPredicate, AssertionStatus
from sema.models.extraction import ExtractedColumn

DECLARED_KEY = "declared_key"
STATED_DOMAIN = "stated_domain"
NAME_MATCH = "name_match"
CONTAINMENT = "containment"
VERIFIED = "verified"
UNVERIFIED = "unverified"
PAST_BOUND = "past the low-cardinality bound"
OWN_KEY = "every non-null value is distinct: its relation's key, not a code"
NO_VALUES = "the column holds no non-null value"
NO_PROFILE = "the column could not be profiled"
UNREAD_RELATION = "the referenced relation was not read in this run"
_SOURCE = "coded_columns"
_VERIFIED_CONFIDENCE = 0.95
_UNVERIFIED_CONFIDENCE = 0.5


@dataclass(frozen=True)
class CodeSource:
    """Where a value set's codes are listed; `key` is None for a dictionary vocabulary."""

    ref: str
    vocabulary: str
    key: ExtractedColumn | None = None


@dataclass
class Candidate:
    column: ExtractedColumn
    source: CodeSource
    evidence: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class Binding:
    candidate: Candidate
    containment: Containment
    linked: bool
    verification: str
    alternatives: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class StatedDomain:
    entity: str | None
    property: str
    vocabulary: str


def place(domain: StatedDomain, columns: Sequence[ExtractedColumn]) -> list[ExtractedColumn]:
    """Columns whose normalized name equals the property's, in the entity's table if one is named.

    `columns` are every column of the run, so an entity's table in one schema limits all.
    """
    wanted = normalized_name(domain.property)
    entity = normalized_name(domain.entity) if domain.entity else None
    named = {(c.schema, c.table_name) for c in columns if normalized_name(c.table_name) == entity}
    return [
        c for c in columns
        if normalized_name(c.name) == wanted and (not named or (c.schema, c.table_name) in named)
    ]


def decide(scored: Sequence[tuple[Candidate, Containment]]) -> list[Binding]:
    """One column's candidates; a data-shape candidate that does not hold is no proposal."""
    holders = [c for c, measured in scored if measured.holds]
    best = max((len(c.evidence) for c in holders), default=-1)
    top = [c for c in holders if len(c.evidence) == best]
    winner = top[0] if len(top) == 1 else None
    bindings = []
    for candidate, measured in scored:
        if not measured.holds and not candidate.evidence:
            continue
        linked = candidate is winner
        verified = linked and bool(candidate.evidence)
        bindings.append(Binding(
            candidate, measured, linked, VERIFIED if verified else UNVERIFIED,
            tuple(sorted(c.source.ref for c in holders if c is not candidate)),
            _reason(measured, linked, verified, tied=candidate in top and winner is None),
        ))
    return bindings


def _reason(measured: Containment, linked: bool, verified: bool, *, tied: bool) -> str:
    if not measured.holds:
        if not measured.observed_distinct:
            return NO_VALUES
        return "containment below the floor"
    if linked:
        return "" if verified else "containment alone: no nominating evidence"
    if tied:
        return "tie: candidates with equal evidence, none linked"
    return "a candidate with more evidence outranks it"


def label_reading(
    pairs: Mapping[tuple[str | None, str | None], int] | None,
) -> tuple[float, dict[str, str | None]] | None:
    """The violation rate and each code's label when the code determines the text, else None.

    Only non-NULL texts count: they must fill at least the floor share of the coded rows, and
    the share of them that is not their code's most common text is the violation rate. A
    failed read (None) or an empty one determines nothing.
    """
    by_code: dict[str, Counter[str]] = {}
    coded_rows = 0
    for (code, text), rows in (pairs or {}).items():
        if code is None:
            continue
        coded_rows += rows
        if text is not None:
            by_code.setdefault(code, Counter())[text] += rows
    texted = sum(sum(counts.values()) for counts in by_code.values())
    if not texted or texted / coded_rows < COVERAGE_THRESHOLD:
        return None
    majority = {code: counts.most_common(1)[0] for code, counts in by_code.items()}
    violation_rate = 1 - sum(rows for _, rows in majority.values()) / texted
    if 1 - violation_rate < COVERAGE_THRESHOLD:
        return None
    labels: dict[str, str | None] = {code: text for code, (text, _) in majority.items()}
    return violation_rate, labels


def label_column(
    readings: Sequence[tuple[ExtractedColumn, tuple[float, dict[str, str | None]]]],
    display: str | None,
) -> tuple[ExtractedColumn, tuple[float, dict[str, str | None]]] | None:
    """The one text column a code key labels with: a constant text never labels, and among
    several the one the display role names, under its normalized name, wins."""
    varying = [
        (text, reading) for text, reading in readings
        if len(reading[1]) < 2 or len(set(reading[1].values())) > 1
    ]
    if len(varying) > 1 and display is not None:
        varying = [
            (text, reading) for text, reading in varying
            if normalized_name(text.name) == normalized_name(display)
        ]
    return varying[0] if len(varying) == 1 else None


def loaded_relation(
    listed: Mapping[str, str | None],
    relations: Sequence[tuple[ExtractedColumn, Mapping[str, str | None]]],
) -> ExtractedColumn | None:
    """The one code key the column holds on that also lists a dictionary list's codes."""
    if not listed:
        return None
    codes = dict.fromkeys(listed, 1)
    keys = [key for key, keyed in relations if containment(codes, keyed).holds]
    return keys[0] if len(keys) == 1 else None


def _assertion(
    subject: str, payload: dict[str, Any], object_ref: str, confidence: float,
    run_id: str, schema: str,
) -> Assertion:
    return Assertion(
        id=str(uuid.uuid4()), subject_ref=subject, predicate=AssertionPredicate.HAS_VALUE_SET,
        payload=payload, object_ref=object_ref, source=_SOURCE, confidence=confidence,
        status=AssertionStatus.AUTO, run_id=run_id, observed_at=datetime.now(timezone.utc),
        source_schema=schema,
    )


def binding_assertions(binding: Binding, run_id: str, schema: str) -> list[Assertion]:
    """One assertion per nominating evidence and one with the containment numbers."""
    candidate, measured = binding.candidate, binding.containment
    subject, value_set = candidate.column.ref or "", candidate.source.ref
    confidence = (
        _VERIFIED_CONFIDENCE if binding.verification == VERIFIED else _UNVERIFIED_CONFIDENCE
    )
    numbers = {
        "evidence": CONTAINMENT, "value_set": value_set,
        "observed_distinct": measured.observed_distinct, "null_rows": measured.null_rows,
        "listed_value_share": measured.listed_value_share,
        "listed_row_share": measured.listed_row_share, "holds": measured.holds,
        "linked": binding.linked, "verification": binding.verification,
        "alternatives": list(binding.alternatives), "reason": binding.reason,
    }
    nominations = [
        {"evidence": kind, "value_set": value_set, **detail}
        for kind, detail in sorted(candidate.evidence.items())
    ]
    return [
        _assertion(subject, payload, value_set, confidence, run_id, schema)
        for payload in (*nominations, numbers)
    ]


def unplaced_assertion(
    subject: str, domain: StatedDomain, run_id: str, schema: str, reason: str | None = None,
) -> Assertion:
    """A stated domain no column takes: none is named for it, or `reason` says why not."""
    payload = {
        "evidence": STATED_DOMAIN, "value_set": domain.vocabulary, "property": domain.property,
        "entity": domain.entity, "linked": False, "verification": UNVERIFIED,
        "reason": reason or f"no column's normalized name equals property {domain.property!r}",
    }
    return _assertion(subject, payload, domain.vocabulary, _UNVERIFIED_CONFIDENCE, run_id, schema)


def unread_key_assertion(
    subject: str, key_ref: str, constraint: str, run_id: str, schema: str,
) -> Assertion:
    """A declared key whose referenced relation the run did not read: no data decides it."""
    payload = {
        "evidence": DECLARED_KEY, "value_set": key_ref, "constraint": constraint,
        "linked": False, "verification": UNVERIFIED, "reason": UNREAD_RELATION,
    }
    return _assertion(subject, payload, key_ref, _UNVERIFIED_CONFIDENCE, run_id, schema)

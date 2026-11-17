"""Coded columns: which candidate value set a column binds to, and where its labels live.

A candidate is a code source (a code relation's key column, or a dictionary vocabulary) with
the nominating evidence that proposed it (`declared_key`, `stated_domain`, `name_match`) and
its containment numbers. Among the candidates whose containment holds, the one with the most
nominating evidence links; the link is verified when it has any. Containment alone links
unverified; holding candidates with equal evidence tie, and none links. Every candidate the
evidence nominates, and every data-shape candidate that holds, is recorded as
`has_value_set` assertions with its numbers, its alternatives and the reason.

A label column is a text column the code determines: its most common text per code agrees
with at least `COVERAGE_THRESHOLD` of the rows. Exactly one such column labels; several, and
no metadata naming one, leave the label unset.
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
    """Columns whose normalized name equals the property's, in the entity's table if one is named."""
    wanted = normalized_name(domain.property)
    entity = normalized_name(domain.entity) if domain.entity else None
    named = {c.table_name for c in columns if normalized_name(c.table_name) == entity}
    return [
        c for c in columns
        if normalized_name(c.name) == wanted and (not named or c.table_name in named)
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
            return "the column holds no non-null value"
        return "containment below the floor"
    if linked:
        return "" if verified else "containment alone: no nominating evidence"
    if tied:
        return "tie: candidates with equal evidence, none linked"
    return "a candidate with more evidence outranks it"


def majority_labels(
    pairs: Mapping[tuple[str | None, str | None], int],
) -> tuple[float | None, dict[str, str | None]]:
    """Share of non-null-code rows whose text is not their code's most common one, and that text."""
    by_code: dict[str, Counter[str | None]] = {}
    for (code, text), rows in pairs.items():
        if code is not None:
            by_code.setdefault(code, Counter())[text] += rows
    total = sum(sum(counts.values()) for counts in by_code.values())
    if not total:
        return None, {}
    majority = {code: counts.most_common(1)[0] for code, counts in by_code.items()}
    agreeing = sum(rows for _, rows in majority.values())
    return 1 - agreeing / total, {code: text for code, (text, _) in majority.items()}


def determines(violation_rate: float | None) -> bool:
    return violation_rate is not None and 1 - violation_rate >= COVERAGE_THRESHOLD


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
    subject: str, domain: StatedDomain, run_id: str, schema: str,
) -> Assertion:
    payload = {
        "evidence": STATED_DOMAIN, "value_set": domain.vocabulary, "property": domain.property,
        "entity": domain.entity, "linked": False, "verification": UNVERIFIED,
        "reason": f"no column's normalized name equals property {domain.property!r}",
    }
    return _assertion(subject, payload, domain.vocabulary, _UNVERIFIED_CONFIDENCE, run_id, schema)

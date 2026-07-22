"""Gate D-lite staging QA — data shapes and pure checks (US-011).

Lives under ``eval/`` (R29-allowlisted) so it may reconcile against the US-002
gold set without tripping the OMOP/OncoTree coupling guard. The three §1.5
staging checks are pure functions over generic :class:`StagingRow` rows:

- **row count**: the staged scope row count equals the source row count.
- **null-rate**: a NULL ``<target_concept_column>`` is legitimate ONLY for a
  NO_MAP row; a RESOLVED-but-NULL (or NO_MAP-but-populated) row is a defect.
- **NO_MAP accounting**: a code the gold set labels RESOLVED but staged as
  NO_MAP is reported, never silently dropped.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence

from sema.eval.mapping_goldset import GoldSet
from sema.eval.mapping_goldset_utils import GoldLabel

_NO_MAP = "NO_MAP"
_RESOLVED = "RESOLVED"


class QAOutcome(str, Enum):
    """Pass/fail outcome of a single check or the whole report."""

    PASS = "PASS"
    FAIL = "FAIL"


@dataclass(frozen=True)
class StagingRow:
    """One staged row, projected to the fields the QA reconciles."""

    source_value: str
    target_value: int | None
    resolution_status: str


@dataclass(frozen=True)
class QACheck:
    """A single named check with a structured failure reason."""

    name: str
    outcome: QAOutcome
    reason: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.outcome is QAOutcome.PASS

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "outcome": self.outcome.value,
            "reason": self.reason,
            "details": self.details,
        }


@dataclass(frozen=True)
class StagingQAReport:
    """Aggregate of the staging checks (PASS only when every check passes)."""

    checks: tuple[QACheck, ...]

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def outcome(self) -> QAOutcome:
        return QAOutcome.PASS if self.passed else QAOutcome.FAIL

    def failures(self) -> list[QACheck]:
        return [c for c in self.checks if not c.passed]

    def as_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome.value,
            "checks": [c.as_dict() for c in self.checks],
        }


def check_row_count(actual: int, expected: int) -> QACheck:
    """Staged scope row count must equal the source row count."""
    details = {"actual": actual, "expected": expected}
    if actual == expected:
        return QACheck("row_count", QAOutcome.PASS, details=details)
    return QACheck(
        "row_count",
        QAOutcome.FAIL,
        reason=f"staged row count {actual} != expected source row count {expected}",
        details=details,
    )


def check_null_rate(rows: Sequence[StagingRow]) -> QACheck:
    """Reconcile NULL ``<target_concept_column>`` against ``resolution_status``."""
    resolved_but_null = sorted(
        {r.source_value for r in rows if r.resolution_status == _RESOLVED and r.target_value is None}
    )
    no_map_but_populated = sorted(
        {r.source_value for r in rows if r.resolution_status == _NO_MAP and r.target_value is not None}
    )
    details: dict[str, Any] = {
        "total": len(rows),
        "null_rows": sum(1 for r in rows if r.target_value is None),
        "no_map_rows": sum(1 for r in rows if r.resolution_status == _NO_MAP),
        "resolved_but_null": resolved_but_null,
        "no_map_but_populated": no_map_but_populated,
    }
    if not resolved_but_null and not no_map_but_populated:
        return QACheck("null_rate", QAOutcome.PASS, details=details)
    return QACheck(
        "null_rate",
        QAOutcome.FAIL,
        reason=(
            f"{len(resolved_but_null)} RESOLVED rows have a NULL target and "
            f"{len(no_map_but_populated)} NO_MAP rows are populated"
        ),
        details=details,
    )


def check_fk_closure(orphan_count: int, *, fk_label: str = "the parent") -> QACheck:
    """Gate-D-lite (S1-07): no child FK may reference a missing parent row."""
    details = {"orphan_rows": orphan_count}
    if orphan_count == 0:
        return QACheck("fk_closure", QAOutcome.PASS, details=details)
    return QACheck(
        "fk_closure",
        QAOutcome.FAIL,
        reason=f"{orphan_count} child rows reference a missing {fk_label} row",
        details=details,
    )


def check_required_not_null(null_counts: Mapping[str, int]) -> QACheck:
    """Gate-D-lite (S1-07): each required (non-nullable) field must have 0 NULLs.

    ``null_counts`` maps each REQUIRED field to its staged NULL count. A field the
    contract marks nullable (D4's ``condition_start_date``) is simply absent from
    this map — its null rate is reported separately, never gated.
    """
    offenders = {field_name: n for field_name, n in null_counts.items() if n > 0}
    details = {"null_counts": dict(null_counts), "offenders": offenders}
    if not offenders:
        return QACheck("required_not_null", QAOutcome.PASS, details=details)
    return QACheck(
        "required_not_null",
        QAOutcome.FAIL,
        reason=(
            f"{len(offenders)} required fields carry NULLs: "
            f"{sorted(offenders)}"
        ),
        details=details,
    )


def check_missing_key_disposition(
    *, written_rows: int, missing_key_rows: int, source_rows: int
) -> QACheck:
    """Gate-D-lite (S1-07/D5): every source row is written OR routed to review.

    The row-count identity is ``written + missing_key == source`` — plain
    equality with source only holds when no key is missing, so the disposition
    count is accounted for explicitly, never dropped.
    """
    accounted = written_rows + missing_key_rows
    details = {
        "written": written_rows,
        "missing_key": missing_key_rows,
        "source": source_rows,
        "accounted": accounted,
    }
    if accounted == source_rows:
        return QACheck("missing_key_disposition", QAOutcome.PASS, details=details)
    return QACheck(
        "missing_key_disposition",
        QAOutcome.FAIL,
        reason=(
            f"written {written_rows} + missing-key {missing_key_rows} = "
            f"{accounted} != source {source_rows}"
        ),
        details=details,
    )


def check_no_map_accounting(rows: Sequence[StagingRow], gold_set: GoldSet) -> QACheck:
    """Reconcile staged NO_MAP codes against the gold set (US-002)."""
    gold_by_code = gold_set.by_code()
    staged_no_map = sorted({r.source_value for r in rows if r.resolution_status == _NO_MAP})
    gold_resolved_but_no_map = [
        code
        for code in staged_no_map
        if code in gold_by_code
        and gold_by_code[code].gold_label is GoldLabel.RESOLVED
    ]
    no_map_without_gold = [
        code
        for code in staged_no_map
        if code in gold_by_code
        and gold_by_code[code].gold_label is GoldLabel.UNLABELLED
    ]
    details: dict[str, Any] = {
        "staged_no_map": staged_no_map,
        "gold_resolved_but_no_map": gold_resolved_but_no_map,
        "no_map_unlabelled_gold": no_map_without_gold,
    }
    if not gold_resolved_but_no_map:
        return QACheck("no_map_accounting", QAOutcome.PASS, details=details)
    return QACheck(
        "no_map_accounting",
        QAOutcome.FAIL,
        reason=(
            f"{len(gold_resolved_but_no_map)} codes the gold set labels RESOLVED "
            "are staged as NO_MAP"
        ),
        details=details,
    )

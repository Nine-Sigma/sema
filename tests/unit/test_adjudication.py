"""US-002 / G-05: the second-review floor and what a retired code still owes.

Adjudication is the one place a single bad label damages ``mapped_precision``
directly (a wrong gold ``NO_MAP`` scores ``fp_map`` straight against a correctly
mapped code), so the floor must mean what it says: a *second* reviewer, and a
declared code that left the scope reported rather than silently forgiven.
"""

from __future__ import annotations

import pytest

from sema.eval.adjudication import (
    MIN_SECOND_REVIEWED_HEAD_LABELS,
    adjudication_qualifiers,
    codes_needing_second_review,
    retired_challenge_codes,
)
from sema.eval.mapping_goldset_utils import GoldLabel, GoldRow, TierState

pytestmark = pytest.mark.unit


def _row(
    code: str,
    state: TierState,
    label: GoldLabel = GoldLabel.UNLABELLED,
    *,
    curator: str | None = "dean",
    second_reviewer: str | None = None,
) -> GoldRow:
    labelled = label is not GoldLabel.UNLABELLED
    return GoldRow(
        oncotree_code=code,
        gold_concept_id=None,
        gold_label=label,
        row_count=10,
        tier_state=state,
        curator=curator if labelled else None,
        review_date="2026-08-12" if labelled else None,
        evidence="OncoTree browser" if labelled else None,
        second_reviewer=second_reviewer,
    )


def _head(n: int, **kwargs: object) -> list[GoldRow]:
    return [
        _row(f"H{i}", TierState.IN_TIER, GoldLabel.NO_MAP, **kwargs)  # type: ignore[arg-type]
        for i in range(n)
    ]


# --- a second reviewer must be a second person ------------------------------


def test_self_review_does_not_clear_the_floor() -> None:
    rows = _head(MIN_SECOND_REVIEWED_HEAD_LABELS + 2, curator="alice", second_reviewer="alice")
    assert adjudication_qualifiers(rows, ()) == ("unadjudicated",)


def test_self_review_is_detected_across_case_and_padding() -> None:
    rows = _head(12, curator=" Alice ", second_reviewer="alice")
    assert adjudication_qualifiers(rows, ()) == ("unadjudicated",)


def test_a_distinct_second_reviewer_clears_the_floor() -> None:
    rows = _head(12, curator="alice", second_reviewer="bob")
    assert adjudication_qualifiers(rows, ()) == ()


def test_a_self_reviewed_challenge_code_still_needs_review() -> None:
    rows = [
        *_head(12, curator="alice", second_reviewer="bob"),
        _row(
            "UESL",
            TierState.CHALLENGE,
            GoldLabel.NO_MAP,
            curator="alice",
            second_reviewer="alice",
        ),
    ]
    assert codes_needing_second_review(rows, ("UESL",)) == ["UESL"]
    assert adjudication_qualifiers(rows, ("UESL",)) == ("unadjudicated",)


# --- a retired challenge code owes no label, but is reported ----------------


def test_a_retired_challenge_code_is_not_owed_a_label() -> None:
    rows = [
        *_head(12, curator="alice", second_reviewer="bob"),
        _row("GONE", TierState.RETIRED),
    ]
    assert codes_needing_second_review(rows, ("GONE",)) == []
    assert adjudication_qualifiers(rows, ("GONE",)) == ()


def test_a_retired_challenge_code_is_reported_separately() -> None:
    rows = [*_head(2), _row("GONE", TierState.RETIRED)]
    assert retired_challenge_codes(rows, ("GONE", "H0")) == ["GONE"]


def test_a_challenge_code_with_no_row_at_all_still_needs_review() -> None:
    """Absent is not retired: nothing has recorded that it left the scope."""
    rows = _head(12, curator="alice", second_reviewer="bob")
    assert codes_needing_second_review(rows, ("NOPE",)) == ["NOPE"]
    assert retired_challenge_codes(rows, ("NOPE",)) == []

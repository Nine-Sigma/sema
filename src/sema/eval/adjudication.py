"""US-002 / G-05: the second-review floor over the DECLARED challenge stratum.

Separated from the report shapes because it answers a different question: not
"how did the resolver score" but "is this oracle adjudicated enough to drop the
``unadjudicated`` qualifier". The floor is defined over the snapshot's
declaration, never over whatever has been labelled so far — every row-derived
shortcut weakens it, and the challenge stratum is the one place a single bad
label damages ``mapped_precision`` directly (a wrong gold ``NO_MAP`` scores
``fp_map`` straight against a correctly-mapped code).
"""

from __future__ import annotations

from collections.abc import Sequence

from sema.eval.mapping_goldset_utils import GoldLabel, GoldRow, TierState

__all__ = [
    "MIN_SECOND_REVIEWED_HEAD_LABELS",
    "adjudication_qualifiers",
    "codes_needing_second_review",
    "independently_reviewed",
    "retired_challenge_codes",
]

# Second-review floor before a verdict may drop the ``unadjudicated`` qualifier:
# every labelled challenge code plus a sample of head labels.
MIN_SECOND_REVIEWED_HEAD_LABELS = 10


def independently_reviewed(row: GoldRow) -> bool:
    """True only when a SECOND person reviewed the label.

    A curator who fills their own name into ``second_reviewer`` has adjudicated
    nothing; taken at face value it cleared the whole floor.
    """
    if not row.second_reviewer or not row.curator:
        return False
    return row.curator.strip().casefold() != row.second_reviewer.strip().casefold()


def adjudication_qualifiers(
    rows: Sequence[GoldRow],
    challenge_codes: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """``('unadjudicated',)`` unless the G-05 second-review floor was met.

    * ``challenge_codes`` must come from ``GoldSetHeader.challenge_codes``. A
      declared challenge code that also ranks inside the head (live case:
      ``IMMC``) resolves to ``IN_TIER``, so a ``tier_state`` check could never
      demand its review — and an entirely unlabelled stratum passed vacuously.
    * the head sample size is ``min(10, len(head))`` over the WHOLE head; taken
      over the labelled head it collapsed to "all of whatever you've labelled".
    * an out-of-tier label is not evidence of an adjudicated oracle.
    """
    declared = (
        frozenset(challenge_codes)
        if challenge_codes is not None
        else frozenset(r.oncotree_code for r in rows if r.tier_state is TierState.CHALLENGE)
    )
    head = [r for r in rows if r.tier_state is TierState.IN_TIER]
    labelled_head = [r for r in head if r.gold_label is not GoldLabel.UNLABELLED]
    if not labelled_head:
        return ("unadjudicated",)
    if codes_needing_second_review(rows, declared):
        return ("unadjudicated",)
    reviewed = sum(1 for r in labelled_head if independently_reviewed(r))
    if reviewed < min(MIN_SECOND_REVIEWED_HEAD_LABELS, len(head)):
        return ("unadjudicated",)
    return ()


def codes_needing_second_review(
    rows: Sequence[GoldRow],
    challenge_codes: Sequence[str] | frozenset[str],
) -> list[str]:
    """Declared challenge codes not yet labelled AND independently reviewed.

    Unlabelled counts as needing review; RETIRED does not. A retired code left
    the declared scope, so nothing live grades it and no curator owes it an
    answer — but it is still reported (:func:`retired_challenge_codes`), because
    a declaration that quietly shrinks is how a stratum disappears unnoticed.
    """
    by_code = {r.oncotree_code: r for r in rows}
    return sorted(
        code
        for code in challenge_codes
        if (row := by_code.get(code)) is None
        or (
            row.tier_state is not TierState.RETIRED
            and (
                row.gold_label is GoldLabel.UNLABELLED
                or not independently_reviewed(row)
            )
        )
    )


def retired_challenge_codes(
    rows: Sequence[GoldRow],
    challenge_codes: Sequence[str],
) -> list[str]:
    """Declared challenge codes that have left the scope — owed nothing, reported."""
    by_code = {r.oncotree_code: r for r in rows}
    return sorted(
        code
        for code in challenge_codes
        if (row := by_code.get(code)) is not None
        and row.tier_state is TierState.RETIRED
    )

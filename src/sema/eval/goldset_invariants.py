"""US-002 / G-03: artifact invariants asserted whenever a snapshot is read.

These are the contract, not defensive checks. ``by_code()`` and ``score()`` are
both dict comprehensions keyed on ``oncotree_code`` and neither checks
uniqueness, so a duplicated code silently last-wins — the corruption most likely
to survive review is also the one a digest over the code->label *mapping* cannot
see. Hence: uniqueness, canonical ordering, and a digest over the ordered rows.

The four tier states must partition the measured universe exactly. Derivation is
ordered, so a code appearing in both frozen populations (live case: ``IMMC``,
rank 130) resolves to ``IN_TIER`` and is graded once, in the head.
"""

from __future__ import annotations

from sema.eval.goldset_snapshot_utils import (
    GoldSetHeader,
    UniverseEntry,
    canonical_sort_key,
)
from sema.eval.mapping_goldset_utils import GoldLabel, GoldRow, TierState

__all__ = [
    "SnapshotInvariantError",
    "assert_row_integrity",
    "assert_snapshot_invariants",
    "derive_states",
]


class SnapshotInvariantError(ValueError):
    """A gold-set snapshot violates its own declared contract."""


def derive_states(
    header: GoldSetHeader,
    universe: tuple[UniverseEntry, ...],
    row_codes: frozenset[str],
) -> dict[str, TierState]:
    """Assign every in-scope code — plus every code that left scope — one state."""
    tier = set(header.tier_codes)
    challenge = set(header.challenge_codes)
    states = {
        entry.code: _state_for(entry.code, tier, challenge) for entry in universe
    }
    manifest = set(states)
    for code in row_codes - manifest:
        states[code] = TierState.RETIRED
    return states


def _state_for(code: str, tier: set[str], challenge: set[str]) -> TierState:
    if code in tier:
        return TierState.IN_TIER
    if code in challenge:
        return TierState.CHALLENGE
    return TierState.OUT_OF_TIER


def assert_snapshot_invariants(
    header: GoldSetHeader,
    rows: list[GoldRow],
    universe: tuple[UniverseEntry, ...],
) -> None:
    """Raise :class:`SnapshotInvariantError` on any breach of the contract."""
    assert_row_integrity(rows)
    _assert_labels(rows)
    _assert_states(header, rows, universe)


def assert_row_integrity(rows: list[GoldRow]) -> None:
    """Uniqueness and canonical order — checked before any digest, so that a
    mismatch is reported as the specific corruption rather than as a hash diff."""
    seen: set[str] = set()
    for row in rows:
        if row.oncotree_code in seen:
            raise SnapshotInvariantError(f"duplicate oncotree_code: {row.oncotree_code}")
        seen.add(row.oncotree_code)
    if rows != sorted(rows, key=canonical_sort_key):
        raise SnapshotInvariantError("rows are not in canonical order")


def _assert_labels(rows: list[GoldRow]) -> None:
    for row in rows:
        if row.gold_label is GoldLabel.UNLABELLED:
            _assert_unlabelled_is_empty(row)
            continue
        _assert_concept_consistency(row)
        for field in ("curator", "review_date", "evidence"):
            if not getattr(row, field):
                raise SnapshotInvariantError(
                    f"{row.oncotree_code}: labelled row is missing {field}"
                )


def _assert_unlabelled_is_empty(row: GoldRow) -> None:
    if row.gold_concept_id is not None or row.target_concept_code is not None:
        raise SnapshotInvariantError(
            f"{row.oncotree_code}: UNLABELLED row carries a target concept"
        )


def _assert_concept_consistency(row: GoldRow) -> None:
    if row.gold_label is GoldLabel.RESOLVED:
        if row.gold_concept_id is None:
            raise SnapshotInvariantError(f"{row.oncotree_code}: RESOLVED without a concept id")
        if not row.target_concept_code:
            raise SnapshotInvariantError(
                f"{row.oncotree_code}: RESOLVED without a durable target_concept_code"
            )
    elif row.gold_concept_id is not None or row.target_concept_code is not None:
        raise SnapshotInvariantError(f"{row.oncotree_code}: NO_MAP carries a target concept")


def _assert_states(
    header: GoldSetHeader,
    rows: list[GoldRow],
    universe: tuple[UniverseEntry, ...],
) -> None:
    manifest = {e.code for e in universe}
    for code in (*header.tier_codes, *header.challenge_codes):
        if code not in manifest:
            raise SnapshotInvariantError(f"frozen population code {code} is not in the universe")
    states = derive_states(header, universe, frozenset(r.oncotree_code for r in rows))
    by_code = {r.oncotree_code: r for r in rows}
    for code, state in states.items():
        row = by_code.get(code)
        if row is None:
            if state in (TierState.IN_TIER, TierState.CHALLENGE):
                raise SnapshotInvariantError(f"{code}: {state.value} code has no gold row")
            continue
        if row.tier_state is not state:
            raise SnapshotInvariantError(
                f"{code}: row state {row.tier_state.value} disagrees with the declared "
                f"scope, which makes it {state.value}"
            )

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
    SnapshotInvariantError,
    UniverseEntry,
    canonical_sort_key,
    canonical_universe_key,
    tier_row_share,
)
from sema.eval.mapping_goldset_utils import GoldLabel, GoldRow, TierState

__all__ = [
    "SnapshotInvariantError",
    "assert_row_integrity",
    "assert_snapshot_invariants",
    "assert_universe_integrity",
    "derive_states",
]

# Floats round-trip exactly through JSON, so the share needs no real slack.
_SHARE_TOLERANCE = 1e-9


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
    assert_universe_integrity(universe)
    _assert_labels(rows)
    _assert_declaration(header, universe)
    _assert_states(header, rows, universe)


def assert_universe_integrity(universe: tuple[UniverseEntry, ...]) -> None:
    """The manifest is a contract, not a listing.

    A duplicate code is the corruption no digest can see: consumers disagree on it
    — :meth:`GoldSetSnapshot.universe_row_total` sums both entries while the drift
    report's ``{code: count}`` dict last-wins — so one artifact yields two
    different denominators.
    """
    seen: set[str] = set()
    for entry in universe:
        if entry.code in seen:
            raise SnapshotInvariantError(f"duplicate universe code: {entry.code}")
        seen.add(entry.code)
        if entry.frozen_row_count < 0:
            raise SnapshotInvariantError(
                f"{entry.code}: negative frozen_row_count {entry.frozen_row_count}"
            )
    if list(universe) != sorted(universe, key=canonical_universe_key):
        raise SnapshotInvariantError("universe manifest is not in canonical order")


def _assert_declaration(header: GoldSetHeader, universe: tuple[UniverseEntry, ...]) -> None:
    """The header's own numbers must agree with the manifest beneath them."""
    if not header.vocab_release:
        raise SnapshotInvariantError(
            "the header declares no vocab_release; a gold_concept_id is a bare "
            "integer without the release that minted it, so an unpinned snapshot "
            "grades every release alike and reports vocabulary churn as resolver "
            "error"
        )
    for name, codes in (("tier", header.tier_codes), ("challenge", header.challenge_codes)):
        if len(set(codes)) != len(codes):
            raise SnapshotInvariantError(
                f"duplicate code in the frozen {name} population; its length is "
                "reported as the population size"
            )
    achieved = tier_row_share(header.tier_codes, universe)
    if abs(header.tier_achieved_row_share - achieved) > _SHARE_TOLERANCE:
        raise SnapshotInvariantError(
            f"declared tier achieved_row_share {header.tier_achieved_row_share} "
            f"contradicts the universe manifest, which makes it {achieved}"
        )


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
    by_state = {r.oncotree_code: r.tier_state for r in rows}
    for code in (*header.tier_codes, *header.challenge_codes):
        if code in manifest:
            continue
        if by_state.get(code) is not TierState.RETIRED:
            raise SnapshotInvariantError(
                f"frozen population code {code} left the declared scope, so it must "
                "carry a RETIRED row — dropping it would discard the label it earned"
            )
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

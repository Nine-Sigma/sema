"""US-002 / G-03+G-04: the three operations that produce a new gold-set snapshot.

Named separately because they have different blast radii:

* **observe** — record current counts. The frozen populations do not move, so
  scores across the two snapshots stay comparable up to the weight change.
* **re-tier** — recompute the frozen head. A NEW benchmark: scores before and
  after are not comparable, and the report must say so.
* **re-scope** — change what is being measured. Retires and admits codes.

All three go through one builder, so all three obey the same rule: a human label
is carried forward verbatim and a code that leaves the scope is retired, never
deleted. Deleting it would discard human labour, and re-deriving a label would
make the eval grade its own homework — this path, not the resolver, is the
likeliest place for that to leak in.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

from sema.eval.goldset_invariants import derive_states
from sema.eval.goldset_snapshot import (
    GoldSetSnapshot,
    snapshot_dir,
    write_snapshot,
)
from sema.eval.goldset_snapshot_utils import (
    GoldSetHeader,
    UniverseEntry,
    ordered_rows_digest,
    tier_row_share,
)
from sema.eval.goldset_source import SourceSpec
from sema.eval.mapping_goldset_utils import GoldLabel, GoldRow, TierState

__all__ = [
    "SnapshotDraft",
    "label_projection_digest",
    "observe",
    "publish",
    "re_scope",
    "re_tier",
    "select_tier",
]


@dataclass(frozen=True)
class SnapshotDraft:
    """An unpublished snapshot: the output of one named operation."""

    header: GoldSetHeader
    rows: list[GoldRow]
    universe: tuple[UniverseEntry, ...]

    def by_code(self) -> dict[str, GoldRow]:
        return {r.oncotree_code: r for r in self.rows}


def label_projection_digest(rows: list[GoldRow]) -> str:
    """Digest the human oracle alone, so an operation can prove it changed none."""
    labelled = sorted(
        (r for r in rows if r.gold_label is not GoldLabel.UNLABELLED),
        key=lambda r: r.oncotree_code,
    )
    return ordered_rows_digest([replace(r, row_count=0, tier_state=TierState.IN_TIER, notes="")
                                for r in labelled])


def select_tier(
    observed: list[tuple[str, int]],
    target_row_share: float,
) -> tuple[tuple[str, ...], float]:
    """Smallest prefix by row count reaching ``target_row_share`` of all rows."""
    ordered = sorted(observed, key=lambda item: (-item[1], item[0]))
    total = sum(n for _, n in ordered)
    if total == 0:
        return tuple(c for c, _ in ordered), 0.0
    cumulative = 0
    codes: list[str] = []
    for code, count in ordered:
        cumulative += count
        codes.append(code)
        if cumulative / total >= target_row_share:
            break
    return tuple(codes), cumulative / total


def observe(
    snapshot: GoldSetSnapshot,
    observed: dict[str, int],
    *,
    version: str,
    date: str,
) -> SnapshotDraft:
    """Re-stamp weights against the same declared scope and frozen populations.

    The header's declared code lists are NOT rewritten — that is what keeps two
    observations comparable. But a code that has left the scope is retired rather
    than held at zero rows: held, its row-weighted contribution vanished while it
    still demanded a human label for a code no longer in the data, capping coverage
    below 1.0 permanently and making the acceptance gate unreachable.
    """
    universe = _universe(observed)
    header = replace(
        snapshot.header,
        snapshot_version=version,
        snapshot_date=date,
        tier_achieved_row_share=tier_row_share(snapshot.header.tier_codes, universe),
    )
    return _build(header, snapshot.rows, universe, observed)


def re_tier(
    snapshot: GoldSetSnapshot,
    observed: dict[str, int],
    *,
    version: str,
    date: str,
    target_row_share: float,
    challenge_codes: tuple[str, ...] | None = None,
) -> SnapshotDraft:
    """Recompute the frozen head — a new benchmark, not a refreshed one."""
    return _retarget(
        snapshot, observed, snapshot.header.source_of_truth,
        version=version, date=date, target_row_share=target_row_share,
        challenge_codes=challenge_codes,
    )


def re_scope(
    snapshot: GoldSetSnapshot,
    observed: dict[str, int],
    spec: SourceSpec,
    *,
    version: str,
    date: str,
    target_row_share: float,
    challenge_codes: tuple[str, ...] | None = None,
) -> SnapshotDraft:
    """Change the declared source of truth, retiring codes that leave it."""
    return _retarget(
        snapshot, observed, spec, version=version, date=date,
        target_row_share=target_row_share, challenge_codes=challenge_codes,
    )


def _retarget(
    snapshot: GoldSetSnapshot,
    observed: dict[str, int],
    spec: SourceSpec,
    *,
    version: str,
    date: str,
    target_row_share: float,
    challenge_codes: tuple[str, ...] | None,
) -> SnapshotDraft:
    tier, achieved = select_tier(list(observed.items()), target_row_share)
    declared = tuple(c for c in (challenge_codes or ()) if c in observed)
    header = replace(
        snapshot.header,
        snapshot_version=version,
        snapshot_date=date,
        source_of_truth=spec,
        tier_codes=tier,
        challenge_codes=declared,
        tier_target_row_share=target_row_share,
        tier_achieved_row_share=achieved,
    )
    return _build(header, snapshot.rows, _universe(observed), observed)


def publish(root: Path, draft: SnapshotDraft) -> Path:
    """Write a new snapshot and advance the pointer. Never touches a prior one."""
    directory = snapshot_dir(draft.header.snapshot_version, root)
    write_snapshot(directory, draft.header, draft.rows, draft.universe)
    (root / "current.json").write_text(
        json.dumps({"snapshot_version": draft.header.snapshot_version}, indent=2) + "\n",
        encoding="utf-8",
    )
    return directory


def _universe(observed: dict[str, int]) -> tuple[UniverseEntry, ...]:
    ordered = sorted(observed.items(), key=lambda item: (-item[1], item[0]))
    return tuple(UniverseEntry(code, count) for code, count in ordered)


def _build(
    header: GoldSetHeader,
    prior_rows: list[GoldRow],
    universe: tuple[UniverseEntry, ...],
    observed: dict[str, int],
) -> SnapshotDraft:
    prior = {r.oncotree_code: r for r in prior_rows}
    states = derive_states(header, universe, frozenset(prior))
    carried = {
        code: state
        for code, state in states.items()
        if code in prior or state in (TierState.IN_TIER, TierState.CHALLENGE)
    }
    scope = ", ".join(header.source_of_truth.scope_values)
    rows = [
        _row_for(code, state, prior.get(code), observed, scope)
        for code, state in carried.items()
    ]
    return SnapshotDraft(header=header, rows=rows, universe=universe)


def _row_for(
    code: str,
    state: TierState,
    prior: GoldRow | None,
    observed: dict[str, int],
    scope: str,
) -> GoldRow:
    if prior is None:
        return GoldRow(
            oncotree_code=code,
            gold_concept_id=None,
            gold_label=GoldLabel.UNLABELLED,
            row_count=observed.get(code, 0),
            notes=_scaffold_note(state, scope),
            tier_state=state,
        )
    row_count = prior.row_count if state is TierState.RETIRED else observed.get(code, 0)
    notes = (
        prior.notes
        if prior.gold_label is not GoldLabel.UNLABELLED
        else _scaffold_note(state, scope)
    )
    return replace(prior, row_count=row_count, tier_state=state, notes=notes)


def _scaffold_note(state: TierState, scope: str) -> str:
    if state is TierState.RETIRED:
        return f"retired: absent from the declared scope ({scope}); label preserved"
    return f"awaiting human label; {state.value} in the declared scope ({scope})"

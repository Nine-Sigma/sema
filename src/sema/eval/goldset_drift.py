"""US-002 / G-03: gold-set drift and benchmark-freshness reporting.

Two contracts live here, deliberately distinct because they cannot both hold
unconditionally — "ingesting a study never reds the suite" and "fail when the
frozen tier stops representing the data" are in tension, and a large enough
ingest must trip the second:

* **drift** — how far the frozen ``row_count`` weights have moved from the live
  data. Reported as a number; never asserted to zero.
* **freshness** — whether the frozen benchmark still represents the current
  population. May go stale, and says so in those words, so a stale benchmark is
  never mistaken for a corrupt artifact.

Drift is an ABSOLUTE-delta ratio: signed deltas let a doubled code cancel a
halved one and report ~0. It is computed only over codes present in both
snapshots, with additions and disappearances listed separately, and a scope
change suppresses it entirely rather than averaging across incomparable scopes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sema.eval.goldset_snapshot import GoldSetSnapshot
from sema.eval.goldset_source import SourceSpec
from sema.eval.mapping_goldset import score_eligible

__all__ = ["DriftReport", "goldset_drift_report"]


@dataclass(frozen=True)
class DriftReport:
    """Frozen-weight drift plus the representativeness monitor."""

    scope_changed: bool
    codes_added: list[str]
    codes_disappeared: list[str]
    per_code: dict[str, int]
    eligible_drift: float | None
    universe_drift: float | None
    frozen_tier_row_share: float
    unseen_code_share: float
    is_stale: bool
    staleness_reason: str = ""
    thresholds: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "scope_changed": self.scope_changed,
            "codes_added": self.codes_added,
            "codes_disappeared": self.codes_disappeared,
            "per_code_delta": self.per_code,
            "drift": {
                "eligible_populations": self.eligible_drift,
                "full_universe": self.universe_drift,
            },
            "freshness": {
                "frozen_tier_row_share": self.frozen_tier_row_share,
                "unseen_code_share": self.unseen_code_share,
                "is_stale": self.is_stale,
                "reason": self.staleness_reason,
                "thresholds": self.thresholds,
            },
        }


def goldset_drift_report(
    snapshot: GoldSetSnapshot,
    observed: dict[str, int],
    observed_spec: SourceSpec,
) -> DriftReport:
    """Compare a frozen snapshot against a live enumeration of the same scope."""
    frozen = {e.code: e.frozen_row_count for e in snapshot.universe}
    scope_changed = observed_spec != snapshot.header.source_of_truth
    shared = sorted(set(frozen) & set(observed))
    per_code = {c: observed[c] - frozen[c] for c in shared if observed[c] != frozen[c]}
    eligible = {r.oncotree_code for r in snapshot.rows if score_eligible(r)}
    freshness = _freshness(snapshot, observed)
    return DriftReport(
        scope_changed=scope_changed,
        codes_added=sorted(set(observed) - set(frozen)),
        codes_disappeared=sorted(set(frozen) - set(observed)),
        per_code=per_code,
        eligible_drift=None if scope_changed else _drift(frozen, observed, set(shared) & eligible),
        universe_drift=None if scope_changed else _drift(frozen, observed, set(shared)),
        **freshness,
    )


def _drift(frozen: dict[str, int], observed: dict[str, int], codes: set[str]) -> float | None:
    """``sum(|new - old|) / sum(old)`` — absolute, so increases cannot cancel."""
    denominator = sum(frozen[c] for c in codes)
    if denominator == 0:
        return None
    return sum(abs(observed[c] - frozen[c]) for c in codes) / denominator


def _freshness(snapshot: GoldSetSnapshot, observed: dict[str, int]) -> dict[str, Any]:
    header = snapshot.header
    total_rows = sum(observed.values())
    tier_rows = sum(observed.get(c, 0) for c in header.tier_codes)
    tier_share = 0.0 if total_rows == 0 else tier_rows / total_rows
    manifest = {e.code for e in snapshot.universe}
    unseen = [c for c in observed if c not in manifest]
    unseen_share = 0.0 if not observed else len(unseen) / len(observed)
    reasons = _staleness_reasons(header, tier_share, unseen_share)
    return {
        "frozen_tier_row_share": tier_share,
        "unseen_code_share": unseen_share,
        "is_stale": bool(reasons),
        "staleness_reason": " ".join(reasons),
        "thresholds": {
            "min_frozen_tier_row_share": header.min_frozen_tier_row_share,
            "max_unseen_code_share": header.max_unseen_code_share,
        },
    }


def _staleness_reasons(header: Any, tier_share: float, unseen_share: float) -> list[str]:
    reasons: list[str] = []
    if tier_share < header.min_frozen_tier_row_share:
        reasons.append(
            f"benchmark stale — the frozen tier now covers {tier_share:.1%} of rows, "
            f"below the declared {header.min_frozen_tier_row_share:.1%} floor; "
            "re-snapshot via `sema eval goldset re-tier`."
        )
    if unseen_share > header.max_unseen_code_share:
        reasons.append(
            f"benchmark stale — {unseen_share:.1%} of observed codes are outside the "
            f"frozen universe, above the declared {header.max_unseen_code_share:.1%} "
            "ceiling; re-snapshot via `sema eval goldset re-scope`."
        )
    return reasons

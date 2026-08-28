"""US-002 / G-01: gold-set snapshot header, universe manifest, and digests.

A ``gold_concept_id`` is a bare integer, meaningless without the vocabulary
release that minted it — ``sema_resolve.value_mapping`` already carries
``vocab_release`` and the gold set did not, so an OMOP refresh read as resolver
drift. The header pins both sides: the source scope AND the target vocabulary,
domain, release, and oracle.

Both frozen populations (the acceptance head and the challenge stratum) are
stored as explicit code lists. A frequency-defined tier is itself
ingest-sensitive, so recomputing it at test time would re-introduce exactly the
drift the declaration removes.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sema.eval.goldset_source import SourceSpec
from sema.eval.mapping_goldset_utils import GoldLabel, GoldRow, TierState

__all__ = [
    "GoldSetHeader",
    "SnapshotInvariantError",
    "UniverseEntry",
    "canonical_universe_key",
    "file_digest",
    "ordered_rows_digest",
    "row_from_json",
    "row_to_json",
    "staged_publish",
    "tier_row_share",
]


class SnapshotInvariantError(ValueError):
    """A gold-set snapshot violates its own declared contract."""

_STATE_RANK = {
    TierState.IN_TIER: 0,
    TierState.CHALLENGE: 1,
    TierState.OUT_OF_TIER: 2,
    TierState.RETIRED: 3,
}


@dataclass(frozen=True)
class UniverseEntry:
    """One code of the measured universe with its frozen row count.

    The manifest freezes ALL codes in scope, not just the ones carrying a gold
    row: without a frozen ``old`` count, per-code drift for the rest is
    incomputable and a large change among them biases the aggregate toward zero.
    This is storage, not eligibility — what may be labelled or scored is decided
    by the tier states, never by what is on disk.
    """

    code: str
    frozen_row_count: int

    def as_dict(self) -> dict[str, Any]:
        return {"code": self.code, "frozen_row_count": self.frozen_row_count}


@dataclass(frozen=True)
class GoldSetHeader:
    """The snapshot's declaration: what it measured, against what, and when."""

    snapshot_version: str
    snapshot_date: str
    source_of_truth: SourceSpec
    target_vocabulary: str
    target_domain: str
    vocab_release: str
    oracle_source: str
    oracle_version: str
    tier_codes: tuple[str, ...]
    challenge_codes: tuple[str, ...]
    tier_target_row_share: float
    tier_achieved_row_share: float
    min_frozen_tier_row_share: float
    max_unseen_code_share: float
    rows_sha256: str = ""
    universe_sha256: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "snapshot_version": self.snapshot_version,
            "snapshot_date": self.snapshot_date,
            "source_of_truth": self.source_of_truth.as_dict(),
            "target_vocabulary": self.target_vocabulary,
            "target_domain": self.target_domain,
            "vocab_release": self.vocab_release,
            "oracle_source": self.oracle_source,
            "oracle_version": self.oracle_version,
            "tier": {
                "target_row_share": self.tier_target_row_share,
                "achieved_row_share": self.tier_achieved_row_share,
                "codes": list(self.tier_codes),
            },
            "challenge": {"codes": list(self.challenge_codes)},
            "freshness": {
                "min_frozen_tier_row_share": self.min_frozen_tier_row_share,
                "max_unseen_code_share": self.max_unseen_code_share,
            },
            "rows_sha256": self.rows_sha256,
            "universe_sha256": self.universe_sha256,
        }

    @classmethod
    def from_dict(cls, obj: dict[str, Any]) -> GoldSetHeader:
        tier = obj["tier"]
        freshness = obj["freshness"]
        return cls(
            snapshot_version=str(obj["snapshot_version"]),
            snapshot_date=str(obj["snapshot_date"]),
            source_of_truth=SourceSpec.from_dict(obj["source_of_truth"]),
            target_vocabulary=str(obj["target_vocabulary"]),
            target_domain=str(obj["target_domain"]),
            vocab_release=str(obj["vocab_release"]),
            oracle_source=str(obj["oracle_source"]),
            oracle_version=str(obj["oracle_version"]),
            tier_codes=tuple(str(c) for c in tier["codes"]),
            challenge_codes=tuple(str(c) for c in obj["challenge"]["codes"]),
            tier_target_row_share=float(tier["target_row_share"]),
            tier_achieved_row_share=float(tier["achieved_row_share"]),
            min_frozen_tier_row_share=float(freshness["min_frozen_tier_row_share"]),
            max_unseen_code_share=float(freshness["max_unseen_code_share"]),
            rows_sha256=str(obj.get("rows_sha256", "")),
            universe_sha256=str(obj.get("universe_sha256", "")),
        )

    def with_digests(self, *, rows: str, universe: str) -> GoldSetHeader:
        from dataclasses import replace

        return replace(self, rows_sha256=rows, universe_sha256=universe)


def row_to_json(row: GoldRow) -> dict[str, Any]:
    return {
        "oncotree_code": row.oncotree_code,
        "gold_concept_id": row.gold_concept_id,
        "target_concept_code": row.target_concept_code,
        "gold_label": row.gold_label.value,
        "tier_state": row.tier_state.value,
        "row_count": row.row_count,
        "curator": row.curator,
        "review_date": row.review_date,
        "evidence": row.evidence,
        "second_reviewer": row.second_reviewer,
        "notes": row.notes,
    }


_ROW_KEYS = frozenset(
    {
        "oncotree_code",
        "gold_concept_id",
        "target_concept_code",
        "gold_label",
        "tier_state",
        "row_count",
        "curator",
        "review_date",
        "evidence",
        "second_reviewer",
        "notes",
    }
)


def row_from_json(obj: dict[str, Any]) -> GoldRow:
    """Parse one artifact row, refusing anything the row shape cannot hold.

    A key the parser drops is a field the artifact silently fails to account
    for — and the reader downstream may not be this one.
    """
    unknown = sorted(set(obj) - _ROW_KEYS)
    if unknown:
        raise SnapshotInvariantError(
            f"{obj.get('oncotree_code', '?')}: unknown gold row key(s) {unknown}"
        )
    concept = obj.get("gold_concept_id")
    return GoldRow(
        oncotree_code=str(obj["oncotree_code"]),
        gold_concept_id=None if concept is None else int(concept),
        gold_label=GoldLabel(obj["gold_label"]),
        row_count=int(obj["row_count"]),
        notes=str(obj.get("notes", "")),
        tier_state=TierState(obj.get("tier_state", TierState.IN_TIER.value)),
        target_concept_code=_optional_str(obj.get("target_concept_code")),
        curator=_optional_str(obj.get("curator")),
        review_date=_optional_str(obj.get("review_date")),
        evidence=_optional_str(obj.get("evidence")),
        second_reviewer=_optional_str(obj.get("second_reviewer")),
    )


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def canonical_sort_key(row: GoldRow) -> tuple[int, int, str]:
    """Canonical artifact order: state, then richest code first, then code."""
    return (_STATE_RANK[row.tier_state], -row.row_count, row.oncotree_code)


def _digest(payload: list[dict[str, Any]]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def ordered_rows_digest(rows: list[GoldRow]) -> str:
    """Digest the ordered row LIST — a dict digest is blind to duplicates."""
    return _digest([row_to_json(r) for r in rows])


def file_digest(path: Path) -> str:
    """SHA-256 over the file's BYTES — the artifact itself, not a projection of it."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


@contextmanager
def staged_publish(
    directory: Path,
    *,
    conflict: Callable[[], BaseException],
) -> Iterator[Path]:
    """Build an immutable directory in a temp sibling, then rename it into place.

    Every artifact under ``eval/`` is written once and refuses to overwrite
    itself, which turns a half-written directory into a permanent block: it
    satisfies the refusal, so the retry that would have completed it is rejected,
    and the version number is burned. Yields the staging path; the rename happens
    only if the body returns. ``conflict`` builds the caller's own already-exists
    error, raised both for a directory that is already there and for a rename
    that loses a race to one.
    """
    if directory.exists():
        raise conflict()
    directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{directory.name}.", dir=directory.parent))
    try:
        yield staging
        # mkdtemp is 0700; a published artifact is evidence others must read.
        staging.chmod(0o755)
        try:
            os.replace(staging, directory)
        except OSError as exc:
            raise conflict() from exc
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def canonical_universe_key(entry: UniverseEntry) -> tuple[int, str]:
    """Canonical manifest order: richest code first, then code. Ordering is what
    makes two manifests diffable across snapshots."""
    return (-entry.frozen_row_count, entry.code)


def tier_row_share(
    tier_codes: tuple[str, ...],
    universe: tuple[UniverseEntry, ...],
) -> float:
    """The frozen tier's share of the manifest's rows.

    One definition, used both to stamp ``tier_achieved_row_share`` and to assert it
    on load: a snapshot that replaced every count beneath a carried-forward share
    contradicted its own manifest, and the CLI echoed the stale figure.
    """
    total = sum(e.frozen_row_count for e in universe)
    if total == 0:
        return 0.0
    tier = set(tier_codes)
    return sum(e.frozen_row_count for e in universe if e.code in tier) / total

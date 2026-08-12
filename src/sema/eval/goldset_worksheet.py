"""US-002 / G-05: the labelling worksheet, and the path labels take back in.

The oracle is a human gate. This module prepares what a curator fills in and
applies what they filled — it can never originate a label.

Two anti-anchoring constraints from D4 shape the worksheet:

* **No candidate targets.** The resolver's answers *are* the ``vocabulary_omop``
  answers, so pre-filling candidates hands the reviewer the resolver's output to
  rubber-stamp. Only source-side OncoTree context is offered.
* **No visible strata.** The 12 challenge codes were selected *because the
  resolver declared uncertainty*. A reviewer told that labels differently, so the
  worksheet neither says so nor orders rows in a way (by frequency, by tier) that
  would let the stratum be picked out by eye.
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import replace
from pathlib import Path

from sema.eval.goldset_ops import SnapshotDraft
from sema.eval.goldset_snapshot import GoldSetSnapshot
from sema.eval.mapping_goldset_utils import GoldLabel, GoldRow, TierState

__all__ = ["WORKSHEET_COLUMNS", "apply_labels", "build_worksheet", "worksheet_codes"]

WORKSHEET_COLUMNS = (
    "oncotree_code",
    "oncotree_name",
    "main_type",
    "tissue",
    "gold_label",
    "gold_concept_id",
    "target_concept_code",
    "curator",
    "review_date",
    "evidence",
    "second_reviewer",
    "notes",
)


def worksheet_codes(snapshot: GoldSetSnapshot, head_size: int) -> list[str]:
    """The head prefix plus every challenge code, interleaved beyond recognition."""
    selected = set(snapshot.header.tier_codes[:head_size]) | set(
        snapshot.header.challenge_codes
    )
    version = snapshot.header.snapshot_version
    return sorted(
        selected,
        key=lambda c: hashlib.sha256(f"{version}:worksheet:{c}".encode()).hexdigest(),
    )


def build_worksheet(
    snapshot: GoldSetSnapshot,
    path: str | Path,
    *,
    head_size: int = 50,
    source_names: dict[str, str] | None = None,
    source_main_types: dict[str, str] | None = None,
    source_tissues: dict[str, str] | None = None,
) -> list[str]:
    """Write a blank labelling worksheet for tier 1. Returns the codes included."""
    names = source_names or {}
    main_types = source_main_types or {}
    tissues = source_tissues or {}
    codes = worksheet_codes(snapshot, head_size)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(WORKSHEET_COLUMNS))
        writer.writeheader()
        for code in codes:
            writer.writerow(
                {
                    "oncotree_code": code,
                    "oncotree_name": names.get(code, ""),
                    "main_type": main_types.get(code, ""),
                    "tissue": tissues.get(code, ""),
                }
            )
    return codes


def apply_labels(
    snapshot: GoldSetSnapshot,
    worksheet: str | Path,
    *,
    version: str,
    date: str,
) -> SnapshotDraft:
    """Apply a curator's completed worksheet, emitting a NEW snapshot draft."""
    labels = _read_worksheet(Path(worksheet))
    by_code = {r.oncotree_code: r for r in snapshot.rows}
    unknown = sorted(set(labels) - set(by_code))
    if unknown:
        raise ValueError(f"worksheet labels codes outside the snapshot: {unknown}")
    rows = [
        _apply(row, labels[row.oncotree_code]) if row.oncotree_code in labels else row
        for row in snapshot.rows
    ]
    header = replace(
        snapshot.header,
        snapshot_version=version,
        snapshot_date=date,
        oracle_version=f"worksheet applied {date}",
    )
    return SnapshotDraft(header=header, rows=rows, universe=snapshot.universe)


def _read_worksheet(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if (r.get("gold_label") or "").strip()]
    labels: dict[str, dict[str, str]] = {}
    for row in rows:
        code = (row.get("oncotree_code") or "").strip()
        if code in labels:
            raise ValueError(f"{code}: worksheet carries more than one label")
        labels[code] = {k: (v or "").strip() for k, v in row.items()}
    return labels


def _apply(row: GoldRow, entry: dict[str, str]) -> GoldRow:
    label = GoldLabel(entry["gold_label"])
    _assert_complete(row.oncotree_code, label, entry)
    concept = entry["gold_concept_id"]
    return replace(
        row,
        gold_label=label,
        gold_concept_id=int(concept) if concept else None,
        target_concept_code=entry["target_concept_code"] or None,
        curator=entry["curator"],
        review_date=entry["review_date"],
        evidence=entry["evidence"],
        second_reviewer=entry["second_reviewer"] or None,
        notes=entry["notes"] or row.notes,
    )


def _assert_complete(code: str, label: GoldLabel, entry: dict[str, str]) -> None:
    if label is GoldLabel.UNLABELLED:
        raise ValueError(f"{code}: UNLABELLED is a scaffold state, not a curator's answer")
    for field in ("curator", "review_date", "evidence"):
        if not entry.get(field):
            raise ValueError(f"{code}: a label needs {field} — it is the annotation floor")
    if label is GoldLabel.RESOLVED:
        if not entry.get("gold_concept_id") or not entry.get("target_concept_code"):
            raise ValueError(
                f"{code}: RESOLVED needs both gold_concept_id and a durable "
                "target_concept_code, which survives a vocabulary release change"
            )
    elif entry.get("gold_concept_id") or entry.get("target_concept_code"):
        raise ValueError(f"{code}: NO_MAP must not carry a target concept")


def challenge_codes_needing_review(rows: list[GoldRow]) -> list[str]:
    """Labelled challenge codes with no second reviewer — the acceptance-gating set."""
    return sorted(
        r.oncotree_code
        for r in rows
        if r.tier_state is TierState.CHALLENGE
        and r.gold_label is not GoldLabel.UNLABELLED
        and not r.second_reviewer
    )

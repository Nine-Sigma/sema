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
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

from sema.eval.goldset_ops import SnapshotDraft
from sema.eval.goldset_snapshot import GoldSetSnapshot
from sema.eval.mapping_goldset_utils import GoldLabel, GoldRow
from sema.eval.mapping_report_utils import codes_needing_second_review

__all__ = [
    "WORKSHEET_COLUMNS",
    "SourceContext",
    "apply_labels",
    "build_worksheet",
    "reference_tissues",
    "worksheet_codes",
]

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


@dataclass(frozen=True)
class SourceContext:
    """Per-code source-side context a curator needs to look a target up alone.

    All three come from the SOURCE side — the OncoTree concept's own name, the
    study's ``CANCER_TYPE`` (OncoTree's ``mainType``), and the tissue from the
    shipped reference CSV. None is derived from ``concept_relationship 'Maps to'``,
    which is the resolver's own path: pre-filling that would hand the reviewer
    the answer to rubber-stamp.
    """

    names: dict[str, str] = field(default_factory=dict)
    main_types: dict[str, str] = field(default_factory=dict)
    tissues: dict[str, str] = field(default_factory=dict)

    def missing(self, codes: Sequence[str]) -> dict[str, list[str]]:
        """Which codes lack which column — a blank cell must not read as 'none exists'."""
        return {
            column: [c for c in codes if not source.get(c)]
            for column, source in (
                ("oncotree_name", self.names),
                ("main_type", self.main_types),
                ("tissue", self.tissues),
            )
        }


def reference_tissues(root: Path) -> dict[str, str]:
    """Tissue per code from the shipped ``oncotree_reference_*.csv`` files.

    OncoTree ancestry is absent from this OMOP build, so the reference CSV is the
    only tissue source — and it covers only part of any tier, which is why
    :meth:`SourceContext.missing` reports the shortfall rather than shipping
    blanks that look like absence.
    """
    tissues: dict[str, str] = {}
    for path in sorted(root.glob("oncotree_reference_*.csv")):
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                tissue = (row.get("tissue") or "").strip()
                if tissue:
                    tissues[(row.get("oncotree_code") or "").strip()] = tissue
    return tissues


def build_worksheet(
    snapshot: GoldSetSnapshot,
    path: str | Path,
    *,
    head_size: int = 50,
    context: SourceContext | None = None,
    source_names: dict[str, str] | None = None,
) -> list[str]:
    """Write a blank labelling worksheet for tier 1. Returns the codes included."""
    resolved = context or SourceContext(names=source_names or {})
    names, main_types, tissues = resolved.names, resolved.main_types, resolved.tissues
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


def challenge_codes_needing_review(
    rows: list[GoldRow],
    challenge_codes: Sequence[str],
) -> list[str]:
    """Declared challenge codes not yet labelled AND second-reviewed.

    Reads the header's declared list, never ``tier_state``: a challenge code that
    also ranks inside the frozen head (live case ``IMMC``) is ``IN_TIER``, so a
    state check would silently exempt it from the review it most needs.
    """
    return codes_needing_second_review(rows, challenge_codes)

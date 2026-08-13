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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

from sema.eval.goldset_ops import SnapshotDraft
from sema.eval.goldset_snapshot import GoldSetSnapshot
from sema.eval.goldset_snapshot_utils import GoldSetHeader
from sema.eval.mapping_goldset_utils import GoldLabel, GoldRow

__all__ = [
    "WORKSHEET_COLUMNS",
    "SourceContext",
    "TargetFacts",
    "apply_labels",
    "build_worksheet",
    "instructions_path",
    "reference_tissues",
    "worksheet_codes",
    "worksheet_concept_ids",
    "write_instructions",
]

WORKSHEET_COLUMNS = (
    "oncotree_code",
    "oncotree_name",
    "main_type",
    "tissue",
    "snapshot_version",
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
                    "snapshot_version": snapshot.header.snapshot_version,
                }
            )
    write_instructions(snapshot.header, target, len(codes))
    return codes


def instructions_path(worksheet: Path) -> Path:
    """The sidecar a worksheet is delivered with — same name, ``.instructions.md``."""
    return worksheet.with_name(f"{worksheet.name}.instructions.md")


def write_instructions(header: GoldSetHeader, worksheet: Path, code_count: int) -> Path:
    """Write the worksheet's sidecar. Returns the path written."""
    path = instructions_path(worksheet)
    path.write_text(_instructions(header, code_count), encoding="utf-8")
    return path


def _instructions(header: GoldSetHeader, code_count: int) -> str:
    """The pins the curator must honour, delivered WITH the worksheet.

    Stated in the artifact rather than in a handoff message: a label is only
    interpretable against the release that minted its concept id, and a
    worksheet outlives whatever conversation shipped it.
    """
    labels = ", ".join(l.value for l in GoldLabel if l is not GoldLabel.UNLABELLED)
    return "\n".join(
        [
            f"# Labelling worksheet — {header.snapshot_version}",
            "",
            f"- **snapshot_version**: `{header.snapshot_version}` — pre-filled in every",
            "  row. Do not edit it: it is what binds these labels to this snapshot.",
            f"- **target vocabulary**: `{header.target_vocabulary}`",
            f"- **target domain**: `{header.target_domain}`",
            f"- **vocabulary release**: `{header.vocab_release}` — a `gold_concept_id` is",
            "  meaningless without it, so a concept id from another release is not a label",
            "  for this gold set.",
            f"- **rows to label**: {code_count}",
            "",
            "## Filling a row",
            "",
            f"1. `gold_label` is one of: {labels}. Leave it blank for a row you have not",
            "   answered — `UNLABELLED` is a scaffold state, not a curator's answer.",
            f"2. `RESOLVED` needs BOTH `gold_concept_id` and `target_concept_code` (the",
            f"   durable {header.target_vocabulary} code, which survives a release change),",
            f"   and the concept must be standard and in the {header.target_domain} domain.",
            "3. `NO_MAP` is a positive claim, not a blank: it carries NO target concept and",
            "   still requires `evidence` saying what you looked for and why nothing fits.",
            "   A wrong NO_MAP scores directly against a correctly-mapped code.",
            "4. `curator`, `review_date` and `evidence` are the annotation floor — every",
            "   label needs all three.",
            "5. `second_reviewer` must be someone OTHER than the curator; your own name",
            "   there adjudicates nothing.",
            "",
            "No candidate targets are pre-filled and the rows are interleaved: both",
            "deliberate, so the oracle stays independent of the resolver being graded.",
            "",
        ]
    )


@dataclass(frozen=True)
class TargetFacts:
    """What the target vocabulary says about one concept id.

    Passed in as data — the pure artifact modules take no database dependency,
    which is also why this check does not live in ``goldset_invariants``:
    snapshot integrity is fixture-backed and must hold off-machine.
    """

    concept_code: str
    vocabulary: str
    domain: str
    standard: bool


def apply_labels(
    snapshot: GoldSetSnapshot,
    worksheet: str | Path,
    *,
    version: str,
    date: str,
    targets: Mapping[int, TargetFacts] | None = None,
) -> SnapshotDraft:
    """Apply a curator's completed worksheet, emitting a NEW snapshot draft.

    ``targets`` — when supplied — checks each RESOLVED label against the concept
    it names, so a transposed digit becomes an error rather than an oracle.
    """
    labels = _read_worksheet(Path(worksheet))
    by_code = {r.oncotree_code: r for r in snapshot.rows}
    unknown = sorted(set(labels) - set(by_code))
    if unknown:
        raise ValueError(f"worksheet labels codes outside the snapshot: {unknown}")
    for code, entry in labels.items():
        _assert_snapshot_pin(code, entry, snapshot.header.snapshot_version)
        if targets is not None:
            _assert_target_facts(code, entry, snapshot.header, targets)
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
    """Read a curator's completed worksheet, refusing a shape it cannot hold.

    The only hand-edited artifact in the chain, and the one that went in
    unchecked: a spreadsheet round-trip that dropped a column surfaced as a bare
    ``KeyError`` from inside the apply, and an added one was read as data no
    reader accounts for. Same rule as :func:`row_from_json`, one layer out.
    """
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        _assert_columns(path, reader.fieldnames)
        rows = [r for r in reader if (r.get("gold_label") or "").strip()]
    labels: dict[str, dict[str, str]] = {}
    for row in rows:
        code = (row.get("oncotree_code") or "").strip()
        if code in labels:
            raise ValueError(f"{code}: worksheet carries more than one label")
        labels[code] = {k: (v or "").strip() for k, v in row.items()}
    return labels


def _assert_columns(path: Path, fieldnames: Sequence[str] | None) -> None:
    if not fieldnames:
        raise ValueError(f"{path}: the worksheet has no header row")
    present = {(name or "").strip() for name in fieldnames}
    missing = sorted(set(WORKSHEET_COLUMNS) - present)
    unknown = sorted(present - set(WORKSHEET_COLUMNS))
    if missing or unknown:
        raise ValueError(
            f"{path}: the worksheet columns do not match the shipped worksheet — "
            f"missing {missing}, unknown {unknown}. Re-export from the worksheet "
            "the snapshot shipped rather than reshaping it."
        )


def worksheet_concept_ids(worksheet: str | Path) -> set[int]:
    """Every ``gold_concept_id`` a completed worksheet names.

    Read before applying so the target facts can be fetched for exactly these
    concepts: the vocabulary holds ~10M rows and a worksheet names a few dozen,
    so loading the table to validate them is gigabytes for a handful of lookups.
    """
    ids: set[int] = set()
    for code, entry in _read_worksheet(Path(worksheet)).items():
        raw = entry.get("gold_concept_id", "")
        if not raw:
            continue
        if not raw.isdigit():
            raise ValueError(f"{code}: gold_concept_id {raw!r} is not a concept id")
        ids.add(int(raw))
    return ids


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


def _assert_snapshot_pin(code: str, entry: dict[str, str], version: str) -> None:
    """Labels belong to ONE snapshot; applied to another they describe other data."""
    stamped = entry.get("snapshot_version", "")
    if not stamped:
        raise ValueError(
            f"{code}: the row carries no snapshot_version — rebuild the worksheet "
            f"from {version} rather than editing an older one"
        )
    if stamped != version:
        raise ValueError(
            f"{code}: labelled against snapshot {stamped}, but this is {version}; "
            "the frozen populations and row counts differ between them"
        )


def _assert_target_facts(
    code: str,
    entry: dict[str, str],
    header: GoldSetHeader,
    targets: Mapping[int, TargetFacts],
) -> None:
    if GoldLabel(entry["gold_label"]) is not GoldLabel.RESOLVED:
        return
    concept_id = int(entry["gold_concept_id"])
    facts = targets.get(concept_id)
    if facts is None:
        raise ValueError(
            f"{code}: gold_concept_id {concept_id} is absent from "
            f"{header.vocab_release}"
        )
    if facts.concept_code != entry["target_concept_code"]:
        raise ValueError(
            f"{code}: concept {concept_id} carries code {facts.concept_code}, not "
            f"the {entry['target_concept_code']} the worksheet names"
        )
    if facts.vocabulary != header.target_vocabulary:
        raise ValueError(
            f"{code}: concept {concept_id} is {facts.vocabulary}, not "
            f"{header.target_vocabulary}"
        )
    if facts.domain != header.target_domain:
        raise ValueError(
            f"{code}: concept {concept_id} is in the {facts.domain} domain, not "
            f"{header.target_domain}"
        )
    if not facts.standard:
        raise ValueError(
            f"{code}: concept {concept_id} is not standard, so it cannot be the "
            "target a mapping is graded against"
        )

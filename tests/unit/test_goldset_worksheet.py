"""US-002 / G-05: the labelling worksheet and the path labels take back in.

Two anti-anchoring rules shape this, both from D4. The resolver's answers ARE the
``vocabulary_omop`` answers, so a worksheet pre-filled with "candidate OMOP
concepts" hands the reviewer the resolver's output to rubber-stamp — self-grading
with a human signature. And the 12 challenge codes were selected *because the
resolver was unsure*: a reviewer told that labels differently, so the worksheet
must neither say so nor let the stratum be picked out by eye.

Labels come back in as a NEW snapshot. Nothing here can write a label that was
not in the human's file.
"""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import pytest

from sema.eval.goldset_ops import label_projection_digest
from sema.eval.goldset_snapshot import GoldSetSnapshot
from sema.eval.goldset_snapshot_utils import GoldSetHeader, UniverseEntry
from sema.eval.goldset_source import SourceKind, SourceSpec
from sema.eval.goldset_worksheet import (
    WORKSHEET_COLUMNS,
    TargetFacts,
    apply_labels,
    build_worksheet,
    worksheet_codes,
)
from sema.eval.mapping_goldset_utils import GoldLabel, GoldRow, TierState

pytestmark = pytest.mark.unit


_SPEC = SourceSpec(
    kind=SourceKind.STAGING,
    table="sema_staging.condition_staging",
    code_column="source_oncotree_code",
    scope_column="source_schema",
    scope_values=("study_a",),
)

_HEAD = ["LUAD", "COAD", "IDC", "PAAD"]
_CHALLENGE = ["UESL", "BTOV"]


def _snapshot() -> GoldSetSnapshot:
    codes = _HEAD + _CHALLENGE
    counts = {"LUAD": 1000, "COAD": 500, "IDC": 400, "PAAD": 300, "UESL": 3, "BTOV": 1}
    header = GoldSetHeader(
        snapshot_version="v1",
        snapshot_date="2026-08-11",
        source_of_truth=_SPEC,
        target_vocabulary="SNOMED",
        target_domain="Condition",
        vocab_release="omop-vocab-2024",
        oracle_source="human curation",
        oracle_version="unlabelled",
        tier_codes=tuple(_HEAD),
        challenge_codes=tuple(_CHALLENGE),
        tier_target_row_share=0.95,
        tier_achieved_row_share=0.99,
        min_frozen_tier_row_share=0.90,
        max_unseen_code_share=0.10,
    )
    rows = [
        GoldRow(
            c, None, GoldLabel.UNLABELLED, counts[c],
            tier_state=TierState.IN_TIER if c in _HEAD else TierState.CHALLENGE,
        )
        for c in codes
    ]
    universe = tuple(UniverseEntry(c, counts[c]) for c in codes)
    return GoldSetSnapshot(header=header, rows=rows, universe=universe)


def _worksheet(tmp_path: Path, head_size: int = 3) -> list[dict[str, str]]:
    path = tmp_path / "worksheet.csv"
    build_worksheet(_snapshot(), path, head_size=head_size, source_names={"LUAD": "Lung Adenocarcinoma"})
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_the_worksheet_covers_the_head_prefix_plus_every_challenge_code(tmp_path: Path) -> None:
    codes = {row["oncotree_code"] for row in _worksheet(tmp_path)}

    assert codes == {"LUAD", "COAD", "IDC", "UESL", "BTOV"}


def test_the_worksheet_never_pre_fills_a_target_concept(tmp_path: Path) -> None:
    """Pre-filled candidates are the resolver's own output wearing a signature."""
    for row in _worksheet(tmp_path):
        assert row["gold_concept_id"] == ""
        assert row["target_concept_code"] == ""
        assert row["gold_label"] == ""


def test_the_worksheet_carries_source_side_context_only(tmp_path: Path) -> None:
    rows = {r["oncotree_code"]: r for r in _worksheet(tmp_path)}

    assert rows["LUAD"]["oncotree_name"] == "Lung Adenocarcinoma"
    assert set(WORKSHEET_COLUMNS).isdisjoint({"tier_state", "row_count", "resolver_status"})


def test_the_challenge_stratum_is_not_identifiable_from_the_worksheet() -> None:
    """Interleaved, and never ordered by the frequency that would give it away.

    Sized like the live tier 1 (50 head + 12 challenge): at that scale a stratum
    ordered by tier or by row_count would be obvious at a glance.
    """
    from sema.eval.goldset_snapshot import GoldSetSnapshot
    from sema.eval.goldset_snapshot_utils import GoldSetHeader

    head = [f"H{i:02d}" for i in range(50)]
    challenge = [f"C{i:02d}" for i in range(12)]
    counts = {c: 10_000 - i * 100 for i, c in enumerate(head)}
    counts.update({c: 12 - i for i, c in enumerate(challenge)})
    header = replace(
        _snapshot().header, tier_codes=tuple(head), challenge_codes=tuple(challenge)
    )
    snapshot = GoldSetSnapshot(
        header=header,
        rows=[
            GoldRow(
                c, None, GoldLabel.UNLABELLED, counts[c],
                tier_state=TierState.IN_TIER if c in head else TierState.CHALLENGE,
            )
            for c in head + challenge
        ],
        universe=tuple(UniverseEntry(c, counts[c]) for c in head + challenge),
    )

    codes = worksheet_codes(snapshot, head_size=50)
    positions = [i for i, c in enumerate(codes) if c in challenge]

    assert len(codes) == 62
    assert positions != list(range(50, 62)), "the stratum must not sit in a block"
    assert min(positions) < 20 and max(positions) > 40, "it must span the worksheet"
    assert codes != sorted(codes, key=lambda c: -counts[c]), "not frequency-ordered"
    assert isinstance(header, GoldSetHeader)


def test_the_worksheet_order_is_reproducible(tmp_path: Path) -> None:
    first = [r["oncotree_code"] for r in _worksheet(tmp_path)]
    again = [r["oncotree_code"] for r in _worksheet(tmp_path / "again")]

    assert first == again


# --- labels coming back in --------------------------------------------------


def _completed(tmp_path: Path, **overrides: str) -> Path:
    path = tmp_path / "done.csv"
    row = {
        "oncotree_code": "LUAD",
        "oncotree_name": "Lung Adenocarcinoma",
        "main_type": "",
        "tissue": "",
        "snapshot_version": "v1",
        "gold_label": "RESOLVED",
        "gold_concept_id": "45768916",
        "target_concept_code": "254626006",
        "curator": "dean",
        "review_date": "2026-08-12",
        "evidence": "SNOMED browser: primary adenocarcinoma of lung",
        "second_reviewer": "sam",
        "notes": "",
    }
    row.update(overrides)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=WORKSHEET_COLUMNS)
        writer.writeheader()
        writer.writerow(row)
    return path


def test_applying_a_worksheet_labels_only_what_the_human_wrote(tmp_path: Path) -> None:
    draft = apply_labels(_snapshot(), _completed(tmp_path), version="v2", date="2026-08-12")
    by_code = draft.by_code()

    assert by_code["LUAD"].gold_label is GoldLabel.RESOLVED
    assert by_code["LUAD"].gold_concept_id == 45768916
    assert by_code["LUAD"].curator == "dean"
    assert by_code["COAD"].gold_label is GoldLabel.UNLABELLED
    assert by_code["COAD"].gold_concept_id is None


def test_a_no_map_must_justify_the_absence_of_a_target(tmp_path: Path) -> None:
    """A NO_MAP is a positive claim, and a wrong one scores fp_map directly."""
    path = _completed(
        tmp_path, gold_label="NO_MAP", gold_concept_id="", target_concept_code="", evidence=""
    )

    with pytest.raises(ValueError, match="evidence"):
        apply_labels(_snapshot(), path, version="v2", date="2026-08-12")


def test_a_resolved_label_without_a_concept_is_rejected(tmp_path: Path) -> None:
    path = _completed(tmp_path, gold_concept_id="")

    with pytest.raises(ValueError, match="LUAD"):
        apply_labels(_snapshot(), path, version="v2", date="2026-08-12")


def test_a_label_for_a_code_outside_the_snapshot_is_rejected(tmp_path: Path) -> None:
    path = _completed(tmp_path, oncotree_code="NOT_IN_SCOPE")

    with pytest.raises(ValueError, match="NOT_IN_SCOPE"):
        apply_labels(_snapshot(), path, version="v2", date="2026-08-12")


def test_applying_an_empty_worksheet_changes_no_label(tmp_path: Path) -> None:
    empty = tmp_path / "empty.csv"
    with empty.open("w", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=WORKSHEET_COLUMNS).writeheader()

    draft = apply_labels(_snapshot(), empty, version="v2", date="2026-08-12")

    assert label_projection_digest(draft.rows) == label_projection_digest(_snapshot().rows)


# --- the worksheet carries the pins -----------------------------------------


_TARGETS = {
    45768916: TargetFacts(
        concept_code="254626006",
        vocabulary="SNOMED",
        domain="Condition",
        standard=True,
    )
}


def test_the_worksheet_stamps_the_snapshot_it_was_built_from(tmp_path: Path) -> None:
    for row in _worksheet(tmp_path):
        assert row["snapshot_version"] == "v1"


def test_labels_cannot_be_applied_to_a_different_snapshot(tmp_path: Path) -> None:
    """62 labels applied to the wrong snapshot are 62 labels about other data."""
    path = _completed(tmp_path, snapshot_version="v0")

    with pytest.raises(ValueError, match="v0"):
        apply_labels(_snapshot(), path, version="v2", date="2026-08-12")


def test_an_unstamped_worksheet_row_is_refused(tmp_path: Path) -> None:
    path = _completed(tmp_path, snapshot_version="")

    with pytest.raises(ValueError, match="snapshot_version"):
        apply_labels(_snapshot(), path, version="v2", date="2026-08-12")


def test_the_worksheet_ships_its_own_instructions(tmp_path: Path) -> None:
    """The pins a curator must honour cannot live only in a reviewer's memory."""
    path = tmp_path / "worksheet.csv"
    build_worksheet(_snapshot(), path)
    instructions = (tmp_path / "worksheet.csv.instructions.md").read_text(encoding="utf-8")

    assert "v1" in instructions
    assert "SNOMED" in instructions and "Condition" in instructions
    assert "omop-vocab-2024" in instructions
    assert "RESOLVED" in instructions and "NO_MAP" in instructions
    assert "evidence" in instructions


def test_a_label_is_checked_against_the_target_it_names(tmp_path: Path) -> None:
    draft = apply_labels(
        _snapshot(), _completed(tmp_path), version="v2", date="2026-08-12",
        targets=_TARGETS,
    )

    assert draft.by_code()["LUAD"].gold_concept_id == 45768916


def test_a_concept_id_that_does_not_carry_the_named_code_is_rejected(
    tmp_path: Path,
) -> None:
    path = _completed(tmp_path, target_concept_code="999999")

    with pytest.raises(ValueError, match="254626006"):
        apply_labels(
            _snapshot(), path, version="v2", date="2026-08-12", targets=_TARGETS
        )


def test_a_concept_from_the_wrong_vocabulary_is_rejected(tmp_path: Path) -> None:
    targets = {45768916: replace(_TARGETS[45768916], vocabulary="ICD10CM")}

    with pytest.raises(ValueError, match="SNOMED"):
        apply_labels(
            _snapshot(), _completed(tmp_path), version="v2", date="2026-08-12",
            targets=targets,
        )


def test_a_concept_from_the_wrong_domain_is_rejected(tmp_path: Path) -> None:
    targets = {45768916: replace(_TARGETS[45768916], domain="Observation")}

    with pytest.raises(ValueError, match="Condition"):
        apply_labels(
            _snapshot(), _completed(tmp_path), version="v2", date="2026-08-12",
            targets=targets,
        )


def test_a_non_standard_concept_is_rejected(tmp_path: Path) -> None:
    targets = {45768916: replace(_TARGETS[45768916], standard=False)}

    with pytest.raises(ValueError, match="standard"):
        apply_labels(
            _snapshot(), _completed(tmp_path), version="v2", date="2026-08-12",
            targets=targets,
        )


def test_a_concept_id_absent_from_the_vocabulary_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="45768916"):
        apply_labels(
            _snapshot(), _completed(tmp_path), version="v2", date="2026-08-12",
            targets={},
        )

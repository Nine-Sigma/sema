"""US-002 / G-01+G-03: the gold set is a frozen, self-describing snapshot.

Covers the artifact invariants asserted on load (unique codes, disjoint and
exhaustive tier states, canonical ordering, label/concept consistency, the
per-label annotation floor) and the ordered-row digest — which is taken over the
row LIST, not a dict, because ``by_code()`` last-wins on duplicates and a digest
over the mapping is blind to exactly the corruption it guards.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sema.eval.goldset_snapshot import (
    SnapshotInvariantError,
    load_snapshot,
    resolve_current_version,
    write_snapshot,
)
from sema.eval.goldset_snapshot_utils import (
    GoldSetHeader,
    UniverseEntry,
    ordered_rows_digest,
    universe_digest,
)
from sema.eval.goldset_source import SourceKind, SourceSpec
from sema.eval.mapping_goldset_utils import GoldLabel, GoldRow, TierState

pytestmark = pytest.mark.unit


_SPEC = SourceSpec(
    kind=SourceKind.STAGING,
    table="sema_staging.condition_staging",
    code_column="source_oncotree_code",
    scope_column="source_schema",
    scope_values=("cbioportal_study_a",),
)

_UNIVERSE = (
    UniverseEntry("LUAD", 100),
    UniverseEntry("COAD", 50),
    UniverseEntry("RARE", 1),
    UniverseEntry("ODD", 1),
)


def _header(**overrides: object) -> GoldSetHeader:
    base = dict(
        snapshot_version="test-v1",
        snapshot_date="2026-08-11",
        source_of_truth=_SPEC,
        target_vocabulary="SNOMED",
        target_domain="Condition",
        vocab_release="omop-vocab-2024",
        oracle_source="human curation",
        oracle_version="unlabelled",
        tier_codes=("LUAD", "COAD"),
        challenge_codes=("ODD",),
        tier_target_row_share=0.95,
        tier_achieved_row_share=150 / 152,
        min_frozen_tier_row_share=0.90,
        max_unseen_code_share=0.10,
    )
    base.update(overrides)
    return GoldSetHeader(**base)  # type: ignore[arg-type]


def _rows() -> list[GoldRow]:
    return [
        GoldRow("LUAD", None, GoldLabel.UNLABELLED, 100, tier_state=TierState.IN_TIER),
        GoldRow("COAD", None, GoldLabel.UNLABELLED, 50, tier_state=TierState.IN_TIER),
        GoldRow("ODD", None, GoldLabel.UNLABELLED, 1, tier_state=TierState.CHALLENGE),
        GoldRow("RARE", None, GoldLabel.UNLABELLED, 1, tier_state=TierState.OUT_OF_TIER),
        GoldRow("GONE", None, GoldLabel.UNLABELLED, 7, tier_state=TierState.RETIRED),
    ]


def _write(tmp_path: Path, rows: list[GoldRow] | None = None, **overrides: object) -> Path:
    directory = tmp_path / "snap"
    write_snapshot(directory, _header(**overrides), rows or _rows(), _UNIVERSE)
    return directory


def test_snapshot_round_trips(tmp_path: Path) -> None:
    snapshot = load_snapshot(_write(tmp_path))

    assert snapshot.header.snapshot_version == "test-v1"
    assert snapshot.header.source_of_truth == _SPEC
    assert [r.oncotree_code for r in snapshot.rows] == ["LUAD", "COAD", "ODD", "RARE", "GONE"]
    assert len(snapshot.universe) == 4


def test_duplicate_codes_are_rejected_on_load(tmp_path: Path) -> None:
    """The corruption a code->label digest cannot see: by_code() silently last-wins."""
    directory = _write(tmp_path)
    gold = directory / "oncotree_condition_slice0.jsonl"
    gold.write_text(gold.read_text() + gold.read_text().splitlines()[0] + "\n", encoding="utf-8")

    with pytest.raises(SnapshotInvariantError, match="duplicate"):
        load_snapshot(directory)


def test_ordered_row_digest_detects_a_duplicated_row() -> None:
    rows = _rows()
    assert ordered_rows_digest(rows) != ordered_rows_digest([rows[0], *rows])


def test_ordered_row_digest_detects_reordering() -> None:
    rows = _rows()
    assert ordered_rows_digest(rows) != ordered_rows_digest(list(reversed(rows)))


def test_tier_states_must_partition_the_universe_exactly(tmp_path: Path) -> None:
    """Four states, mutually exclusive: no code is in two, none is in none."""
    snapshot = load_snapshot(_write(tmp_path))
    states = snapshot.states_by_code()

    manifest_codes = {e.code for e in snapshot.universe}
    assert set(states) == manifest_codes | {"GONE"}
    assert {c for c, s in states.items() if s is TierState.RETIRED} == {"GONE"}
    assert manifest_codes.isdisjoint({"GONE"})
    buckets = [
        {c for c, s in states.items() if s is state}
        for state in TierState
    ]
    for i, left in enumerate(buckets):
        for right in buckets[i + 1:]:
            assert left.isdisjoint(right)


def test_row_state_disagreeing_with_the_header_is_rejected(tmp_path: Path) -> None:
    rows = _rows()
    rows[3] = GoldRow("RARE", None, GoldLabel.UNLABELLED, 1, tier_state=TierState.IN_TIER)

    with pytest.raises(SnapshotInvariantError, match="RARE"):
        load_snapshot(_write(tmp_path, rows))


def test_a_manifest_code_may_not_be_marked_retired(tmp_path: Path) -> None:
    rows = _rows()
    rows[3] = GoldRow("RARE", None, GoldLabel.UNLABELLED, 1, tier_state=TierState.RETIRED)

    with pytest.raises(SnapshotInvariantError, match="RARE"):
        load_snapshot(_write(tmp_path, rows))


def test_a_tier_code_missing_from_the_manifest_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(SnapshotInvariantError, match="GHOST"):
        _write(tmp_path, tier_codes=("LUAD", "COAD", "GHOST"))


def test_every_in_tier_code_must_have_a_row(tmp_path: Path) -> None:
    with pytest.raises(SnapshotInvariantError, match="COAD"):
        _write(tmp_path, [r for r in _rows() if r.oncotree_code != "COAD"])


def test_non_canonical_ordering_is_rejected(tmp_path: Path) -> None:
    directory = _write(tmp_path)
    gold = directory / "oncotree_condition_slice0.jsonl"
    lines = gold.read_text(encoding="utf-8").splitlines()
    gold.write_text("\n".join([lines[1], lines[0], *lines[2:]]) + "\n", encoding="utf-8")

    with pytest.raises(SnapshotInvariantError, match="canonical"):
        load_snapshot(directory)


def test_resolved_label_requires_a_concept_id(tmp_path: Path) -> None:
    rows = _rows()
    rows[0] = GoldRow(
        "LUAD", None, GoldLabel.RESOLVED, 100, tier_state=TierState.IN_TIER,
        curator="dean", review_date="2026-08-11", evidence="OncoTree browser",
    )

    with pytest.raises(SnapshotInvariantError, match="LUAD"):
        _write(tmp_path, rows)


def test_no_map_label_must_not_carry_a_concept_id(tmp_path: Path) -> None:
    rows = _rows()
    rows[0] = GoldRow(
        "LUAD", 1234, GoldLabel.NO_MAP, 100, tier_state=TierState.IN_TIER,
        curator="dean", review_date="2026-08-11", evidence="no acceptable target",
    )

    with pytest.raises(SnapshotInvariantError, match="LUAD"):
        _write(tmp_path, rows)


def test_a_label_without_provenance_is_rejected(tmp_path: Path) -> None:
    """The annotation floor: curator, date, and evidence per label."""
    rows = _rows()
    rows[0] = GoldRow(
        "LUAD", 45768916, GoldLabel.RESOLVED, 100, tier_state=TierState.IN_TIER,
        target_concept_code="254626006",
    )

    with pytest.raises(SnapshotInvariantError, match="curator"):
        _write(tmp_path, rows)


def test_a_resolved_label_must_carry_a_durable_concept_code(tmp_path: Path) -> None:
    """A bare concept_id does not survive a vocabulary release change."""
    rows = _rows()
    rows[0] = GoldRow(
        "LUAD", 45768916, GoldLabel.RESOLVED, 100, tier_state=TierState.IN_TIER,
        curator="dean", review_date="2026-08-11", evidence="OncoTree browser",
    )

    with pytest.raises(SnapshotInvariantError, match="target_concept_code"):
        _write(tmp_path, rows)


def test_a_labelled_snapshot_with_full_provenance_loads(tmp_path: Path) -> None:
    rows = _rows()
    rows[0] = GoldRow(
        "LUAD", 45768916, GoldLabel.RESOLVED, 100, tier_state=TierState.IN_TIER,
        target_concept_code="254626006",
        curator="dean", review_date="2026-08-11", evidence="OncoTree browser: LUAD",
    )
    snapshot = load_snapshot(_write(tmp_path, rows))

    assert snapshot.by_code()["LUAD"].target_concept_code == "254626006"


def test_tampering_with_the_universe_manifest_is_detected(tmp_path: Path) -> None:
    directory = _write(tmp_path)
    manifest = directory / "universe.jsonl"
    manifest.write_text(
        manifest.read_text(encoding="utf-8").replace('"frozen_row_count": 100', '"frozen_row_count": 999'),
        encoding="utf-8",
    )

    with pytest.raises(SnapshotInvariantError, match="universe"):
        load_snapshot(directory)


def test_tampering_with_a_row_is_detected(tmp_path: Path) -> None:
    directory = _write(tmp_path)
    gold = directory / "oncotree_condition_slice0.jsonl"
    gold.write_text(gold.read_text(encoding="utf-8").replace("COAD", "COADX"), encoding="utf-8")

    with pytest.raises(SnapshotInvariantError, match="rows"):
        load_snapshot(directory)


def test_digests_are_written_into_the_header(tmp_path: Path) -> None:
    directory = _write(tmp_path)
    header = json.loads((directory / "meta.json").read_text(encoding="utf-8"))

    assert header["rows_sha256"] == ordered_rows_digest(_rows())
    assert header["universe_sha256"] == universe_digest(_UNIVERSE)


def test_write_snapshot_refuses_to_overwrite_a_published_snapshot(tmp_path: Path) -> None:
    """'The artifact is frozen' — refresh emits a new version, never a rewrite."""
    directory = _write(tmp_path)

    with pytest.raises(SnapshotInvariantError, match="already published"):
        write_snapshot(directory, _header(), _rows(), _UNIVERSE)


def test_resolve_current_version_reads_the_pointer(tmp_path: Path) -> None:
    (tmp_path / "current.json").write_text(json.dumps({"snapshot_version": "v9"}), encoding="utf-8")

    assert resolve_current_version(tmp_path) == "v9"

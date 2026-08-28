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

from sema.eval.goldset_invariants import assert_universe_integrity
from sema.eval.goldset_snapshot import (
    SnapshotInvariantError,
    load_snapshot,
    resolve_current_version,
    write_snapshot,
)
from sema.eval.goldset_snapshot_utils import (
    GoldSetHeader,
    UniverseEntry,
    canonical_universe_key,
    file_digest,
    ordered_rows_digest,
    row_from_json,
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


def test_digests_bind_the_bytes_that_were_written(tmp_path: Path) -> None:
    """The digest must verify the ARTIFACT, not a re-projection of it.

    Digesting ``row_to_json`` output re-derives the payload from the parsed rows,
    so anything the parser drops — an injected key, a reordered field, trailing
    junk — is invisible to it. Hashing the file bytes cannot look away.
    """
    directory = _write(tmp_path)
    header = json.loads((directory / "meta.json").read_text(encoding="utf-8"))

    assert header["rows_sha256"] == file_digest(
        directory / "oncotree_condition_slice0.jsonl"
    )
    assert header["universe_sha256"] == file_digest(directory / "universe.jsonl")


def test_an_empty_digest_is_not_a_pass(tmp_path: Path) -> None:
    """An unstamped header used to SKIP verification instead of failing it."""
    directory = _write(tmp_path)
    meta = directory / "meta.json"
    header = json.loads(meta.read_text(encoding="utf-8"))
    header["rows_sha256"] = ""
    meta.write_text(json.dumps(header, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(SnapshotInvariantError, match="rows_sha256"):
        load_snapshot(directory)


def test_an_injected_row_key_is_refused_at_the_parser(tmp_path: Path) -> None:
    """A key the parser silently drops is a field the artifact cannot account for."""
    with pytest.raises(SnapshotInvariantError, match="surprise"):
        row_from_json(
            {
                "oncotree_code": "LUAD",
                "gold_concept_id": None,
                "gold_label": "UNLABELLED",
                "row_count": 1,
                "surprise": "smuggled",
            }
        )


def test_whitespace_reformatting_of_the_rows_file_is_detected(tmp_path: Path) -> None:
    directory = _write(tmp_path)
    rows_file = directory / "oncotree_condition_slice0.jsonl"
    rows_file.write_text(
        rows_file.read_text(encoding="utf-8").replace('", "', '",  "'), encoding="utf-8"
    )

    with pytest.raises(SnapshotInvariantError, match="rows_sha256"):
        load_snapshot(directory)


def test_write_snapshot_refuses_to_overwrite_a_published_snapshot(tmp_path: Path) -> None:
    """'The artifact is frozen' — refresh emits a new version, never a rewrite."""
    directory = _write(tmp_path)

    with pytest.raises(SnapshotInvariantError, match="already published"):
        write_snapshot(directory, _header(), _rows(), _UNIVERSE)


def test_resolve_current_version_reads_the_pointer(tmp_path: Path) -> None:
    (tmp_path / "current.json").write_text(json.dumps({"snapshot_version": "v9"}), encoding="utf-8")

    assert resolve_current_version(tmp_path) == "v9"


# --- the universe manifest is a contract too ---------------------------------


def _write_universe(tmp_path: Path, universe: tuple[UniverseEntry, ...]) -> Path:
    directory = tmp_path / "snap"
    write_snapshot(directory, _header(), _rows(), universe)
    return directory


def test_a_duplicated_universe_code_is_refused(tmp_path: Path) -> None:
    """Consumers disagree on a duplicate: universe_row_total() sums both entries
    while the drift report's dict last-wins, so the same artifact yields two
    different denominators. A digest over what is on disk cannot see the conflict."""
    universe = (*_UNIVERSE, UniverseEntry("LUAD", 999))

    with pytest.raises(SnapshotInvariantError, match="duplicate universe code: LUAD"):
        _write_universe(tmp_path, universe)


def test_a_negative_frozen_row_count_is_refused(tmp_path: Path) -> None:
    universe = (UniverseEntry("LUAD", 100), UniverseEntry("COAD", 50),
                UniverseEntry("RARE", -1), UniverseEntry("ODD", 1))

    with pytest.raises(SnapshotInvariantError, match="negative"):
        _write_universe(tmp_path, universe)


def test_an_out_of_order_universe_on_disk_is_refused() -> None:
    """Canonical order is what makes the manifest diffable across snapshots. Checked
    on the invariant directly: a reordered FILE trips the digest first, so this is
    the only way to reach the rule."""
    universe = (UniverseEntry("COAD", 50), UniverseEntry("LUAD", 100))

    with pytest.raises(SnapshotInvariantError, match="canonical order"):
        assert_universe_integrity(universe)


def test_write_snapshot_normalizes_the_universe_order(tmp_path: Path) -> None:
    """The writer normalizes and the reader verifies, as it already does for rows."""
    directory = _write_universe(
        tmp_path, (UniverseEntry("COAD", 50), UniverseEntry("LUAD", 100),
                   UniverseEntry("RARE", 1), UniverseEntry("ODD", 1))
    )

    assert [e.code for e in load_snapshot(directory).universe] == ["LUAD", "COAD", "ODD", "RARE"]


def test_a_duplicated_frozen_population_member_is_refused(tmp_path: Path) -> None:
    """len(tier_codes) is reported as the tier size, so a duplicate inflates it."""
    with pytest.raises(SnapshotInvariantError, match="duplicate"):
        write_snapshot(
            tmp_path / "snap", _header(tier_codes=("LUAD", "COAD", "LUAD")),
            _rows(), _UNIVERSE,
        )


def test_a_tier_code_in_both_frozen_populations_is_allowed(tmp_path: Path) -> None:
    """Live case IMMC: declared in both lists, graded once, in the head."""
    directory = tmp_path / "snap"
    write_snapshot(directory, _header(challenge_codes=("ODD", "LUAD")), _rows(), _UNIVERSE)

    assert load_snapshot(directory).by_code()["LUAD"].tier_state is TierState.IN_TIER


def test_an_achieved_tier_share_that_contradicts_the_manifest_is_refused(
    tmp_path: Path,
) -> None:
    """observe() carried this forward while replacing every count beneath it."""
    with pytest.raises(SnapshotInvariantError, match="achieved_row_share"):
        write_snapshot(
            tmp_path / "snap", _header(tier_achieved_row_share=0.42), _rows(), _UNIVERSE
        )


def test_a_declared_tier_code_absent_from_the_universe_must_be_retired(
    tmp_path: Path,
) -> None:
    """A code that left the scope is retired, not a broken declaration — but it must
    still carry a row, else the label it earned is simply gone."""
    universe = (UniverseEntry("LUAD", 100), UniverseEntry("ODD", 1), UniverseEntry("RARE", 1))
    rows = [r for r in _rows() if r.oncotree_code != "COAD"]

    with pytest.raises(SnapshotInvariantError, match="COAD"):
        write_snapshot(tmp_path / "snap", _header(tier_achieved_row_share=100 / 102), rows, universe)


def test_an_interrupted_publish_leaves_no_directory_to_block_the_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A partial snapshot satisfied the "already published" refusal forever.

    ``write_eval_run`` was made all-or-nothing for exactly this; the snapshot
    writer is the artifact with a version number it can permanently burn.
    """
    import sema.eval.goldset_snapshot as module

    calls: list[Path] = []

    def _explode(path: Path, payload: object) -> None:
        calls.append(path)
        if len(calls) == 2:
            raise OSError("disk full")
        path.write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(module, "_write_jsonl", _explode)
    directory = tmp_path / "snap"

    with pytest.raises(OSError, match="disk full"):
        write_snapshot(directory, _header(), _rows(), _UNIVERSE)

    assert not directory.exists()
    assert list(tmp_path.iterdir()) == []


def test_a_header_that_declares_no_vocabulary_release_is_rejected(tmp_path: Path) -> None:
    """The release pin is skipped when the header declares nothing to pin to.

    Same shape as the blank ``rows_sha256`` hole: an undeclared release made an
    unpinnable snapshot indistinguishable from a pinned one, and grading across
    releases reports vocabulary churn as resolver error.
    """
    with pytest.raises(SnapshotInvariantError, match="vocab_release"):
        write_snapshot(tmp_path / "snap", _header(vocab_release=""), _rows(), _UNIVERSE)


def test_losing_a_publish_race_is_the_already_published_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two writers both pass the exists check; the loser must not get a raw OSError."""
    import sema.eval.goldset_snapshot_utils as utils

    directory = tmp_path / "snap"

    def _lose(src: object, dst: object) -> None:
        directory.mkdir(parents=True)
        (directory / "meta.json").write_text("{}", encoding="utf-8")
        raise OSError("directory not empty")

    monkeypatch.setattr(utils.os, "replace", _lose)

    with pytest.raises(SnapshotInvariantError, match="already published"):
        write_snapshot(directory, _header(), _rows(), _UNIVERSE)

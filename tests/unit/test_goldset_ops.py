"""US-002 / G-03+G-04: observe, re-tier, re-scope — three named operations.

They have different blast radii and are named separately for that reason:
``observe`` re-stamps weights, ``re-tier`` moves the benchmark, ``re-scope``
changes what is being measured. None may mutate a published snapshot, because
``row_count`` is the row-weighted scoring weight — rewriting it in place would
silently change historical metrics with labels and decisions unchanged.

The strongest invariant guards the likeliest self-labelling leak path: a refresh
must never write a ``gold_concept_id``. Enforced by a projection digest, not by
convention.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sema.eval.goldset_ops import (
    label_projection_digest,
    observe,
    publish,
    re_scope,
    re_tier,
    select_tier,
)
from sema.eval.goldset_snapshot import (
    SnapshotInvariantError,
    load_snapshot,
    resolve_current_version,
    snapshot_dir,
    write_snapshot,
)
from sema.eval.goldset_snapshot_utils import GoldSetHeader, UniverseEntry
from sema.eval.goldset_source import SourceKind, SourceSpec
from sema.eval.mapping_goldset_utils import GoldLabel, GoldRow, TierState

pytestmark = pytest.mark.unit


def _spec(*studies: str) -> SourceSpec:
    return SourceSpec(
        kind=SourceKind.STAGING,
        table="sema_staging.condition_staging",
        code_column="source_oncotree_code",
        scope_column="source_schema",
        scope_values=studies or ("study_a",),
    )


_OBSERVED = {"LUAD": 100, "COAD": 100, "RARE": 10}


def _header(**overrides: object) -> GoldSetHeader:
    base: dict[str, object] = dict(
        snapshot_version="v1",
        snapshot_date="2026-08-11",
        source_of_truth=_spec(),
        target_vocabulary="SNOMED",
        target_domain="Condition",
        vocab_release="omop-vocab-2024",
        oracle_source="human curation",
        oracle_version="unlabelled",
        tier_codes=("LUAD", "COAD"),
        challenge_codes=(),
        tier_target_row_share=0.95,
        tier_achieved_row_share=200 / 210,
        min_frozen_tier_row_share=0.90,
        max_unseen_code_share=0.10,
    )
    base.update(overrides)
    return GoldSetHeader(**base)  # type: ignore[arg-type]


def _rows() -> list[GoldRow]:
    return [
        GoldRow(
            "LUAD", 45768916, GoldLabel.RESOLVED, 100,
            notes="curated from the OncoTree browser",
            tier_state=TierState.IN_TIER, target_concept_code="254626006",
            curator="dean", review_date="2026-08-11", evidence="OncoTree browser: LUAD",
        ),
        GoldRow(
            "COAD", None, GoldLabel.UNLABELLED, 100,
            notes="awaiting human label; observed in an older scope",
            tier_state=TierState.IN_TIER,
        ),
        GoldRow("RARE", None, GoldLabel.UNLABELLED, 10, tier_state=TierState.OUT_OF_TIER),
    ]


@pytest.fixture()
def published(tmp_path: Path) -> Path:
    universe = tuple(UniverseEntry(c, n) for c, n in _OBSERVED.items())
    write_snapshot(snapshot_dir("v1", tmp_path), _header(), _rows(), universe)
    (tmp_path / "current.json").write_text('{"snapshot_version": "v1"}\n', encoding="utf-8")
    return tmp_path


# --- select_tier ------------------------------------------------------------


def test_select_tier_takes_the_smallest_prefix_reaching_the_target() -> None:
    codes, share = select_tier([("A", 60), ("B", 36), ("C", 3), ("D", 1)], 0.95)

    assert codes == ("A", "B")
    assert share == pytest.approx(96 / 100)


def test_select_tier_falls_back_to_the_whole_universe() -> None:
    codes, share = select_tier([("A", 1), ("B", 1)], 1.0)

    assert codes == ("A", "B")
    assert share == pytest.approx(1.0)


# --- observe ----------------------------------------------------------------


def test_observe_restamps_weights_and_preserves_labels(published: Path) -> None:
    prior = load_snapshot(snapshot_dir("v1", published))
    draft = observe(prior, {"LUAD": 250, "COAD": 100, "RARE": 10}, version="v2", date="2026-09-01")

    assert {r.oncotree_code: r.row_count for r in draft.rows}["LUAD"] == 250
    assert label_projection_digest(draft.rows) == label_projection_digest(prior.rows)
    assert draft.header.tier_codes == prior.header.tier_codes


def test_observe_never_writes_a_gold_concept_id(published: Path) -> None:
    """The refresh path is the likeliest self-labelling leak, so it gets the strongest test."""
    prior = load_snapshot(snapshot_dir("v1", published))
    draft = observe(prior, {"LUAD": 250, "COAD": 100, "RARE": 10}, version="v2", date="2026-09-01")

    labels = {r.oncotree_code: (r.gold_concept_id, r.gold_label) for r in draft.rows}
    assert labels["COAD"] == (None, GoldLabel.UNLABELLED)
    assert labels["LUAD"] == (45768916, GoldLabel.RESOLVED)


def test_observe_retires_a_head_code_that_disappears(published: Path) -> None:
    """Keeping it IN_TIER at zero rows was the worst of both: its row-weighted
    contribution vanished while it still demanded a human label for a code that is
    no longer in the scope — capping coverage below 1.0 permanently."""
    prior = load_snapshot(snapshot_dir("v1", published))
    draft = observe(prior, {"LUAD": 250, "RARE": 10}, version="v2", date="2026-09-01")
    retired = draft.by_code()["COAD"]

    assert retired.tier_state is TierState.RETIRED
    assert "COAD" not in {e.code for e in draft.universe}
    assert draft.header.tier_codes == prior.header.tier_codes, "the DECLARATION is not rewritten"


def test_observe_preserves_a_retired_codes_last_known_weight(published: Path) -> None:
    prior = load_snapshot(snapshot_dir("v1", published))
    draft = observe(prior, {"COAD": 100, "RARE": 10}, version="v2", date="2026-09-01")

    assert draft.by_code()["LUAD"].row_count == 100, "a zeroed weight is not a measurement"
    assert draft.by_code()["LUAD"].gold_concept_id == 45768916


def test_observe_recomputes_the_achieved_tier_share(published: Path) -> None:
    """It carried the prior share forward while replacing every count beneath it, so
    a published snapshot contradicted its own universe manifest."""
    prior = load_snapshot(snapshot_dir("v1", published))
    draft = observe(prior, {"LUAD": 100, "COAD": 100, "RARE": 800}, version="v2", date="2026-09-01")

    assert draft.header.tier_achieved_row_share == pytest.approx(200 / 1000)
    assert draft.header.tier_achieved_row_share != prior.header.tier_achieved_row_share


def test_observe_is_idempotent(published: Path) -> None:
    prior = load_snapshot(snapshot_dir("v1", published))
    first = observe(prior, _OBSERVED, version="v2", date="2026-09-01")
    second = observe(prior, _OBSERVED, version="v2", date="2026-09-01")

    assert first.rows == second.rows
    assert first.universe == second.universe


def test_observe_refreshes_stale_scaffold_notes(published: Path) -> None:
    prior = load_snapshot(snapshot_dir("v1", published))
    draft = observe(prior, _OBSERVED, version="v2", date="2026-09-01")
    by_code = draft.by_code()

    assert "older scope" not in by_code["COAD"].notes
    assert "study_a" in by_code["COAD"].notes
    assert by_code["LUAD"].notes == "curated from the OncoTree browser", "human notes are kept"


# --- re-tier ----------------------------------------------------------------


def test_re_tier_moves_the_benchmark_and_keeps_displaced_rows(published: Path) -> None:
    prior = load_snapshot(snapshot_dir("v1", published))
    draft = re_tier(
        prior, {"LUAD": 100, "COAD": 1, "RARE": 1}, version="v2", date="2026-09-01",
        target_row_share=0.95,
    )

    assert draft.header.tier_codes == ("LUAD",)
    assert draft.by_code()["COAD"].tier_state is TierState.OUT_OF_TIER
    assert label_projection_digest(draft.rows) == label_projection_digest(prior.rows)


def test_re_tier_admits_new_codes_as_unlabelled(published: Path) -> None:
    prior = load_snapshot(snapshot_dir("v1", published))
    draft = re_tier(
        prior, {"NEW": 500, "LUAD": 100, "COAD": 100, "RARE": 10}, version="v2",
        date="2026-09-01", target_row_share=0.95,
    )
    new_row = draft.by_code()["NEW"]

    assert new_row.tier_state is TierState.IN_TIER
    assert new_row.gold_label is GoldLabel.UNLABELLED
    assert new_row.gold_concept_id is None


def test_re_tier_can_declare_a_challenge_stratum(published: Path) -> None:
    prior = load_snapshot(snapshot_dir("v1", published))
    draft = re_tier(
        prior, {"LUAD": 100, "COAD": 1, "RARE": 1}, version="v2", date="2026-09-01",
        target_row_share=0.95, challenge_codes=("RARE",),
    )

    assert draft.by_code()["RARE"].tier_state is TierState.CHALLENGE
    assert draft.header.challenge_codes == ("RARE",)


# --- re-scope ---------------------------------------------------------------


def test_re_scope_retires_a_code_that_left_the_scope(published: Path) -> None:
    """A code that disappears is retired, never deleted — deleting discards labour."""
    prior = load_snapshot(snapshot_dir("v1", published))
    draft = re_scope(
        prior, {"COAD": 100, "RARE": 10}, _spec("study_b"), version="v2",
        date="2026-09-01", target_row_share=0.95,
    )
    retired = draft.by_code()["LUAD"]

    assert retired.tier_state is TierState.RETIRED
    assert retired.gold_concept_id == 45768916, "the human label survives retirement"
    assert "LUAD" not in {e.code for e in draft.universe}


def test_re_scope_records_the_new_source_of_truth(published: Path) -> None:
    prior = load_snapshot(snapshot_dir("v1", published))
    draft = re_scope(
        prior, {"COAD": 100}, _spec("study_b"), version="v2", date="2026-09-01",
        target_row_share=0.95,
    )

    assert draft.header.source_of_truth.scope_values == ("study_b",)
    assert draft.header.snapshot_date == "2026-09-01"


# --- publishing -------------------------------------------------------------


def test_publish_leaves_the_prior_snapshot_byte_identical(published: Path) -> None:
    prior_dir = snapshot_dir("v1", published)
    before = {p.name: p.read_bytes() for p in sorted(prior_dir.iterdir())}
    prior = load_snapshot(prior_dir)

    publish(published, observe(prior, {"LUAD": 999, "COAD": 100, "RARE": 10},
                               version="v2", date="2026-09-01"))

    assert {p.name: p.read_bytes() for p in sorted(prior_dir.iterdir())} == before


def test_publish_advances_the_current_pointer(published: Path) -> None:
    prior = load_snapshot(snapshot_dir("v1", published))

    publish(published, observe(prior, _OBSERVED, version="v2", date="2026-09-01"))

    assert resolve_current_version(published) == "v2"
    assert load_snapshot(snapshot_dir("v2", published)).header.snapshot_version == "v2"


def test_publishing_over_an_existing_version_is_refused(published: Path) -> None:
    prior = load_snapshot(snapshot_dir("v1", published))

    with pytest.raises(SnapshotInvariantError, match="already published"):
        publish(published, observe(prior, _OBSERVED, version="v1", date="2026-09-01"))


def test_a_published_draft_reloads_clean(published: Path) -> None:
    prior = load_snapshot(snapshot_dir("v1", published))
    publish(published, re_tier(prior, {"LUAD": 100, "COAD": 1, "RARE": 1},
                               version="v2", date="2026-09-01", target_row_share=0.95))

    reloaded = load_snapshot(snapshot_dir("v2", published))

    assert label_projection_digest(reloaded.rows) == label_projection_digest(prior.rows)

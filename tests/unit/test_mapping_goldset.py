"""US-002: Slice-0 OncoTree->SNOMED gold set + mapping eval harness.

Hermetic unit tests for the §1.5(f) confusion matrix. A tiny synthetic gold
set + decision set with known Zone assignments must compute mapped_precision,
mapped_recall, auto_resolution_rate, and no_map_accuracy to the §1.5 values.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from sema.eval.mapping_goldset import (
    GoldSet,
    GoldSetReport,
    load_gold_set,
    score,
)
from sema.eval.mapping_goldset_utils import (
    Decision,
    GoldLabel,
    GoldRow,
    ResolutionStatus,
    Zone,
    classify_cell,
    derive_zone,
    frequency_bucket,
)
from sema.models.planner.lifecycle import Status

pytestmark = pytest.mark.unit


# --- Zone derivation (§1.5(c)) ----------------------------------------------


def test_derive_zone_auto_accepted_resolved_is_zone1() -> None:
    z = derive_zone(Status.auto_accepted, ResolutionStatus.RESOLVED, 4314337)
    assert z is Zone.ZONE_1


def test_derive_zone_human_pinned_resolved_is_zone1() -> None:
    z = derive_zone(Status.human_pinned, ResolutionStatus.RESOLVED, 1)
    assert z is Zone.ZONE_1


def test_derive_zone_no_map_is_zone3_regardless_of_status() -> None:
    z = derive_zone(Status.auto_accepted, ResolutionStatus.NO_MAP, None)
    assert z is Zone.ZONE_3


def test_derive_zone_candidate_is_zone2() -> None:
    z = derive_zone(Status.candidate, ResolutionStatus.RESOLVED, 99)
    assert z is Zone.ZONE_2


def test_derive_zone_resolved_with_null_concept_is_not_zone1() -> None:
    z = derive_zone(Status.auto_accepted, ResolutionStatus.RESOLVED, None)
    assert z is not Zone.ZONE_1


# --- Cell classification (§1.5(f)) ------------------------------------------


def _gold(code: str, concept: int | None, label: GoldLabel) -> GoldRow:
    return GoldRow(
        oncotree_code=code,
        gold_concept_id=concept,
        gold_label=label,
        row_count=1,
        notes="",
    )


def test_classify_gold_resolved_zone1_match_is_tp() -> None:
    g = _gold("LUAD", 4314337, GoldLabel.RESOLVED)
    assert classify_cell(g, Zone.ZONE_1, 4314337) == "tp"


def test_classify_gold_resolved_zone1_mismatch_is_wrong() -> None:
    g = _gold("LUAD", 4314337, GoldLabel.RESOLVED)
    assert classify_cell(g, Zone.ZONE_1, 999) == "wrong"


def test_classify_gold_resolved_zone3_is_fn() -> None:
    g = _gold("LUAD", 4314337, GoldLabel.RESOLVED)
    assert classify_cell(g, Zone.ZONE_3, None) == "fn"


def test_classify_gold_resolved_zone2_is_recall_miss() -> None:
    g = _gold("LUAD", 4314337, GoldLabel.RESOLVED)
    assert classify_cell(g, Zone.ZONE_2, 4314337) == "recall_miss"


def test_classify_gold_no_map_zone3_is_tn() -> None:
    g = _gold("XXXX", None, GoldLabel.NO_MAP)
    assert classify_cell(g, Zone.ZONE_3, None) == "tn"


def test_classify_gold_no_map_zone1_is_fp_map() -> None:
    g = _gold("XXXX", None, GoldLabel.NO_MAP)
    assert classify_cell(g, Zone.ZONE_1, 123) == "fp_map"


# --- Confusion matrix metrics on a known synthetic set ----------------------


def _decision(
    code: str,
    concept: int | None,
    status: Status,
    res: ResolutionStatus,
) -> Decision:
    return Decision(
        source_code=code,
        concept_id=concept,
        status=status,
        resolution_status=res,
        no_map_reason="no standard candidate" if res is ResolutionStatus.NO_MAP else None,
    )


def test_confusion_matrix_computes_section_1_5_values() -> None:
    """5 gold-RESOLVED + 2 gold-NO_MAP with known predictions.

    Predictions (no Zone-2, so the documented TP/(TP+WRONG+FN) recall
    denominator equals the full gold-RESOLVED denominator):
      A resolved->correct (TP), B resolved->correct (TP),
      C resolved->wrong concept (WRONG), D resolved->NO_MAP (FN),
      E resolved->correct (TP),
      X no_map->NO_MAP (TN), Y no_map->resolved concept (FP_map)
    => TP=3 WRONG=1 FN=1 TN=1 FP_map=1
      mapped_precision = 3/(3+1+1) = 0.6
      mapped_recall    = 3/(3+1+1) = 0.6
      auto_res_rate    = (3+1+1)/7 = 5/7
      no_map_accuracy  = 1/(1+1)   = 0.5  (reported SEPARATELY)
    """
    gold = [
        _gold("A", 10, GoldLabel.RESOLVED),
        _gold("B", 20, GoldLabel.RESOLVED),
        _gold("C", 30, GoldLabel.RESOLVED),
        _gold("D", 40, GoldLabel.RESOLVED),
        _gold("E", 50, GoldLabel.RESOLVED),
        _gold("X", None, GoldLabel.NO_MAP),
        _gold("Y", None, GoldLabel.NO_MAP),
    ]
    decisions = [
        _decision("A", 10, Status.auto_accepted, ResolutionStatus.RESOLVED),
        _decision("B", 20, Status.auto_accepted, ResolutionStatus.RESOLVED),
        _decision("C", 31, Status.auto_accepted, ResolutionStatus.RESOLVED),
        _decision("D", None, Status.auto_accepted, ResolutionStatus.NO_MAP),
        _decision("E", 50, Status.human_pinned, ResolutionStatus.RESOLVED),
        _decision("X", None, Status.auto_accepted, ResolutionStatus.NO_MAP),
        _decision("Y", 77, Status.auto_accepted, ResolutionStatus.RESOLVED),
    ]
    report = score(gold, decisions)
    m = report.distinct_code
    assert (m.tp, m.wrong, m.fn, m.tn, m.fp_map) == (3, 1, 1, 1, 1)
    assert m.mapped_precision == pytest.approx(0.6)
    assert m.mapped_recall == pytest.approx(0.6)
    assert m.auto_resolution_rate == pytest.approx(5 / 7)
    assert m.no_map_accuracy == pytest.approx(0.5)


def test_no_map_predicted_for_real_concept_is_fn_never_dropped() -> None:
    gold = [_gold("D", 40, GoldLabel.RESOLVED)]
    decisions = [_decision("D", None, Status.auto_accepted, ResolutionStatus.NO_MAP)]
    m = score(gold, decisions).distinct_code
    assert m.fn == 1
    assert m.tn == 0  # a gold-RESOLVED NO_MAP prediction is FN, not TN


def test_no_map_predicted_for_no_map_gold_is_tn() -> None:
    gold = [_gold("X", None, GoldLabel.NO_MAP)]
    decisions = [_decision("X", None, Status.auto_accepted, ResolutionStatus.NO_MAP)]
    m = score(gold, decisions).distinct_code
    assert m.tn == 1
    assert m.fn == 0


def test_zone2_prediction_for_gold_resolved_is_recall_miss() -> None:
    gold = [
        _gold("A", 10, GoldLabel.RESOLVED),
        _gold("B", 20, GoldLabel.RESOLVED),
    ]
    decisions = [
        _decision("A", 10, Status.auto_accepted, ResolutionStatus.RESOLVED),
        _decision("B", 20, Status.candidate, ResolutionStatus.RESOLVED),  # Zone-2 tie
    ]
    m = score(gold, decisions).distinct_code
    assert m.recall_miss == 1
    # recall denominator includes the recall miss: 1/(1+0+0+1) = 0.5
    assert m.mapped_recall == pytest.approx(0.5)
    # the Zone-2 code is NOT a Zone-1 auto-accept
    assert m.auto_resolution_rate == pytest.approx(0.5)


def test_metrics_none_on_empty_denominator() -> None:
    m = score([], []).distinct_code
    assert m.mapped_precision is None
    assert m.mapped_recall is None
    assert m.no_map_accuracy is None


# --- Granularity: distinct-code vs row-weighted vs frequency bucket ----------


def test_row_weighted_differs_from_distinct_code() -> None:
    gold = [
        GoldRow("BIG", 10, GoldLabel.RESOLVED, 1000, ""),
        GoldRow("SMALL", 20, GoldLabel.RESOLVED, 1, ""),
    ]
    decisions = [
        _decision("BIG", 10, Status.auto_accepted, ResolutionStatus.RESOLVED),  # TP
        _decision("SMALL", 99, Status.auto_accepted, ResolutionStatus.RESOLVED),  # WRONG
    ]
    report = score(gold, decisions)
    assert report.distinct_code.mapped_precision == pytest.approx(0.5)
    # row-weighted: TP weight 1000, WRONG weight 1 -> 1000/1001
    assert report.row_weighted.mapped_precision == pytest.approx(1000 / 1001)


def test_frequency_buckets_partition_codes() -> None:
    assert frequency_bucket(5000) == "high"
    assert frequency_bucket(500) == "medium"
    assert frequency_bucket(5) == "low"


def test_report_has_per_bucket_matrices() -> None:
    gold = [
        GoldRow("BIG", 10, GoldLabel.RESOLVED, 5000, ""),
        GoldRow("SMALL", 20, GoldLabel.RESOLVED, 5, ""),
    ]
    decisions = [
        _decision("BIG", 10, Status.auto_accepted, ResolutionStatus.RESOLVED),
        _decision("SMALL", 20, Status.auto_accepted, ResolutionStatus.RESOLVED),
    ]
    report = score(gold, decisions)
    assert "high" in report.per_bucket
    assert "low" in report.per_bucket
    assert report.per_bucket["high"].tp == 1
    assert report.per_bucket["low"].tp == 1


# --- Gold set artifact loading + coverage -----------------------------------

_GOLD_PATH = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "data"
    / "gold"
    / "oncotree_condition_slice0.jsonl"
)


def test_gold_set_artifact_exists_and_loads() -> None:
    rows = load_gold_set(_GOLD_PATH)
    assert len(rows) >= 64  # every distinct observed ONCOTREE_CODE
    codes = {r.oncotree_code for r in rows}
    assert "LUAD" in codes


def test_gold_set_rows_have_required_columns() -> None:
    rows = load_gold_set(_GOLD_PATH)
    for r in rows:
        assert r.oncotree_code
        assert r.row_count >= 1
        assert r.gold_label in GoldLabel
        if r.gold_label is GoldLabel.RESOLVED:
            assert r.gold_concept_id is not None
        else:
            assert r.gold_concept_id is None


def test_gold_set_unlabelled_remainder_surfaced() -> None:
    gs = GoldSet(load_gold_set(_GOLD_PATH))
    # Ralph scaffolded the file; the human-label gate is unfinished, so the
    # unlabelled remainder must be discoverable, never silently treated labelled.
    assert gs.labelled_count + len(gs.unlabelled_codes()) == len(gs.rows)
    assert gs.coverage_fraction() == pytest.approx(
        gs.labelled_count / len(gs.rows)
    )


def test_goldset_helpers_and_serialization() -> None:
    gs = GoldSet(
        [
            _gold("A", 10, GoldLabel.RESOLVED),
            GoldRow("U", None, GoldLabel.UNLABELLED, 3, "awaiting human"),
        ]
    )
    assert gs.by_code()["A"].gold_concept_id == 10
    assert [r.oncotree_code for r in gs.labelled_rows()] == ["A"]
    assert gs.unlabelled_codes() == ["U"]
    assert GoldSet([]).coverage_fraction() == 0.0


def test_report_as_dict_is_json_serializable() -> None:
    gold = [_gold("A", 10, GoldLabel.RESOLVED)]
    decisions = [_decision("A", 10, Status.auto_accepted, ResolutionStatus.RESOLVED)]
    payload = score(gold, decisions).as_dict()
    import json

    text = json.dumps(payload)
    assert '"distinct_code"' in text
    assert payload["distinct_code"]["mapped_precision"] == 1.0


def test_distinct_oncotree_sql_builder() -> None:
    from sema.eval.mapping_goldset import distinct_oncotree_sql

    sql = distinct_oncotree_sql(["cbioportal_a", "cbioportal_b"])
    assert "cbioportal_a.sample" in sql
    assert "UNION ALL" in sql
    with pytest.raises(ValueError):
        distinct_oncotree_sql([])


def test_enumerate_distinct_codes_with_fake_cursor() -> None:
    from sema.eval.mapping_goldset import enumerate_distinct_codes

    class _Fake:
        def __init__(self) -> None:
            self._next: list[Any] = []

        def execute(self, sql: str) -> None:
            if "information_schema" in sql:
                self._next = [("cbioportal_x",)]
            else:
                assert "cbioportal_x.sample" in sql
                self._next = [("LUAD", 5957), ("COAD", 10)]

        def fetchall(self) -> list[Any]:
            return self._next

    assert enumerate_distinct_codes(_Fake()) == [("LUAD", 5957), ("COAD", 10)]


def test_enumerate_distinct_codes_no_schemas_returns_empty() -> None:
    from sema.eval.mapping_goldset import enumerate_distinct_codes

    class _Empty:
        def execute(self, sql: str) -> None:
            return None

        def fetchall(self) -> list[Any]:
            return []

    assert enumerate_distinct_codes(_Empty()) == []


def test_confusion_matrix_add_rejects_unknown_cell() -> None:
    from sema.eval.mapping_goldset_utils import ConfusionMatrix

    with pytest.raises(ValueError):
        ConfusionMatrix().add("bogus")


def test_score_excludes_unlabelled_gold_rows() -> None:
    gold = [
        _gold("A", 10, GoldLabel.RESOLVED),
        GoldRow("U", None, GoldLabel.UNLABELLED, 5, "awaiting human"),
    ]
    decisions = [
        _decision("A", 10, Status.auto_accepted, ResolutionStatus.RESOLVED),
        _decision("U", 1, Status.auto_accepted, ResolutionStatus.RESOLVED),
    ]
    report = score(gold, decisions)
    assert report.distinct_code.tp == 1
    assert report.scored_codes == 1
    assert "U" in report.unscored_unlabelled

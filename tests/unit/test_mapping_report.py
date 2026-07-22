"""US-012: Mapping eval report (confusion matrix vs gold set, thresholds).

Hermetic unit tests for the end-to-end report that runs the frozen §1.5(f)
metric module (US-002) over resolved decisions and applies the acceptance
thresholds. The report NEVER self-certifies acceptance: at less than 100%
human-labelled gold coverage it is ``provisional — not accepted``.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pytest

from sema.eval.mapping_goldset import GoldSet
from sema.eval.mapping_goldset_utils import (
    Decision,
    GoldLabel,
    GoldRow,
    ResolutionStatus,
)
from sema.eval.mapping_report import (
    build_mapping_report,
    decisions_from_store,
)
from sema.eval.mapping_report_utils import (
    MIN_AUTO_RESOLUTION_RATE,
    MIN_MAPPED_PRECISION,
    STRUCTURAL_PRECISION_CAVEAT,
    AcceptanceVerdict,
    MappingReport,
    decision_from_value_mapping,
    evaluate_acceptance,
)
from sema.models.planner.lifecycle import Status
from showcase.cbioportal_to_omop.omop_policy import OMOP_ONCOTREE_CONDITION_REF
from sema.resolve.value_mapping_store import ValueMappingStore
from sema.resolve.value_mapping_store_utils import (
    ResolutionStatus as StoreResolutionStatus,
)
from sema.resolve.value_mapping_store_utils import ValueMapping

pytestmark = pytest.mark.unit


# --- builders ---------------------------------------------------------------


def _gold(
    code: str,
    concept: int | None,
    label: GoldLabel,
    row_count: int = 1,
) -> GoldRow:
    return GoldRow(
        oncotree_code=code,
        gold_concept_id=concept,
        gold_label=label,
        row_count=row_count,
        notes="",
    )


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
        no_map_reason="dead end" if res is ResolutionStatus.NO_MAP else None,
    )


def _all_correct(codes: list[tuple[str, int]]) -> tuple[list[GoldRow], list[Decision]]:
    gold = [_gold(c, cid, GoldLabel.RESOLVED) for c, cid in codes]
    decisions = [
        _decision(c, cid, Status.auto_accepted, ResolutionStatus.RESOLVED)
        for c, cid in codes
    ]
    return gold, decisions


def _value_mapping(
    code: str,
    concept: int | None,
    res: StoreResolutionStatus,
    *,
    target_property_ref: str = "target.stage.condition_concept_id",
) -> ValueMapping:
    return ValueMapping(
        source_vocabulary="OncoTree",
        normalized_source_value=code,
        target_property_ref=target_property_ref,
        target_field="condition_concept_id",
        vocab_binding="binding.condition",
        concept_id=concept,
        vocab_release="vocab-2024",
        valid_start=None,
        valid_end=None,
        resolution_status=res,
        no_map_reason="dead end" if res is StoreResolutionStatus.NO_MAP else None,
        confidence=1.0,
        status=Status.auto_accepted,
        resolver_policy_ref=OMOP_ONCOTREE_CONDITION_REF,
        run_id="r1",
    )


# --- §1.5(f) metrics at three granularities ---------------------------------


def test_report_exposes_three_granularities() -> None:
    gold = [
        _gold("BIG", 10, GoldLabel.RESOLVED, row_count=5000),
        _gold("SMALL", 20, GoldLabel.RESOLVED, row_count=5),
    ]
    decisions = [
        _decision("BIG", 10, Status.auto_accepted, ResolutionStatus.RESOLVED),
        _decision("SMALL", 20, Status.auto_accepted, ResolutionStatus.RESOLVED),
    ]
    report = build_mapping_report(GoldSet(gold), decisions)
    assert report.score.distinct_code.mapped_precision == pytest.approx(1.0)
    assert report.score.row_weighted.mapped_precision == pytest.approx(1.0)
    assert "high" in report.score.per_bucket
    assert "low" in report.score.per_bucket


def test_no_map_accuracy_reported_separately_not_in_verdict() -> None:
    """Perfect precision+auto but poor no_map_accuracy still ACCEPTS at full cov."""
    gold = [
        _gold("A", 10, GoldLabel.RESOLVED),
        _gold("B", 20, GoldLabel.RESOLVED),
        _gold("X", None, GoldLabel.NO_MAP),
    ]
    decisions = [
        _decision("A", 10, Status.auto_accepted, ResolutionStatus.RESOLVED),
        _decision("B", 20, Status.auto_accepted, ResolutionStatus.RESOLVED),
        # gold NO_MAP but predicted a concept -> fp_map (no_map_accuracy 0.0)
        _decision("X", 99, Status.auto_accepted, ResolutionStatus.RESOLVED),
    ]
    report = build_mapping_report(GoldSet(gold), decisions)
    m = report.score.distinct_code
    # no_map_accuracy is bad but it is NOT a gating threshold
    assert m.no_map_accuracy == pytest.approx(0.0)
    # precision = tp/(tp+wrong+fp_map) = 2/3 -> below 0.95 -> RUNNING
    assert report.verdict is AcceptanceVerdict.RUNNING_NOT_ACCEPTED
    # confirm no_map_accuracy surfaces in the dict, separate from the verdict math
    assert report.as_dict()["metrics"]["distinct_code"]["no_map_accuracy"] == 0.0


# --- F1: labelled-gold contradiction predicate (gates --strict) --------------


def test_partial_labelled_gold_that_agrees_has_no_contradiction() -> None:
    # A is labelled and agrees; B is unlabelled -> coverage < 1 but NO contradiction.
    gold = [_gold("A", 10, GoldLabel.RESOLVED)]
    decisions = [
        _decision("A", 10, Status.auto_accepted, ResolutionStatus.RESOLVED),
        _decision("B", 20, Status.auto_accepted, ResolutionStatus.RESOLVED),
    ]
    report = build_mapping_report(GoldSet(gold), decisions)
    # B has no gold label; only A is scored, and it agrees -> no contradiction.
    assert report.has_labelled_contradiction() is False


def test_wrong_concept_is_a_contradiction() -> None:
    gold = [_gold("A", 10, GoldLabel.RESOLVED)]
    decisions = [_decision("A", 99, Status.auto_accepted, ResolutionStatus.RESOLVED)]
    report = build_mapping_report(GoldSet(gold), decisions)
    assert report.has_labelled_contradiction() is True


def test_gold_resolved_predicted_no_map_is_a_contradiction() -> None:
    gold = [_gold("A", 10, GoldLabel.RESOLVED)]
    decisions = [_decision("A", None, Status.auto_accepted, ResolutionStatus.NO_MAP)]
    report = build_mapping_report(GoldSet(gold), decisions)
    assert report.has_labelled_contradiction() is True


def test_gold_no_map_predicted_resolved_is_a_contradiction() -> None:
    gold = [_gold("X", None, GoldLabel.NO_MAP)]
    decisions = [_decision("X", 99, Status.auto_accepted, ResolutionStatus.RESOLVED)]
    report = build_mapping_report(GoldSet(gold), decisions)
    assert report.has_labelled_contradiction() is True


# --- acceptance thresholds + coverage gate ----------------------------------


def test_full_coverage_above_thresholds_is_accepted() -> None:
    gold, decisions = _all_correct([(f"C{i}", i) for i in range(1, 11)])
    report = build_mapping_report(GoldSet(gold), decisions)
    assert report.coverage_fraction == pytest.approx(1.0)
    assert report.score.distinct_code.mapped_precision == pytest.approx(1.0)
    assert report.score.distinct_code.auto_resolution_rate == pytest.approx(1.0)
    assert report.verdict is AcceptanceVerdict.ACCEPTED


def test_subset_coverage_is_provisional_even_with_perfect_metrics() -> None:
    gold = [
        _gold("A", 10, GoldLabel.RESOLVED),
        GoldRow("U", None, GoldLabel.UNLABELLED, 3, "awaiting human"),
    ]
    decisions = [
        _decision("A", 10, Status.auto_accepted, ResolutionStatus.RESOLVED),
    ]
    report = build_mapping_report(GoldSet(gold), decisions)
    assert report.coverage_fraction == pytest.approx(0.5)
    assert report.verdict is AcceptanceVerdict.PROVISIONAL_NOT_ACCEPTED
    assert "U" in report.unlabelled_codes
    assert "coverage" in report.verdict_reason.lower()


def test_full_coverage_below_precision_is_running() -> None:
    gold = [
        _gold("A", 10, GoldLabel.RESOLVED),
        _gold("B", 20, GoldLabel.RESOLVED),
    ]
    decisions = [
        _decision("A", 10, Status.auto_accepted, ResolutionStatus.RESOLVED),
        _decision("B", 99, Status.auto_accepted, ResolutionStatus.RESOLVED),  # wrong
    ]
    report = build_mapping_report(GoldSet(gold), decisions)
    assert report.score.distinct_code.mapped_precision == pytest.approx(0.5)
    assert report.verdict is AcceptanceVerdict.RUNNING_NOT_ACCEPTED


def test_full_coverage_below_auto_resolution_is_running() -> None:
    gold = [_gold(f"C{i}", i, GoldLabel.RESOLVED) for i in range(1, 11)]
    decisions = []
    # 6 correct auto-accepts, 4 Zone-2 candidate (correct concept) -> auto = 0.6
    for i in range(1, 7):
        decisions.append(
            _decision(f"C{i}", i, Status.auto_accepted, ResolutionStatus.RESOLVED)
        )
    for i in range(7, 11):
        decisions.append(
            _decision(f"C{i}", i, Status.candidate, ResolutionStatus.RESOLVED)
        )
    report = build_mapping_report(GoldSet(gold), decisions)
    assert report.score.distinct_code.auto_resolution_rate == pytest.approx(0.6)
    assert report.verdict is AcceptanceVerdict.RUNNING_NOT_ACCEPTED


def test_full_coverage_undefined_precision_is_running() -> None:
    """All gold RESOLVED but every prediction NO_MAP -> precision denom 0."""
    gold = [_gold("A", 10, GoldLabel.RESOLVED)]
    decisions = [_decision("A", None, Status.auto_accepted, ResolutionStatus.NO_MAP)]
    report = build_mapping_report(GoldSet(gold), decisions)
    assert report.score.distinct_code.mapped_precision is None
    assert report.verdict is AcceptanceVerdict.RUNNING_NOT_ACCEPTED


def test_evaluate_acceptance_thresholds_are_the_documented_values() -> None:
    assert MIN_MAPPED_PRECISION == pytest.approx(0.95)
    assert MIN_AUTO_RESOLUTION_RATE == pytest.approx(0.70)


def test_evaluate_acceptance_boundary_exactly_at_thresholds_accepts() -> None:
    gold = [_gold(f"C{i}", i, GoldLabel.RESOLVED) for i in range(1, 21)]
    decisions = []
    # 19 correct + 1 wrong -> precision 0.95 exactly; all Zone-1 -> auto 1.0
    for i in range(1, 20):
        decisions.append(
            _decision(f"C{i}", i, Status.auto_accepted, ResolutionStatus.RESOLVED)
        )
    decisions.append(
        _decision("C20", 999, Status.auto_accepted, ResolutionStatus.RESOLVED)
    )
    report = build_mapping_report(GoldSet(gold), decisions)
    assert report.score.distinct_code.mapped_precision == pytest.approx(0.95)
    assert report.verdict is AcceptanceVerdict.ACCEPTED


# --- serialization + human summary ------------------------------------------


def test_as_dict_is_json_serializable_and_carries_caveat() -> None:
    gold, decisions = _all_correct([("A", 10)])
    report = build_mapping_report(GoldSet(gold), decisions)
    payload = report.as_dict()
    text = json.dumps(payload)
    assert "structural_precision_caveat" in text
    assert payload["structural_precision_caveat"] == STRUCTURAL_PRECISION_CAVEAT
    assert payload["thresholds"]["mapped_precision"] == pytest.approx(0.95)
    assert payload["metrics"]["distinct_code"]["mapped_precision"] == 1.0


def test_human_summary_states_verdict_and_structural_caveat() -> None:
    gold, decisions = _all_correct([("A", 10)])
    summary = build_mapping_report(GoldSet(gold), decisions).human_summary()
    assert "STRUCTURAL" in summary
    assert "accepted" in summary.lower()
    assert "no_map_accuracy" in summary


def test_human_summary_handles_none_metrics() -> None:
    gold = [_gold("A", 10, GoldLabel.RESOLVED)]
    decisions = [_decision("A", None, Status.auto_accepted, ResolutionStatus.NO_MAP)]
    summary = build_mapping_report(GoldSet(gold), decisions).human_summary()
    assert "n/a" in summary  # undefined precision rendered, not crashing


# --- value-mapping-store reading (US-005 → §1.5(f)) -------------------------


def test_decision_from_value_mapping_resolved_and_no_map() -> None:
    resolved = decision_from_value_mapping(
        _value_mapping("LUAD", 45768916, StoreResolutionStatus.RESOLVED)
    )
    assert resolved.source_code == "LUAD"
    assert resolved.concept_id == 45768916
    assert resolved.resolution_status is ResolutionStatus.RESOLVED

    no_map = decision_from_value_mapping(
        _value_mapping("ZZZ", None, StoreResolutionStatus.NO_MAP)
    )
    assert no_map.concept_id is None
    assert no_map.resolution_status is ResolutionStatus.NO_MAP
    assert no_map.no_map_reason == "dead end"


def test_decisions_from_store_reads_value_mapping_store(tmp_path: Path) -> None:
    conn = duckdb.connect(str(tmp_path / "vm.duckdb"))
    store = ValueMappingStore(conn)
    store.upsert(
        [
            _value_mapping("LUAD", 45768916, StoreResolutionStatus.RESOLVED),
            _value_mapping("ZZZ", None, StoreResolutionStatus.NO_MAP),
        ]
    )
    decisions = decisions_from_store(store)
    by_code = {d.source_code: d for d in decisions}
    assert by_code["LUAD"].concept_id == 45768916
    assert by_code["ZZZ"].resolution_status is ResolutionStatus.NO_MAP

    # Both gold codes RESOLVED + correctly auto-accepted -> precision 1.0, auto 1.0.
    decisions = decisions_from_store(store)
    luad = next(d for d in decisions if d.source_code == "LUAD")
    gold = GoldSet([_gold("LUAD", 45768916, GoldLabel.RESOLVED)])
    report = build_mapping_report(gold, [luad])
    assert report.verdict is AcceptanceVerdict.ACCEPTED
    store.close()


def test_decisions_from_store_scope_filter(tmp_path: Path) -> None:
    conn = duckdb.connect(str(tmp_path / "vm.duckdb"))
    store = ValueMappingStore(conn)
    store.upsert(
        [
            _value_mapping("LUAD", 1, StoreResolutionStatus.RESOLVED),
            _value_mapping(
                "LUAD",
                2,
                StoreResolutionStatus.RESOLVED,
                target_property_ref="target.other.some_concept_id",
            ),
        ]
    )
    scoped = decisions_from_store(
        store, target_property_ref="target.other.some_concept_id"
    )
    assert [d.concept_id for d in scoped] == [2]
    store.close()


def test_decisions_from_store_all_scope_filters(tmp_path: Path) -> None:
    conn = duckdb.connect(str(tmp_path / "vm.duckdb"))
    store = ValueMappingStore(conn)
    store.upsert([_value_mapping("LUAD", 1, StoreResolutionStatus.RESOLVED)])
    assert decisions_from_store(store, source_vocabulary="Other") == []
    assert (
        decisions_from_store(store, resolver_policy_ref="other.policy") == []
    )
    assert decisions_from_store(store, vocab_release="vocab-1999") == []
    kept = decisions_from_store(
        store,
        source_vocabulary="OncoTree",
        resolver_policy_ref=OMOP_ONCOTREE_CONDITION_REF,
        vocab_release="vocab-2024",
    )
    assert [d.concept_id for d in kept] == [1]
    store.close()


def test_report_from_store_reads_gold_file(tmp_path: Path) -> None:
    from sema.eval.mapping_report import report_from_store

    conn = duckdb.connect(str(tmp_path / "vm.duckdb"))
    store = ValueMappingStore(conn)
    store.upsert([_value_mapping("LUAD", 45768916, StoreResolutionStatus.RESOLVED)])
    gold_path = tmp_path / "gold.jsonl"
    gold_path.write_text(
        json.dumps(
            {
                "oncotree_code": "LUAD",
                "gold_concept_id": 45768916,
                "gold_label": "RESOLVED",
                "row_count": 7,
                "notes": "",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    report = report_from_store(store, gold_path)
    assert report.verdict is AcceptanceVerdict.ACCEPTED
    assert report.score.distinct_code.tp == 1
    store.close()


def test_evaluate_acceptance_provisional_surfaces_gap() -> None:
    gold = GoldSet(
        [
            _gold("A", 10, GoldLabel.RESOLVED),
            GoldRow("U", None, GoldLabel.UNLABELLED, 1, ""),
        ]
    )
    verdict, reason = evaluate_acceptance(None, gold.coverage_fraction())
    assert verdict is AcceptanceVerdict.PROVISIONAL_NOT_ACCEPTED
    assert "100%" in reason


def test_mapping_report_is_frozen_dataclass() -> None:
    gold, decisions = _all_correct([("A", 10)])
    report = build_mapping_report(GoldSet(gold), decisions)
    assert isinstance(report, MappingReport)
    with pytest.raises(Exception):
        report.verdict = AcceptanceVerdict.ACCEPTED  # type: ignore[misc]

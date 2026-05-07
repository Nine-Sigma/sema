"""Tests for the resolution-planner capability."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

pytestmark = pytest.mark.unit


def _run_prov() -> object:
    from sema.models.planner.provenance import RunProvenance

    return RunProvenance(
        run_id="run-1",
        target_model_version="omop-cdm-5.4",
        target_schema_snapshot_hash="t",
        vocab_release="v",
        context_card_version="c",
        prompt_template_version="t1",
        few_shot_set_version="f",
        constraint_version="cv",
        llm_model="m",
        embedding_model="e",
    )


def _src(source_id: str = "cbioportal_gbm") -> object:
    from sema.models.planner.provenance import SourceScope

    return SourceScope(
        source_id=source_id,
        source_schema_hash=f"s-{source_id}",
        source_profile_hash=f"p-{source_id}",
    )


def test_resolution_strategy_values() -> None:
    from sema.models.planner.resolution import ResolutionStrategy

    assert {s.value for s in ResolutionStrategy} == {
        "DETERMINISTIC_HASH",
        "FUZZY_BLOCK_AND_SCORE",
        "GRAPH_CLOSURE",
        "MULTI_KEY_UNION",
    }


def test_cycle_handling_values() -> None:
    from sema.models.planner.resolution import CycleHandling

    assert {c.value for c in CycleHandling} == {
        "REJECT",
        "BREAK_AT_DEPTH",
        "MARK_AND_CONTINUE",
    }


def test_resolution_verdict_values() -> None:
    from sema.models.planner.resolution import ResolutionVerdict

    assert {v.value for v in ResolutionVerdict} == {
        "resolved",
        "ambiguous",
        "unresolved",
        "awaiting_review",
    }


def test_deterministic_hash_payload() -> None:
    from sema.models.planner.resolution import (
        DeterministicHashPayload,
    )

    p = DeterministicHashPayload(
        source_key_refs=["cbio.study_id", "cbio.patient.patient_id"],
    )
    assert len(p.source_key_refs) == 2


def test_fuzzy_payload_requires_features() -> None:
    from sema.models.planner.resolution import FuzzyBlockAndScorePayload

    with pytest.raises(ValidationError):
        FuzzyBlockAndScorePayload(blocking_keys=[], similarity_features=[])
    with pytest.raises(ValidationError):
        FuzzyBlockAndScorePayload(
            blocking_keys=["addr.norm_name_prefix"],
            similarity_features=[],
        )


def test_multi_key_union_requires_two() -> None:
    from sema.models.planner.resolution import MultiKeyUnionPayload

    with pytest.raises(ValidationError):
        MultiKeyUnionPayload(source_key_refs=["acris.bbl"])

    p = MultiKeyUnionPayload(
        source_key_refs=["acris.bbl", "acris.address", "dof.parcel_id"]
    )
    assert len(p.source_key_refs) == 3


def test_resolution_plan_graph_closure_requires_cycle_handling() -> None:
    from sema.models.planner.resolution import (
        GraphClosurePayload,
        ResolutionPlan,
        ResolutionStrategy,
    )

    with pytest.raises(ValidationError):
        ResolutionPlan(
            id="r-1",
            sources=[_src()],
            target_identity_ref="canonical.llc_id",
            strategy=ResolutionStrategy.GRAPH_CLOSURE,
            payload=GraphClosurePayload(walk_relationship="OWNS"),
            transitive_closure=False,
            confidence=0.9,
            provenance_run=_run_prov(),
            timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )


def test_resolution_plan_transitive_requires_cycle_handling() -> None:
    from sema.models.planner.resolution import (
        DeterministicHashPayload,
        ResolutionPlan,
        ResolutionStrategy,
    )

    with pytest.raises(ValidationError):
        ResolutionPlan(
            id="r-2",
            sources=[_src()],
            target_identity_ref="canonical.id",
            strategy=ResolutionStrategy.DETERMINISTIC_HASH,
            payload=DeterministicHashPayload(source_key_refs=["cbio.x"]),
            transitive_closure=True,
            confidence=1.0,
            provenance_run=_run_prov(),
            timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )


def test_deterministic_hash_implies_confidence_one() -> None:
    from sema.models.planner.resolution import (
        DeterministicHashPayload,
        ResolutionPlan,
        ResolutionStrategy,
    )

    with pytest.raises(ValidationError):
        ResolutionPlan(
            id="r-3",
            sources=[_src()],
            target_identity_ref="canonical.id",
            strategy=ResolutionStrategy.DETERMINISTIC_HASH,
            payload=DeterministicHashPayload(source_key_refs=["cbio.x"]),
            confidence=0.9,
            provenance_run=_run_prov(),
            timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )


def test_multi_source_resolution_records_each_scope() -> None:
    from sema.models.planner.resolution import (
        MultiKeyUnionPayload,
        ResolutionPlan,
        ResolutionStrategy,
    )

    plan = ResolutionPlan(
        id="r-4",
        sources=[_src("acris.deeds"), _src("dof.parcels")],
        target_identity_ref="canonical.property_id",
        strategy=ResolutionStrategy.MULTI_KEY_UNION,
        payload=MultiKeyUnionPayload(
            source_key_refs=["acris.bbl", "dof.parcel_id"]
        ),
        confidence=0.85,
        provenance_run=_run_prov(),
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    assert {s.source_id for s in plan.sources} == {"acris.deeds", "dof.parcels"}


def test_resolution_verdict_resolved_for_clean() -> None:
    from sema.models.planner.resolution import (
        ResolutionVerdict,
        derive_resolution_verdict,
    )

    v = derive_resolution_verdict(
        produced_for_every_input=True,
        ambiguous_assignments=False,
        cycle_blocked=False,
        any_block_flag=False,
        plan_review_pending=False,
    )
    assert v == ResolutionVerdict.resolved


def test_resolution_verdict_ambiguous_on_fuzzy_tie() -> None:
    from sema.models.planner.resolution import (
        ResolutionVerdict,
        derive_resolution_verdict,
    )

    v = derive_resolution_verdict(
        produced_for_every_input=True,
        ambiguous_assignments=True,
        cycle_blocked=False,
        any_block_flag=False,
        plan_review_pending=False,
    )
    assert v == ResolutionVerdict.ambiguous


def test_resolution_verdict_unresolved_on_cycle_block() -> None:
    from sema.models.planner.resolution import (
        ResolutionVerdict,
        derive_resolution_verdict,
    )

    v = derive_resolution_verdict(
        produced_for_every_input=False,
        ambiguous_assignments=False,
        cycle_blocked=True,
        any_block_flag=True,
        plan_review_pending=False,
    )
    assert v == ResolutionVerdict.unresolved


def test_resolution_verdict_awaiting_review() -> None:
    from sema.models.planner.resolution import (
        ResolutionVerdict,
        derive_resolution_verdict,
    )

    v = derive_resolution_verdict(
        produced_for_every_input=True,
        ambiguous_assignments=False,
        cycle_blocked=False,
        any_block_flag=True,
        plan_review_pending=True,
    )
    assert v == ResolutionVerdict.awaiting_review


def test_resolution_dependency_round_trip() -> None:
    from sema.models.planner.resolution import ResolutionDependency

    rd = ResolutionDependency(
        upstream_plan_id="r-4",
        canonical_identity_column="canonical.property_id",
    )
    rt = ResolutionDependency.model_validate(rd.model_dump(mode="json"))
    assert rt.upstream_plan_id == "r-4"

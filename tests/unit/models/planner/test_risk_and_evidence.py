"""Tests for the risk-and-evidence capability."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

pytestmark = pytest.mark.unit


def test_severity_values() -> None:
    from sema.models.planner.risk import Severity

    assert {s.value for s in Severity} == {"info", "warn", "block"}


def test_source_stage_values() -> None:
    from sema.models.planner.risk import SourceStage

    assert {s.value for s in SourceStage} == {
        "candidate_gen",
        "producer",
        "constraint",
        "verify",
        "transform",
    }


def test_suggested_action_values() -> None:
    from sema.models.planner.risk import SuggestedAction

    assert {a.value for a in SuggestedAction} == {
        "review",
        "request_more_samples",
        "reject",
        "ignore_with_reason",
    }


def test_evidence_mode_values() -> None:
    from sema.models.planner.risk import EvidenceMode

    assert {m.value for m in EvidenceMode} == {
        "RAW",
        "CATEGORICAL",
        "HASH",
        "COUNT_ONLY",
        "EXCERPT",
    }


def test_sensitivity_class_values() -> None:
    from sema.models.planner.risk import SensitivityClass

    assert "PUBLIC" in {c.value for c in SensitivityClass}
    assert "PHI" in {c.value for c in SensitivityClass}
    assert "PII" in {c.value for c in SensitivityClass}


def test_default_evidence_mode_for_phi() -> None:
    from sema.models.planner.risk import (
        SensitivityClass,
        default_evidence_mode,
        EvidenceMode,
    )

    assert default_evidence_mode(SensitivityClass.PHI) == EvidenceMode.CATEGORICAL
    assert default_evidence_mode(SensitivityClass.PII) == EvidenceMode.HASH
    assert default_evidence_mode(SensitivityClass.PUBLIC) == EvidenceMode.RAW


def test_evidence_count_only_rejects_literal() -> None:
    from sema.models.planner.risk import (
        Evidence,
        EvidenceMode,
        SensitivityClass,
    )

    with pytest.raises(ValidationError):
        Evidence(
            mode=EvidenceMode.COUNT_ONLY,
            payload={"value": "literal"},
            sensitivity_class=SensitivityClass.PUBLIC,
            source_ref="cbio.patient.gender",
        )


def test_evidence_count_only_with_count_payload() -> None:
    from sema.models.planner.risk import (
        Evidence,
        EvidenceMode,
        SensitivityClass,
    )

    e = Evidence(
        mode=EvidenceMode.COUNT_ONLY,
        payload={"count": 42},
        sensitivity_class=SensitivityClass.PHI,
        source_ref="cbio.patient.gender",
    )
    assert e.mode == EvidenceMode.COUNT_ONLY


def test_evidence_raw_against_phi_requires_override() -> None:
    from sema.models.planner.risk import (
        Evidence,
        EvidenceMode,
        SensitivityClass,
    )

    with pytest.raises(ValidationError):
        Evidence(
            mode=EvidenceMode.RAW,
            payload={"value": "Mr. Smith"},
            sensitivity_class=SensitivityClass.PHI,
            source_ref="cbio.patient.name",
        )

    e = Evidence(
        mode=EvidenceMode.RAW,
        payload={"value": "Mr. Smith"},
        sensitivity_class=SensitivityClass.PHI,
        source_ref="cbio.patient.name",
        explicit_raw_override=True,
    )
    assert e.explicit_raw_override is True


def test_risk_code_registered() -> None:
    from sema.models.planner.risk import RiskCode

    expected = {
        "RISK_VOCAB_DOMAIN_MISMATCH",
        "RISK_PIVOT_CARDINALITY_UNVERIFIED",
        "RISK_TEMPORAL_LOST",
        "RISK_AMBIGUOUS_TARGET",
        "RISK_OBLIGATION_REQUIRED_FIELD_MISSING",
        "RISK_OBLIGATION_FK_UNSATISFIED",
        "RISK_OBLIGATION_MINIMUM_VIABLE_ROW_VIOLATED",
        "RISK_DEFAULT_APPLIED",
        "RISK_RESOLUTION_DEPENDENCY_MISSING",
        "RISK_LLC_CYCLE_DETECTED",
        "RISK_ASSEMBLER_CONFLICT_RESOLVED",
    }
    actual = {c.value for c in RiskCode}
    assert expected.issubset(actual)


def test_risk_flag_construction() -> None:
    from sema.models.planner.risk import (
        Evidence,
        EvidenceMode,
        RiskCode,
        RiskFlag,
        SensitivityClass,
        Severity,
        SourceStage,
        SuggestedAction,
    )

    rf = RiskFlag(
        code=RiskCode.RISK_VOCAB_DOMAIN_MISMATCH,
        severity=Severity.warn,
        evidence=[
            Evidence(
                mode=EvidenceMode.CATEGORICAL,
                payload={"distinct": 3, "pattern": "[A-Z]+"},
                sensitivity_class=SensitivityClass.PHI,
                source_ref="cbio.patient.gender",
            )
        ],
        source_stage=SourceStage.producer,
        suggested_action=SuggestedAction.review,
    )
    assert rf.code == RiskCode.RISK_VOCAB_DOMAIN_MISMATCH
    assert len(rf.evidence) == 1


def test_risk_flag_evidence_must_be_list() -> None:
    from sema.models.planner.risk import (
        RiskCode,
        RiskFlag,
        Severity,
        SourceStage,
        SuggestedAction,
    )

    with pytest.raises(ValidationError):
        RiskFlag(
            code=RiskCode.RISK_AMBIGUOUS_TARGET,
            severity=Severity.warn,
            evidence="bare string",
            source_stage=SourceStage.producer,
            suggested_action=SuggestedAction.review,
        )


def test_risk_flag_round_trip() -> None:
    from sema.models.planner.risk import (
        Evidence,
        EvidenceMode,
        RiskCode,
        RiskFlag,
        SensitivityClass,
        Severity,
        SourceStage,
        SuggestedAction,
    )

    rf = RiskFlag(
        code=RiskCode.RISK_AMBIGUOUS_TARGET,
        severity=Severity.warn,
        evidence=[
            Evidence(
                mode=EvidenceMode.COUNT_ONLY,
                payload={"count": 7},
                sensitivity_class=SensitivityClass.PHI,
                source_ref="x",
            ),
            Evidence(
                mode=EvidenceMode.CATEGORICAL,
                payload={"shape": "alpha"},
                sensitivity_class=SensitivityClass.PHI,
                source_ref="x",
            ),
        ],
        source_stage=SourceStage.constraint,
        suggested_action=SuggestedAction.review,
    )
    payload = rf.model_dump(mode="json")
    rt = RiskFlag.model_validate(payload)
    assert len(rt.evidence) == 2
    assert rt.evidence[0].mode == EvidenceMode.COUNT_ONLY
    assert rt.evidence[1].mode == EvidenceMode.CATEGORICAL

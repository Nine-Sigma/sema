"""risk-and-evidence capability: structured RiskFlag + typed Evidence."""

from __future__ import annotations

from enum import Enum
from typing import Any, Self

from pydantic import BaseModel, Field, model_validator


class Severity(str, Enum):
    info = "info"
    warn = "warn"
    block = "block"


class SourceStage(str, Enum):
    candidate_gen = "candidate_gen"
    producer = "producer"
    constraint = "constraint"
    verify = "verify"
    transform = "transform"


class SuggestedAction(str, Enum):
    review = "review"
    request_more_samples = "request_more_samples"
    reject = "reject"
    ignore_with_reason = "ignore_with_reason"


class EvidenceMode(str, Enum):
    RAW = "RAW"
    CATEGORICAL = "CATEGORICAL"
    HASH = "HASH"
    COUNT_ONLY = "COUNT_ONLY"
    EXCERPT = "EXCERPT"


class SensitivityClass(str, Enum):
    PUBLIC = "PUBLIC"
    PII = "PII"
    PHI = "PHI"
    FINANCIAL_RESTRICTED = "FINANCIAL_RESTRICTED"
    CONFIDENTIAL = "CONFIDENTIAL"


_DEFAULT_MODE: dict[SensitivityClass, EvidenceMode] = {
    SensitivityClass.PUBLIC: EvidenceMode.RAW,
    SensitivityClass.PII: EvidenceMode.HASH,
    SensitivityClass.PHI: EvidenceMode.CATEGORICAL,
    SensitivityClass.FINANCIAL_RESTRICTED: EvidenceMode.HASH,
    SensitivityClass.CONFIDENTIAL: EvidenceMode.CATEGORICAL,
}


def default_evidence_mode(sensitivity: SensitivityClass) -> EvidenceMode:
    return _DEFAULT_MODE[sensitivity]


class RiskCode(str, Enum):
    RISK_VOCAB_DOMAIN_MISMATCH = "RISK_VOCAB_DOMAIN_MISMATCH"
    RISK_PIVOT_CARDINALITY_UNVERIFIED = "RISK_PIVOT_CARDINALITY_UNVERIFIED"
    RISK_TEMPORAL_LOST = "RISK_TEMPORAL_LOST"
    RISK_AMBIGUOUS_TARGET = "RISK_AMBIGUOUS_TARGET"
    RISK_OBLIGATION_REQUIRED_FIELD_MISSING = "RISK_OBLIGATION_REQUIRED_FIELD_MISSING"
    RISK_OBLIGATION_FK_UNSATISFIED = "RISK_OBLIGATION_FK_UNSATISFIED"
    RISK_OBLIGATION_MINIMUM_VIABLE_ROW_VIOLATED = (
        "RISK_OBLIGATION_MINIMUM_VIABLE_ROW_VIOLATED"
    )
    RISK_DEFAULT_APPLIED = "RISK_DEFAULT_APPLIED"
    RISK_RESOLUTION_DEPENDENCY_MISSING = "RISK_RESOLUTION_DEPENDENCY_MISSING"
    RISK_LLC_CYCLE_DETECTED = "RISK_LLC_CYCLE_DETECTED"
    RISK_ASSEMBLER_CONFLICT_RESOLVED = "RISK_ASSEMBLER_CONFLICT_RESOLVED"


class Evidence(BaseModel):
    mode: EvidenceMode | None = None
    payload: dict[str, Any]
    sensitivity_class: SensitivityClass
    source_ref: str = Field(min_length=1)
    explicit_raw_override: bool = False

    @model_validator(mode="after")
    def _validate_mode_payload(self) -> Self:
        if self.mode is None:
            object.__setattr__(
                self, "mode", default_evidence_mode(self.sensitivity_class)
            )
        assert self.mode is not None
        _validate_mode_payload(self.mode, self.payload)
        _validate_phi_raw(self.mode, self.sensitivity_class, self.explicit_raw_override)
        return self


def _validate_mode_payload(mode: EvidenceMode, payload: dict[str, Any]) -> None:
    if mode is EvidenceMode.COUNT_ONLY:
        keys = set(payload.keys())
        if not keys.issubset({"count", "distinct"}):
            raise ValueError("COUNT_ONLY rejects literal-value payloads")
    if mode is EvidenceMode.HASH and "hash" not in payload and "digest" not in payload:
        raise ValueError("HASH mode requires a hash/digest payload field")


def _validate_phi_raw(
    mode: EvidenceMode, sens: SensitivityClass, override: bool
) -> None:
    if mode is EvidenceMode.RAW and sens is SensitivityClass.PHI and not override:
        raise ValueError(
            "RAW mode against PHI requires explicit_raw_override=True"
        )


class RiskFlag(BaseModel):
    code: RiskCode
    severity: Severity
    evidence: list[Evidence] = Field(default_factory=list)
    source_stage: SourceStage
    suggested_action: SuggestedAction

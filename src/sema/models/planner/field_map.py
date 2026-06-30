"""mapping-planner: FieldMap and RowIdentity."""

from __future__ import annotations

import hashlib
from typing import Any, ClassVar, Self

from pydantic import BaseModel, Field, model_validator

from sema.models.planner._enums import MaterializationMode
from sema.models.planner._refs import RefStr
from sema.models.planner.lifecycle import Status
from sema.models.planner.patterns import (
    ConstantValue,
    MappingPattern,
    PatternPayload,
    expected_payload_type,
)


def coerce_pattern_payload(data: Any) -> Any:
    if not isinstance(data, dict):
        return data
    pattern = data.get("pattern")
    payload = data.get("payload")
    if isinstance(payload, dict) and isinstance(pattern, str):
        target = expected_payload_type(MappingPattern(pattern))
        data["payload"] = target.model_validate(payload)
    return data


class FieldMap(BaseModel):
    target_field_ref: RefStr
    pattern: MappingPattern
    payload: PatternPayload
    status: Status = Status.candidate

    @model_validator(mode="before")
    @classmethod
    def _coerce_payload(cls, data: Any) -> Any:
        return coerce_pattern_payload(data)

    @model_validator(mode="after")
    def _validate_payload_matches_pattern(self) -> Self:
        expected = expected_payload_type(self.pattern)
        if not isinstance(self.payload, expected):
            raise ValueError(
                f"pattern {self.pattern.value} requires payload of {expected.__name__}"
            )
        return self

    def covers_required_field(self) -> bool:
        """§1.5(e): only an accepted, value-producing FieldMap covers a field.

        A rejected status, a ``NO_MAP`` pattern, or a ``CONSTANT`` with a NULL
        literal never satisfies a required obligation field.
        """
        if self.status is Status.rejected:
            return False
        if self.pattern is MappingPattern.NO_MAP:
            return False
        if (
            self.pattern is MappingPattern.CONSTANT
            and isinstance(self.payload, ConstantValue)
            and self.payload.literal_value is None
        ):
            return False
        return True


class RowIdentity(BaseModel):
    target_row_key_rule: str = Field(min_length=1)
    source_lineage: list[RefStr] = Field(min_length=1)
    materialization_mode: MaterializationMode


def derive_row_key(identity: RowIdentity, source_values: dict[str, Any]) -> str:
    parts = [identity.target_row_key_rule]
    for ref in identity.source_lineage:
        parts.append(f"{ref}={source_values.get(ref, '')}")
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return digest

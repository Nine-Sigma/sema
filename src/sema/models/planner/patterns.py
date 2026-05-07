"""mapping-planner: 11 patterns + per-pattern payload schemas."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from sema.models.planner._refs import RefStr


class _StrictPayload(BaseModel):
    """Base for per-pattern payloads; rejects unknown fields per spec 3.2."""

    model_config = ConfigDict(extra="forbid")


class MappingPattern(str, Enum):
    DIRECT_COPY = "DIRECT_COPY"
    CONSTANT = "CONSTANT"
    DERIVED = "DERIVED"
    VOCAB_LOOKUP = "VOCAB_LOOKUP"
    JOIN_LOOKUP = "JOIN_LOOKUP"
    PIVOT = "PIVOT"
    UNPIVOT = "UNPIVOT"
    SPLIT = "SPLIT"
    AGGREGATE = "AGGREGATE"
    ROW_GENERATION = "ROW_GENERATION"
    NO_MAP = "NO_MAP"


class NoMapScope(str, Enum):
    GLOBAL = "GLOBAL"
    TARGET_ENTITY = "TARGET_ENTITY"
    TARGET_PROPERTY = "TARGET_PROPERTY"


class AggregateFunction(str, Enum):
    COUNT = "COUNT"
    SUM = "SUM"
    AVG = "AVG"
    MIN = "MIN"
    MAX = "MAX"
    FIRST = "FIRST"
    LAST = "LAST"
    ANY = "ANY"
    ALL = "ALL"
    ARRAY_AGG = "ARRAY_AGG"
    JSON_AGG = "JSON_AGG"


class CardinalityAssumption(str, Enum):
    one_to_one = "one_to_one"
    many_to_one = "many_to_one"


class PivotExpansionMode(str, Enum):
    one_row_per_key = "one_row_per_key"
    multi_column = "multi_column"


class DirectCopyPayload(_StrictPayload):
    source_field_ref: RefStr


class ConstantValue(_StrictPayload):
    literal_value: Any
    target_type: str = Field(min_length=1)


class DerivedExpression(_StrictPayload):
    source_field_refs: list[RefStr] = Field(min_length=1)
    expression_ast: dict[str, Any]
    nullability_rule: str | None = None


class VocabLookup(_StrictPayload):
    vocabulary_ref: RefStr
    source_value_ref: RefStr
    domain_constraint_ref: RefStr
    require_standard: bool
    allow_zero_default: bool
    resolver_policy_ref: RefStr
    effective_date_ref: RefStr | None = None


class JoinKeyPair(_StrictPayload):
    from_field_ref: RefStr
    to_field_ref: RefStr


class JoinLookup(_StrictPayload):
    from_source_ref: RefStr
    to_source_ref: RefStr
    join_keys: list[JoinKeyPair] = Field(min_length=1)
    select_field_ref: RefStr
    cardinality_assumption: CardinalityAssumption | None = None
    resolution_dependency_ref: RefStr | None = None


class PivotMapping(_StrictPayload):
    source_table_ref: RefStr
    key_field_ref: RefStr
    value_field_ref: RefStr
    partition_keys: list[RefStr] = Field(min_length=1)
    expansion_mode: PivotExpansionMode
    max_keys: int | None = Field(default=None, gt=0)


class UnpivotMapping(_StrictPayload):
    source_table_ref: RefStr
    key_columns: list[RefStr] = Field(min_length=1)
    key_name_target_field: RefStr
    value_target_field: RefStr
    null_skip: bool = False


class SplitRuleKind(str, Enum):
    regex = "regex"
    delimiter = "delimiter"


class SplitRule(_StrictPayload):
    kind: SplitRuleKind
    pattern: str | None = None
    delimiter: str | None = None
    positions: list[str] | None = None


class SplitMapping(_StrictPayload):
    source_field_ref: RefStr
    split_rule: SplitRule
    output_target_fields: dict[str, RefStr] = Field(min_length=1)


class AggregateOp(_StrictPayload):
    target_field_ref: RefStr
    aggregate_function: AggregateFunction
    source_field_ref: RefStr


class AggregateMapping(_StrictPayload):
    source_table_ref: RefStr
    group_by_keys: list[RefStr] = Field(min_length=1)
    aggregations: list[AggregateOp] = Field(min_length=1)
    filter_predicate: dict[str, Any] | None = None
    having_predicate: dict[str, Any] | None = None


class GenerationRule(_StrictPayload):
    kind: Literal["distinct_keys", "window_envelope"]
    keys: list[RefStr] | None = None
    partition: list[RefStr] | None = None
    min_field: RefStr | None = None
    max_field: RefStr | None = None


class RowGenerationMapping(_StrictPayload):
    source_scope_ref: RefStr
    generation_rule: GenerationRule
    populated_field_maps: list[Any] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_field_maps(self) -> Self:
        from sema.models.planner.field_map import FieldMap

        coerced: list[Any] = []
        for i, fm in enumerate(self.populated_field_maps):
            if isinstance(fm, FieldMap):
                coerced.append(fm)
                continue
            if isinstance(fm, dict):
                coerced.append(FieldMap.model_validate(fm))
                continue
            raise ValueError(
                f"populated_field_maps[{i}] must be a FieldMap (got {type(fm).__name__})"
            )
        object.__setattr__(self, "populated_field_maps", coerced)
        return self


class NoMapPayload(_StrictPayload):
    reason: str = Field(min_length=1)
    scope: NoMapScope
    target_entity_ref: RefStr | None = None
    target_property_ref: RefStr | None = None

    @model_validator(mode="after")
    def _validate_scope(self) -> Self:
        if self.scope is NoMapScope.TARGET_PROPERTY and not self.target_property_ref:
            raise ValueError("scope=TARGET_PROPERTY MUST identify a target Property")
        if self.scope is NoMapScope.TARGET_ENTITY and not self.target_entity_ref:
            raise ValueError("scope=TARGET_ENTITY MUST identify a target Entity")
        return self


PatternPayload = (
    DirectCopyPayload
    | ConstantValue
    | DerivedExpression
    | VocabLookup
    | JoinLookup
    | PivotMapping
    | UnpivotMapping
    | SplitMapping
    | AggregateMapping
    | RowGenerationMapping
    | NoMapPayload
)


_PATTERN_PAYLOAD_TYPES: dict[MappingPattern, type[BaseModel]] = {
    MappingPattern.DIRECT_COPY: DirectCopyPayload,
    MappingPattern.CONSTANT: ConstantValue,
    MappingPattern.DERIVED: DerivedExpression,
    MappingPattern.VOCAB_LOOKUP: VocabLookup,
    MappingPattern.JOIN_LOOKUP: JoinLookup,
    MappingPattern.PIVOT: PivotMapping,
    MappingPattern.UNPIVOT: UnpivotMapping,
    MappingPattern.SPLIT: SplitMapping,
    MappingPattern.AGGREGATE: AggregateMapping,
    MappingPattern.ROW_GENERATION: RowGenerationMapping,
    MappingPattern.NO_MAP: NoMapPayload,
}


def expected_payload_type(pattern: MappingPattern) -> type[BaseModel]:
    return _PATTERN_PAYLOAD_TYPES[pattern]

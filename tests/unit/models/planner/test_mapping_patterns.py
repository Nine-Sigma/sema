"""Tests for the mapping-planner pattern enum and per-pattern payloads."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

pytestmark = pytest.mark.unit


def test_mapping_pattern_has_eleven_values() -> None:
    from sema.models.planner.patterns import MappingPattern

    expected = {
        "DIRECT_COPY",
        "CONSTANT",
        "DERIVED",
        "VOCAB_LOOKUP",
        "JOIN_LOOKUP",
        "PIVOT",
        "UNPIVOT",
        "SPLIT",
        "AGGREGATE",
        "ROW_GENERATION",
        "NO_MAP",
    }
    assert {p.value for p in MappingPattern} == expected


def test_unknown_pattern_rejected() -> None:
    from sema.models.planner.patterns import MappingPattern

    with pytest.raises(ValueError):
        MappingPattern("FUZZY_LOOKUP")


def test_no_map_scope_values() -> None:
    from sema.models.planner.patterns import NoMapScope

    assert {s.value for s in NoMapScope} == {
        "GLOBAL",
        "TARGET_ENTITY",
        "TARGET_PROPERTY",
    }


def test_direct_copy_payload() -> None:
    from sema.models.planner.patterns import DirectCopyPayload

    p = DirectCopyPayload(source_field_ref="cbio.patient.gender")
    assert p.source_field_ref == "cbio.patient.gender"


def test_constant_payload_rejects_null_for_required() -> None:
    from sema.models.planner.patterns import ConstantValue

    p = ConstantValue(literal_value=42, target_type="int")
    assert p.literal_value == 42


def test_derived_expression_requires_inputs() -> None:
    from sema.models.planner.patterns import DerivedExpression

    with pytest.raises(ValidationError):
        DerivedExpression(source_field_refs=[], expression_ast={"op": "year"})


def test_vocab_lookup_requires_all_hooks() -> None:
    from sema.models.planner.patterns import VocabLookup

    with pytest.raises(ValidationError):
        VocabLookup(
            vocabulary_ref="omop.SNOMED",
            source_value_ref="x",
            domain_constraint_ref="d",
            require_standard=True,
            allow_zero_default=False,
        )

    p = VocabLookup(
        vocabulary_ref="omop.SNOMED",
        source_value_ref="cbio.cancer_type",
        domain_constraint_ref="omop.condition_concept_id.domain=Condition",
        require_standard=True,
        allow_zero_default=False,
        resolver_policy_ref="omop.snomed.condition.v1",
    )
    assert p.effective_date_ref is None


def test_join_lookup_requires_keys() -> None:
    from sema.models.planner.patterns import JoinKeyPair, JoinLookup

    with pytest.raises(ValidationError):
        JoinLookup(
            from_source_ref="cbio.a",
            to_source_ref="cbio.b",
            join_keys=[],
            select_field_ref="cbio.x",
        )

    p = JoinLookup(
        from_source_ref="cbio.a",
        to_source_ref="cbio.b",
        join_keys=[
            JoinKeyPair(from_field_ref="cbio.a.id", to_field_ref="cbio.b.id")
        ],
        select_field_ref="cbio.b.value",
    )
    assert len(p.join_keys) == 1


def test_pivot_partition_keys_required() -> None:
    from sema.models.planner.patterns import PivotMapping

    with pytest.raises(ValidationError):
        PivotMapping(
            source_table_ref="cbio.t",
            key_field_ref="cbio.t.k",
            value_field_ref="cbio.t.v",
            partition_keys=[],
            expansion_mode="multi_column",
        )


def test_unpivot_requires_columns() -> None:
    from sema.models.planner.patterns import UnpivotMapping

    with pytest.raises(ValidationError):
        UnpivotMapping(
            source_table_ref="cbio.t",
            key_columns=[],
            key_name_target_field="omop.t.name",
            value_target_field="omop.t.val",
        )


def test_split_outputs_required() -> None:
    from sema.models.planner.patterns import SplitMapping, SplitRule

    with pytest.raises(ValidationError):
        SplitMapping(
            source_field_ref="cbio.t.x",
            split_rule=SplitRule(kind="regex", pattern="(?<a>.*)"),
            output_target_fields={},
        )


def test_aggregate_function_closed_set() -> None:
    from sema.models.planner.patterns import (
        AggregateFunction,
        AggregateMapping,
        AggregateOp,
    )

    with pytest.raises(ValueError):
        AggregateFunction("MEDIAN")

    p = AggregateMapping(
        source_table_ref="cbio.t",
        group_by_keys=["cbio.t.patient_id"],
        aggregations=[
            AggregateOp(
                target_field_ref="omop.t.count_obs",
                aggregate_function=AggregateFunction.COUNT,
                source_field_ref="cbio.t.obs_id",
            )
        ],
    )
    assert p.aggregations[0].aggregate_function == AggregateFunction.COUNT


def test_row_generation_requires_field_maps() -> None:
    from sema.models.planner.patterns import RowGenerationMapping

    with pytest.raises(ValidationError):
        RowGenerationMapping(
            source_scope_ref="cbio.x",
            generation_rule={
                "kind": "distinct_keys",
                "keys": ["cbio.t.patient_id"],
            },
            populated_field_maps=[],
        )


def test_no_map_payload_scope_required() -> None:
    from sema.models.planner.patterns import NoMapPayload, NoMapScope

    p = NoMapPayload(reason="INTERNAL_BOOKKEEPING_FIELD", scope=NoMapScope.GLOBAL)
    assert p.scope == NoMapScope.GLOBAL


def test_no_map_target_property_requires_property_ref() -> None:
    from sema.models.planner.patterns import NoMapPayload, NoMapScope

    with pytest.raises(ValidationError):
        NoMapPayload(reason="x", scope=NoMapScope.TARGET_PROPERTY)


def test_no_map_target_entity_requires_entity_ref() -> None:
    from sema.models.planner.patterns import NoMapPayload, NoMapScope

    with pytest.raises(ValidationError):
        NoMapPayload(reason="x", scope=NoMapScope.TARGET_ENTITY)


def test_payload_polymorphism_no_kind_flag() -> None:
    from sema.models.planner.patterns import DirectCopyPayload, PivotMapping

    direct_fields = set(DirectCopyPayload.model_fields.keys())
    pivot_fields = set(PivotMapping.model_fields.keys())
    assert "kind" not in direct_fields
    assert "kind" not in pivot_fields
    assert "target_artifact_kind" not in direct_fields
    assert "target_artifact_kind" not in pivot_fields


def test_direct_copy_rejects_extra_target_artifact_kind() -> None:
    from sema.models.planner.patterns import DirectCopyPayload

    with pytest.raises(ValidationError):
        DirectCopyPayload(
            source_field_ref="cbio.x", target_artifact_kind="GRAPH_NODE"
        )


def test_pivot_rejects_unknown_field() -> None:
    from sema.models.planner.patterns import PivotExpansionMode, PivotMapping

    with pytest.raises(ValidationError):
        PivotMapping(
            source_table_ref="cbio.t",
            key_field_ref="cbio.t.k",
            value_field_ref="cbio.t.v",
            partition_keys=["cbio.t.p"],
            expansion_mode=PivotExpansionMode.multi_column,
            secret_flag=True,
        )


def test_row_generation_rejects_non_field_map_entries() -> None:
    from sema.models.planner.patterns import RowGenerationMapping

    with pytest.raises(ValidationError):
        RowGenerationMapping(
            source_scope_ref="cbio.events_per_patient",
            generation_rule={
                "kind": "window_envelope",
                "partition": ["cbio.events.patient_id"],
                "min_field": "cbio.events.event_date",
                "max_field": "cbio.events.event_date",
            },
            populated_field_maps=["not-a-field-map"],
        )


def test_row_generation_accepts_field_map_dicts() -> None:
    from sema.models.planner.patterns import (
        MappingPattern,
        RowGenerationMapping,
    )

    rg = RowGenerationMapping(
        source_scope_ref="cbio.events_per_patient",
        generation_rule={
            "kind": "window_envelope",
            "partition": ["cbio.events.patient_id"],
            "min_field": "cbio.events.event_date",
            "max_field": "cbio.events.event_date",
        },
        populated_field_maps=[
            {
                "target_field_ref": "omop.person.person_id",
                "pattern": "DIRECT_COPY",
                "payload": {"source_field_ref": "cbio.patient_id"},
            }
        ],
    )
    assert rg.populated_field_maps[0].pattern == MappingPattern.DIRECT_COPY


def test_row_generation_rule_must_be_typed() -> None:
    from sema.models.planner.patterns import RowGenerationMapping

    with pytest.raises(ValidationError):
        RowGenerationMapping(
            source_scope_ref="cbio.x",
            generation_rule={"kind": "unknown_kind"},
            populated_field_maps=[
                {
                    "target_field_ref": "omop.t.f",
                    "pattern": "DIRECT_COPY",
                    "payload": {"source_field_ref": "cbio.s"},
                }
            ],
        )


def test_bare_string_ref_rejected() -> None:
    from sema.models.planner.patterns import DirectCopyPayload

    with pytest.raises(ValidationError):
        DirectCopyPayload(source_field_ref="x")
    with pytest.raises(ValidationError):
        DirectCopyPayload(source_field_ref="bare_identifier")

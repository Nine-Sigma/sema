"""Tests for the target-model capability."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

pytestmark = pytest.mark.unit


def test_model_role_enum_values() -> None:
    from sema.models.planner._enums import ModelRole

    assert {r.value for r in ModelRole} == {"SOURCE", "TARGET"}


def test_target_artifact_kind_values() -> None:
    from sema.models.planner._enums import TargetArtifactKind

    assert {k.value for k in TargetArtifactKind} == {
        "TABLE_ROW",
        "GRAPH_NODE",
        "GRAPH_EDGE",
    }


def test_primary_key_strategy_values() -> None:
    from sema.models.planner._enums import PrimaryKeyStrategy

    assert {s.value for s in PrimaryKeyStrategy} == {
        "DETERMINISTIC_HASH",
        "EXTERNAL_SEQUENCE",
        "NATURAL_KEY",
        "COMPOUND",
    }


def test_materialization_mode_values() -> None:
    from sema.models.planner._enums import MaterializationMode

    assert {m.value for m in MaterializationMode} == {
        "INSERT_ONLY",
        "MERGE",
        "REPLACE_PARTITION",
    }


def test_entity_default_role_is_source() -> None:
    from sema.models.graph_nodes import Entity
    from sema.models.planner._enums import ModelRole

    e = Entity(
        id="e1",
        name="cbio.patient",
        source="cbio",
        confidence=0.9,
        source_id="cbio",
    )
    assert e.model_role == ModelRole.SOURCE
    assert e.target_model_id is None
    assert e.source_id == "cbio"


def test_entity_source_role_requires_source_id() -> None:
    from sema.models.graph_nodes import Entity

    with pytest.raises(ValidationError):
        Entity(id="e", name="x", source="s", confidence=0.9)


def test_entity_target_role_with_kind() -> None:
    from sema.models.graph_nodes import Entity
    from sema.models.planner._enums import ModelRole, TargetArtifactKind

    e = Entity(
        id="e2",
        name="omop.person",
        source="omop_loader",
        confidence=1.0,
        model_role=ModelRole.TARGET,
        target_model_id="omop-cdm-5.4",
        kind=TargetArtifactKind.TABLE_ROW,
    )
    assert e.kind == TargetArtifactKind.TABLE_ROW


def test_entity_target_must_declare_kind() -> None:
    from sema.models.graph_nodes import Entity
    from sema.models.planner._enums import ModelRole

    with pytest.raises(ValidationError):
        Entity(
            id="e3",
            name="omop.person",
            source="omop_loader",
            confidence=1.0,
            model_role=ModelRole.TARGET,
            target_model_id="omop-cdm-5.4",
        )


def test_entity_source_kind_forbidden() -> None:
    from sema.models.graph_nodes import Entity
    from sema.models.planner._enums import ModelRole, TargetArtifactKind

    with pytest.raises(ValidationError):
        Entity(
            id="e4",
            name="cbio.patient",
            source="cbio",
            confidence=0.9,
            model_role=ModelRole.SOURCE,
            kind=TargetArtifactKind.TABLE_ROW,
        )


def test_entity_role_collision_rejected() -> None:
    from sema.models.graph_nodes import Entity
    from sema.models.planner._enums import ModelRole

    with pytest.raises(ValidationError):
        Entity(
            id="e5",
            name="x",
            source="s",
            confidence=0.9,
            model_role=ModelRole.SOURCE,
            source_id="cbio",
            target_model_id="omop-cdm-5.4",
        )


def test_entity_target_requires_target_model_id() -> None:
    from sema.models.graph_nodes import Entity
    from sema.models.planner._enums import ModelRole, TargetArtifactKind

    with pytest.raises(ValidationError):
        Entity(
            id="e6",
            name="omop.person",
            source="loader",
            confidence=1.0,
            model_role=ModelRole.TARGET,
            kind=TargetArtifactKind.TABLE_ROW,
        )


def test_property_role_default() -> None:
    from sema.models.graph_nodes import Property, SemanticType
    from sema.models.planner._enums import ModelRole

    p = Property(
        id="p1",
        name="gender",
        semantic_type=SemanticType.CATEGORICAL,
        source="cbio",
        confidence=0.9,
        source_id="cbio",
    )
    assert p.model_role == ModelRole.SOURCE


def test_term_role_collision_rejected() -> None:
    from sema.models.graph_nodes import Term
    from sema.models.planner._enums import ModelRole

    with pytest.raises(ValidationError):
        Term(
            id="t1",
            code="X",
            label="X",
            source="x",
            confidence=0.9,
            model_role=ModelRole.TARGET,
            target_model_id="omop-cdm-5.4",
            source_id="cbio",
        )


def test_constraint_default_role() -> None:
    from sema.models.planner.target_model import Constraint
    from sema.models.planner._enums import ModelRole

    c = Constraint(
        id="c1", name="not_null", rule_kind="NULLABILITY", source_id="cbio"
    )
    assert c.model_role == ModelRole.SOURCE


def test_foreign_key_obligation() -> None:
    from sema.models.planner.target_model import ForeignKeyObligation

    fk = ForeignKeyObligation(
        referenced_entity="omop.person",
        join_keys=[("person_id", "person_id")],
        same_build_required=True,
    )
    assert fk.referenced_entity == "omop.person"
    assert fk.same_build_required is True


def test_domain_constraint() -> None:
    from sema.models.planner.target_model import DomainConstraint

    dc = DomainConstraint(
        property_name="gender_concept_id",
        domain_id="Gender",
    )
    assert dc.domain_id == "Gender"


def test_row_predicate_and_clause_evaluates() -> None:
    from sema.models.planner.target_model import (
        FieldPresence,
        RowPredicate,
    )

    pred = RowPredicate(
        op="AND",
        clauses=[
            FieldPresence(field="person_id"),
            FieldPresence(field="measurement_concept_id"),
            FieldPresence(field="measurement_date"),
        ],
    )
    assert pred.evaluate({"person_id", "measurement_concept_id", "measurement_date"})
    assert not pred.evaluate({"person_id", "measurement_concept_id"})


def test_row_predicate_or_clause() -> None:
    from sema.models.planner.target_model import (
        FieldPresence,
        RowPredicate,
    )

    pred = RowPredicate(
        op="OR",
        clauses=[FieldPresence(field="bbl"), FieldPresence(field="parcel_id")],
    )
    assert pred.evaluate({"bbl"})
    assert pred.evaluate({"parcel_id"})
    assert not pred.evaluate(set())


def test_row_predicate_field_equality() -> None:
    from sema.models.planner.target_model import (
        FieldEquality,
        RowPredicate,
    )

    pred = RowPredicate(
        op="AND",
        clauses=[FieldEquality(field="status", value="active")],
    )
    assert pred.evaluate({"status"}, values={"status": "active"})
    assert not pred.evaluate({"status"}, values={"status": "archived"})


def test_target_obligation_round_trip() -> None:
    from sema.models.planner._enums import PrimaryKeyStrategy
    from sema.models.planner.target_model import (
        ExternalSequenceMappingTable,
        FieldPresence,
        ForeignKeyObligation,
        RowPredicate,
        TargetObligation,
    )

    ob = TargetObligation(
        target_entity="omop.person",
        required_fields=["person_id", "gender_concept_id", "year_of_birth"],
        nullable_fields=["race_concept_id"],
        primary_key=PrimaryKeyStrategy.EXTERNAL_SEQUENCE,
        external_sequence=ExternalSequenceMappingTable(
            mapping_table_name="cbio_patient_to_omop_person",
            canonical_identity_column="canonical_patient_id",
            sequence_column="person_id",
        ),
        foreign_keys=[
            ForeignKeyObligation(
                referenced_entity="omop.person",
                join_keys=[("person_id", "person_id")],
            )
        ],
        allowed_defaults={"race_concept_id": 0},
        minimum_viable_row=RowPredicate(
            op="AND",
            clauses=[FieldPresence(field="person_id")],
        ),
    )
    payload = ob.model_dump(mode="json")
    rt = TargetObligation.model_validate(payload)
    assert rt.target_entity == ob.target_entity
    assert rt.required_fields == ob.required_fields
    assert rt.primary_key == PrimaryKeyStrategy.EXTERNAL_SEQUENCE


def test_target_obligation_minimum_viable_row_eval() -> None:
    from sema.models.planner._enums import PrimaryKeyStrategy
    from sema.models.planner.target_model import (
        FieldPresence,
        RowPredicate,
        TargetObligation,
    )

    ob = TargetObligation(
        target_entity="omop.measurement",
        required_fields=["person_id", "measurement_concept_id", "measurement_date"],
        primary_key=PrimaryKeyStrategy.NATURAL_KEY,
        minimum_viable_row=RowPredicate(
            op="AND",
            clauses=[
                FieldPresence(field="person_id"),
                FieldPresence(field="measurement_concept_id"),
                FieldPresence(field="measurement_date"),
            ],
        ),
    )
    assert ob.minimum_viable_row.evaluate(set(ob.required_fields))
    assert not ob.minimum_viable_row.evaluate({"person_id", "measurement_concept_id"})


def test_external_sequence_requires_mapping_table() -> None:
    from sema.models.planner._enums import PrimaryKeyStrategy
    from sema.models.planner.target_model import TargetObligation

    with pytest.raises(ValidationError):
        TargetObligation(
            target_entity="omop.person",
            required_fields=["person_id"],
            primary_key=PrimaryKeyStrategy.EXTERNAL_SEQUENCE,
        )


def test_non_external_sequence_rejects_mapping_table() -> None:
    from sema.models.planner._enums import PrimaryKeyStrategy
    from sema.models.planner.target_model import (
        ExternalSequenceMappingTable,
        TargetObligation,
    )

    with pytest.raises(ValidationError):
        TargetObligation(
            target_entity="omop.person",
            required_fields=["person_id"],
            primary_key=PrimaryKeyStrategy.NATURAL_KEY,
            external_sequence=ExternalSequenceMappingTable(
                mapping_table_name="x",
                canonical_identity_column="x",
                sequence_column="x",
            ),
        )

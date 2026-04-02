import json
import pytest
from unittest.mock import MagicMock

pytestmark = pytest.mark.unit

from sema.consumers.nl2sql.consumer import NL2SQLConsumer
from sema.consumers.nl2sql.prompting import build_sql_prompt
from sema.consumers.nl2sql.validation import validate_sql_against_sco
from sema.models.context import (
    SemanticContextObject,
    ResolvedEntity,
    ResolvedProperty,
    PhysicalAsset,
    JoinPath,
    JoinPredicate,
    GovernedValue,
    Provenance,
)


@pytest.fixture
def sample_sco():
    return SemanticContextObject(
        entities=[
            ResolvedEntity(
                name="Cancer Diagnosis",
                description="Primary dx",
                physical_table="cdm.clinical.cancer_diagnosis",
                properties=[
                    ResolvedProperty(
                        name="Diagnosis Type",
                        semantic_type="categorical",
                        physical_column="dx_type_cd",
                        physical_table="cdm.clinical.cancer_diagnosis",
                        provenance=Provenance(source="llm", confidence=0.8),
                    ),
                    ResolvedProperty(
                        name="Stage",
                        semantic_type="categorical",
                        physical_column="tnm_stage",
                        physical_table="cdm.clinical.cancer_diagnosis",
                        provenance=Provenance(source="llm", confidence=0.8),
                    ),
                ],
                provenance=Provenance(source="llm", confidence=0.8),
            ),
        ],
        physical_assets=[
            PhysicalAsset(
                catalog="cdm", schema="clinical",
                table="cancer_diagnosis",
                columns=["dx_type_cd", "tnm_stage", "patient_id", "date_of_diagnosis"],
            ),
            PhysicalAsset(
                catalog="cdm", schema="clinical",
                table="cancer_surgery",
                columns=["patient_id", "procedure_date"],
            ),
        ],
        join_paths=[
            JoinPath(
                from_table="cdm.clinical.cancer_diagnosis",
                to_table="cdm.clinical.cancer_surgery",
                join_predicates=[
                    JoinPredicate(
                        left_table="cdm.clinical.cancer_diagnosis",
                        left_column="patient_id",
                        right_table="cdm.clinical.cancer_surgery",
                        right_column="patient_id",
                    ),
                ],
                cardinality_hint="one-to-many",
                confidence=0.8,
            ),
        ],
        governed_values=[
            GovernedValue(
                property_name="Stage",
                column="tnm_stage",
                table="cancer_diagnosis",
                values=[
                    {"code": "Stage III", "label": "Stage III"},
                    {"code": "Stage IIIA", "label": "Stage IIIA"},
                    {"code": "Stage IIIB", "label": "Stage IIIB"},
                ],
            ),
        ],
        consumer="nl2sql",
    )


class TestPromptConstruction:
    def test_prompt_includes_tables(self, sample_sco):
        prompt = build_sql_prompt(sample_sco, "stage 3 patients")
        assert "cancer_diagnosis" in prompt
        assert "cancer_surgery" in prompt

    def test_prompt_includes_columns(self, sample_sco):
        prompt = build_sql_prompt(sample_sco, "stage 3 patients")
        assert "dx_type_cd" in prompt
        assert "tnm_stage" in prompt

    def test_prompt_includes_join_paths(self, sample_sco):
        prompt = build_sql_prompt(sample_sco, "stage 3 patients")
        assert "patient_id" in prompt

    def test_prompt_includes_governed_values(self, sample_sco):
        prompt = build_sql_prompt(sample_sco, "stage 3 patients")
        assert "Stage III" in prompt
        assert "Stage IIIA" in prompt

    def test_prompt_includes_closed_world_rules(self, sample_sco):
        prompt = build_sql_prompt(sample_sco, "test")
        assert "ONLY" in prompt
        assert "exact" in prompt.lower() or "precisely" in prompt.lower()

    def test_prompt_includes_question(self, sample_sco):
        prompt = build_sql_prompt(sample_sco, "stage 3 colorectal patients")
        assert "stage 3 colorectal patients" in prompt


class TestSQLValidation:
    def test_valid_sql_passes(self, sample_sco):
        sql = (
            "SELECT dx_type_cd, tnm_stage "
            "FROM cdm.clinical.cancer_diagnosis "
            "WHERE tnm_stage IN ('Stage III', 'Stage IIIA')"
        )
        errors = validate_sql_against_sco(sql, sample_sco)
        assert len(errors) == 0

    def test_unknown_table_fails(self, sample_sco):
        sql = "SELECT * FROM cdm.clinical.unknown_table"
        errors = validate_sql_against_sco(sql, sample_sco)
        assert len(errors) > 0
        assert any("unknown_table" in e for e in errors)

    def test_unknown_column_fails(self, sample_sco):
        sql = (
            "SELECT unknown_col "
            "FROM cdm.clinical.cancer_diagnosis"
        )
        errors = validate_sql_against_sco(sql, sample_sco)
        assert len(errors) > 0

    def test_syntax_error_fails(self, sample_sco):
        sql = "SELCT * FORM cancer_diagnosis"
        errors = validate_sql_against_sco(sql, sample_sco)
        assert len(errors) > 0


class TestRetryLoop:
    def test_retry_on_validation_failure(self, sample_sco):
        from sema.consumers.base import ConsumerDeps, ConsumerRequest

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [
            MagicMock(content="SELECT unknown_col FROM cdm.clinical.cancer_diagnosis"),
            MagicMock(content="SELECT dx_type_cd FROM cdm.clinical.cancer_diagnosis"),
        ]
        consumer = NL2SQLConsumer()
        deps = ConsumerDeps(llm=mock_llm)
        req = ConsumerRequest(question="test", operation="plan")
        plan = consumer.plan(req, sample_sco, deps)
        assert plan.valid
        assert mock_llm.invoke.call_count == 2

    def test_max_retries_returns_errors(self, sample_sco):
        from sema.consumers.base import ConsumerDeps, ConsumerRequest

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="SELECT bad_col FROM bad_table"
        )
        consumer = NL2SQLConsumer()
        deps = ConsumerDeps(llm=mock_llm)
        req = ConsumerRequest(question="test", operation="plan")
        plan = consumer.plan(req, sample_sco, deps)
        assert not plan.valid
        assert len(plan.errors) > 0
        assert mock_llm.invoke.call_count == 3  # initial + 2 retries


class TestExecutionModes:
    def test_plan_returns_sql_plan(self, sample_sco):
        from sema.consumers.base import ConsumerDeps, ConsumerRequest

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="SELECT dx_type_cd FROM cdm.clinical.cancer_diagnosis"
        )
        consumer = NL2SQLConsumer()
        deps = ConsumerDeps(llm=mock_llm)
        req = ConsumerRequest(question="test", operation="plan")
        plan = consumer.plan(req, sample_sco, deps)
        assert plan.sql
        assert plan.valid

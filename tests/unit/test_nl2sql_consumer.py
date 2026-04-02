"""Unit tests for NL2SQL consumer — typed API, run() dispatch, prompting, validation."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.unit

from sema.consumers.base import ConsumerDeps, ConsumerRequest
from sema.consumers.nl2sql.consumer import NL2SQLConsumer
from sema.consumers.nl2sql.prompting import build_sql_prompt
from sema.consumers.nl2sql.synthesize import synthesize_results
from sema.consumers.nl2sql.validation import validate_sql_against_sco
from sema.models.context import (
    GovernedValue,
    JoinPath,
    JoinPredicate,
    PhysicalAsset,
    Provenance,
    ResolvedEntity,
    ResolvedProperty,
    SemanticContextObject,
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
                        provenance=Provenance(
                            source="llm", confidence=0.8,
                        ),
                    ),
                ],
                provenance=Provenance(source="llm", confidence=0.8),
            ),
        ],
        physical_assets=[
            PhysicalAsset(
                catalog="cdm", schema="clinical",
                table="cancer_diagnosis",
                columns=[
                    "dx_type_cd", "tnm_stage",
                    "patient_id", "date_of_diagnosis",
                ],
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
                ],
            ),
        ],
        consumer="nl2sql",
    )


# ── Typed API tests (task 4.7) ──


class TestPlanTypedAPI:
    def test_plan_returns_sql_plan(self, sample_sco):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="SELECT dx_type_cd FROM cdm.clinical.cancer_diagnosis",
        )
        consumer = NL2SQLConsumer()
        deps = ConsumerDeps(llm=mock_llm)
        req = ConsumerRequest(question="show diagnoses", operation="plan")
        plan = consumer.plan(req, sample_sco, deps)

        assert plan.valid is True
        assert plan.sql.startswith("SELECT")
        assert plan.attempts == 1
        assert plan.errors == []

    def test_plan_retries_on_invalid(self, sample_sco):
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [
            MagicMock(content="SELECT bad_col FROM cdm.clinical.cancer_diagnosis"),
            MagicMock(content="SELECT dx_type_cd FROM cdm.clinical.cancer_diagnosis"),
        ]
        consumer = NL2SQLConsumer()
        deps = ConsumerDeps(llm=mock_llm)
        req = ConsumerRequest(question="test", operation="plan")
        plan = consumer.plan(req, sample_sco, deps)

        assert plan.valid is True
        assert plan.attempts == 2

    def test_plan_exhausted_retries(self, sample_sco):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="SELECT bad_col FROM bad_table",
        )
        consumer = NL2SQLConsumer()
        deps = ConsumerDeps(llm=mock_llm)
        req = ConsumerRequest(question="test", operation="plan")
        plan = consumer.plan(req, sample_sco, deps)

        assert plan.valid is False
        assert len(plan.errors) > 0
        assert plan.attempts == 3

    def test_plan_llm_error(self, sample_sco):
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("timeout")
        consumer = NL2SQLConsumer()
        deps = ConsumerDeps(llm=mock_llm)
        req = ConsumerRequest(question="test", operation="plan")
        plan = consumer.plan(req, sample_sco, deps)

        assert plan.valid is False
        assert any("LLM error" in e for e in plan.errors)


class TestExplainTypedAPI:
    def test_explain_returns_str(self):
        consumer = NL2SQLConsumer()
        from sema.consumers.nl2sql.consumer import SQLPlan

        plan = SQLPlan(sql="SELECT 1", valid=True)
        runtime = MagicMock()
        runtime.explain.return_value = "Scan parquet"
        deps = ConsumerDeps(sql_runtime=runtime)

        result = consumer.explain(plan, deps)
        assert result == "Scan parquet"
        runtime.explain.assert_called_once_with("SELECT 1")


class TestExecuteTypedAPI:
    def test_execute_returns_sql_result(self, sample_sco):
        consumer = NL2SQLConsumer()
        from sema.consumers.nl2sql.consumer import SQLPlan

        plan = SQLPlan(sql="SELECT 1", valid=True)
        runtime = MagicMock()
        runtime.execute.return_value = {
            "columns": ["a"],
            "rows": [{"a": 1}],
            "row_count": 1,
        }
        llm = MagicMock()
        llm.invoke.return_value = MagicMock(content="Found 1 row.")
        deps = ConsumerDeps(llm=llm, sql_runtime=runtime)
        req = ConsumerRequest(question="test", operation="execute")

        result = consumer.execute(plan, req, deps)
        assert result.data["row_count"] == 1
        assert result.summary == "Found 1 row."


# ── run() protocol dispatch tests (task 4.8) ──


class TestRunDispatch:
    def test_run_plan_returns_consumer_result(self, sample_sco):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="SELECT dx_type_cd FROM cdm.clinical.cancer_diagnosis",
        )
        consumer = NL2SQLConsumer()
        deps = ConsumerDeps(llm=mock_llm)
        req = ConsumerRequest(question="test", operation="plan")
        result = consumer.run(req, sample_sco, deps)

        assert result.valid is True
        assert result.artifact.startswith("SELECT")

    def test_run_gates_execute_on_invalid_plan(self, sample_sco):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="SELECT bad_col FROM bad_table",
        )
        runtime = MagicMock()
        consumer = NL2SQLConsumer()
        deps = ConsumerDeps(llm=mock_llm, sql_runtime=runtime)
        req = ConsumerRequest(question="test", operation="execute")
        result = consumer.run(req, sample_sco, deps)

        assert result.valid is False
        runtime.execute.assert_not_called()

    def test_run_unsupported_operation(self, sample_sco):
        consumer = NL2SQLConsumer()
        deps = ConsumerDeps()
        req = ConsumerRequest(question="test", operation="summarize")
        with pytest.raises(ValueError, match="summarize"):
            consumer.run(req, sample_sco, deps)

    def test_execute_without_runtime_raises(self):
        consumer = NL2SQLConsumer()
        from sema.consumers.nl2sql.consumer import SQLPlan

        plan = SQLPlan(sql="SELECT 1", valid=True)
        deps = ConsumerDeps()
        req = ConsumerRequest(question="test", operation="execute")
        with pytest.raises(ValueError, match="runtime"):
            consumer.execute(plan, req, deps)


# ── Prompt tests (task 4.9) ──


class TestBuildSqlPrompt:
    def test_prompt_includes_tables(self, sample_sco):
        prompt = build_sql_prompt(sample_sco, "show patients")
        assert "cancer_diagnosis" in prompt
        assert "cancer_surgery" in prompt

    def test_prompt_includes_columns(self, sample_sco):
        prompt = build_sql_prompt(sample_sco, "show patients")
        assert "dx_type_cd" in prompt
        assert "tnm_stage" in prompt

    def test_prompt_includes_governed_values(self, sample_sco):
        prompt = build_sql_prompt(sample_sco, "stage 3 patients")
        assert "Stage III" in prompt

    def test_prompt_includes_rules(self, sample_sco):
        prompt = build_sql_prompt(sample_sco, "test")
        assert "ONLY" in prompt

    def test_prompt_budget_truncation(self):
        many_values = [
            {"code": f"VAL_{i}", "label": f"Value {i}"}
            for i in range(200)
        ]
        sco = SemanticContextObject(
            physical_assets=[
                PhysicalAsset(
                    catalog="c", schema="s", table="t",
                    columns=[f"col_{i}" for i in range(50)],
                ),
            ],
            governed_values=[
                GovernedValue(
                    property_name="p", column="c", table="t",
                    values=many_values,
                ),
            ],
            consumer="nl2sql",
        )
        prompt = build_sql_prompt(sco, "test", max_chars=500)
        assert "[truncated]" in prompt


# ── Validation tests (task 4.10) ──


class TestValidation:
    def test_valid_sql_passes(self, sample_sco):
        sql = (
            "SELECT dx_type_cd, tnm_stage "
            "FROM cdm.clinical.cancer_diagnosis "
            "WHERE tnm_stage = 'Stage III'"
        )
        errors = validate_sql_against_sco(sql, sample_sco)
        assert errors == []

    def test_unknown_table_with_closest(self, sample_sco):
        sql = "SELECT * FROM cdm.clinical.cancer_diagnosi"
        errors = validate_sql_against_sco(sql, sample_sco)
        assert len(errors) > 0
        assert "cancer_diagnosi" in errors[0]

    def test_column_on_wrong_table(self, sample_sco):
        sql = (
            "SELECT cancer_surgery.dx_type_cd "
            "FROM cdm.clinical.cancer_surgery"
        )
        errors = validate_sql_against_sco(sql, sample_sco)
        assert len(errors) > 0
        assert "not found on table" in errors[0]

    def test_bare_column_graceful(self, sample_sco):
        sql = (
            "SELECT dx_type_cd "
            "FROM cdm.clinical.cancer_diagnosis"
        )
        errors = validate_sql_against_sco(sql, sample_sco)
        assert errors == []

    def test_bare_unknown_column(self, sample_sco):
        sql = (
            "SELECT unknown_col "
            "FROM cdm.clinical.cancer_diagnosis"
        )
        errors = validate_sql_against_sco(sql, sample_sco)
        assert len(errors) > 0
        assert "Did you mean" in errors[0]

    def test_syntax_error(self, sample_sco):
        errors = validate_sql_against_sco(
            "SELECT FROM WHERE (((", sample_sco,
        )
        assert any("syntax" in e.lower() or "error" in e.lower() for e in errors)

    def test_dialect_parameter(self, sample_sco):
        sql = "SELECT dx_type_cd FROM cdm.clinical.cancer_diagnosis"
        errors = validate_sql_against_sco(
            sql, sample_sco, dialect="databricks",
        )
        assert errors == []


# ── Synthesize tests (reusing existing patterns) ──


class TestSynthesize:
    def test_nonempty_results(self):
        llm = MagicMock()
        llm.invoke.return_value = MagicMock(
            content="Found 5 patients.",
        )
        result = synthesize_results(
            "show patients", "SELECT 1",
            {"rows": [{"a": 1}], "row_count": 1},
            llm,
        )
        assert "patients" in result

    def test_empty_results(self):
        llm = MagicMock()
        llm.invoke.return_value = MagicMock(
            content="No matching records were found.",
        )
        result = synthesize_results(
            "rare query", "SELECT 1",
            {"rows": [], "row_count": 0},
            llm,
        )
        assert "no" in result.lower() or "No" in result

    def test_llm_failure_fallback(self):
        llm = MagicMock()
        llm.invoke.side_effect = Exception("timeout")
        result = synthesize_results(
            "test", "SELECT 1",
            {"rows": [{"a": 1}], "row_count": 1},
            llm,
        )
        assert "1 rows" in result

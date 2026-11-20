"""Unit tests for NL2SQL consumer run() dispatch and the statement's output column list."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from sema.consumers.base import ConsumerDeps, ConsumerRequest
from sema.consumers.nl2sql.consumer import NL2SQLConsumer
from sema.models.config import QueryConfig
from sema.models.context import SemanticContextObject
from sema.pipeline.orchestrate import run_query
from tests.unit.test_nl2sql_consumer import sample_sco  # noqa: F401

pytestmark = pytest.mark.unit

DX = "cdm.clinical.cancer_diagnosis"

STATEMENT_COLUMNS = [
    (f"SELECT tnm_stage AS stage, COUNT(*) FROM {DX} GROUP BY 1", ["stage", "COUNT(*)"]),
    (f"SELECT patient_id, dx_type_cd FROM {DX}", ["patient_id", "dx_type_cd"]),
]


def _llm_returning(sql: str) -> MagicMock:
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content=sql)
    return llm


def _planned_columns(sco: SemanticContextObject, sql: str) -> list[str]:
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content=sql)
    plan = NL2SQLConsumer().plan(
        ConsumerRequest(question="test", operation="plan"), sco, ConsumerDeps(llm=mock_llm),
    )
    assert plan.valid, plan.errors
    return plan.columns


class TestPlanOutputColumns:
    """`SQLPlan.columns` holds one entry per output column, in order: the alias, else the
    bare column's name, else the expression's SQL text."""

    def test_an_unaliased_expression_records_its_sql_text(self, sample_sco):
        sql = f"SELECT tnm_stage || dx_type_cd, dx_type_cd FROM {DX}"
        assert _planned_columns(sample_sco, sql) == ["tnm_stage || dx_type_cd", "dx_type_cd"]

    def test_an_unaliased_aggregate_keeps_its_place(self, sample_sco):
        sql = f"SELECT dx_type_cd, COUNT(*) FROM {DX} GROUP BY 1"
        assert _planned_columns(sample_sco, sql) == ["dx_type_cd", "COUNT(*)"]

    def test_a_star_is_recorded_as_written(self, sample_sco):
        assert _planned_columns(sample_sco, f"SELECT * FROM {DX}") == ["*"]

    def test_a_union_records_the_first_branch_columns(self, sample_sco):
        sql = (f"SELECT dx_type_cd AS kind, COUNT(*) FROM {DX} GROUP BY 1 "
               f"UNION ALL SELECT tnm_stage, COUNT(*) FROM {DX} GROUP BY 1")
        assert _planned_columns(sample_sco, sql) == ["kind", "COUNT(*)"]


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


class TestConsumerResultColumns:
    """The consumer's output records the statement's column list, in order."""

    @pytest.mark.parametrize(("sql", "columns"), STATEMENT_COLUMNS)
    def test_run_carries_the_statement_columns(
        self, sample_sco: SemanticContextObject, sql: str, columns: list[str],
    ) -> None:
        result = NL2SQLConsumer().run(
            ConsumerRequest(question="test", operation="plan"),
            sample_sco,
            ConsumerDeps(llm=_llm_returning(sql)),
        )
        assert result.valid, result.errors
        assert result.columns == columns

    @pytest.mark.parametrize(("sql", "columns"), STATEMENT_COLUMNS)
    def test_query_response_carries_the_statement_columns(
        self, sample_sco: SemanticContextObject, sql: str, columns: list[str],
    ) -> None:
        with (
            patch("sema.pipeline.orchestrate._get_llm", return_value=_llm_returning(sql)),
            patch("sema.pipeline.orchestrate._retrieve_context", return_value=sample_sco),
        ):
            response = run_query(QueryConfig(question="test", operation="plan"))
        assert response["validation"]["valid"], response["validation"]["errors"]
        assert response["columns"] == columns

"""Unit tests for the statement's output column list on the NL2SQL consumer's results."""

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


class TestConsumerResultColumns:
    """The consumer's output records the statement's column list, in order."""

    @pytest.mark.parametrize(("sql", "columns"), STATEMENT_COLUMNS)
    def test_run_carries_the_statement_columns(
        self, sample_sco: SemanticContextObject, sql: str, columns: list[str],  # noqa: F811
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
        self, sample_sco: SemanticContextObject, sql: str, columns: list[str],  # noqa: F811
    ) -> None:
        with (
            patch("sema.pipeline.orchestrate._get_llm", return_value=_llm_returning(sql)),
            patch("sema.pipeline.orchestrate._retrieve_context", return_value=sample_sco),
        ):
            response = run_query(QueryConfig(question="test", operation="plan"))
        assert response["validation"]["valid"], response["validation"]["errors"]
        assert response["columns"] == columns

import json
import pytest
from unittest.mock import MagicMock

pytestmark = pytest.mark.unit

from sema.pipeline.synthesize import synthesize_results


class TestResultSynthesis:
    def test_synthesizes_nonempty_results(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="Found 5 Stage III colorectal patients diagnosed in 2024."
        )
        result = synthesize_results(
            mock_llm,
            "stage 3 colorectal patients",
            "SELECT * FROM cancer_diagnosis WHERE stage = 'Stage III'",
            {"rows": [{"patient_id": "P1", "stage": "Stage III"}], "row_count": 1},
        )
        assert "Stage III" in result or "colorectal" in result

    def test_handles_empty_results(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="No matching records were found for the query."
        )
        result = synthesize_results(
            mock_llm,
            "rare cancer type",
            "SELECT * FROM cancer_diagnosis WHERE dx = 'RARE'",
            {"rows": [], "row_count": 0},
        )
        assert "no" in result.lower() or "No" in result

    def test_llm_failure_returns_fallback(self):
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("LLM timeout")
        result = synthesize_results(
            mock_llm,
            "test",
            "SELECT 1",
            {"rows": [{"a": 1}], "row_count": 1},
        )
        assert "1 rows" in result

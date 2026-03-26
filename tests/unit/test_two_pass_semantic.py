"""Tests for two-pass semantic strategy and prompt optimization (Groups 5 & 6)."""
import math
import pytest
from unittest.mock import MagicMock, patch, call

pytestmark = pytest.mark.unit

from sema.engine.semantic import (
    SemanticEngine,
    TableInterpretation,
    PropertyInterpretation,
    _PropertyBatchResult,
    build_interpretation_prompt,
    build_summary_prompt,
    build_property_prompt,
)
from sema.llm_client import (
    LLMClient,
    TableSummary,
)
from sema.models.assertions import Assertion, AssertionPredicate
from sema.connectors.databricks import DatabricksConnector
from sema.models.config import DatabricksConfig, ProfilingConfig


def _make_columns(n):
    return [{"name": f"col_{i}", "data_type": "STRING"} for i in range(n)]


def _make_columns_with_values(n, top_k=10):
    cols = []
    for i in range(n):
        cols.append({
            "name": f"col_{i}",
            "data_type": "STRING",
            "top_values": [{"value": f"val_{j}"} for j in range(top_k)],
        })
    return cols


# ---------------------------------------------------------------------------
# Table summary pass tests (Task 5.1)
# ---------------------------------------------------------------------------

class TestTableSummaryPass:
    def test_summary_prompt_includes_all_column_names(self):
        meta = {
            "table_name": "patients",
            "comment": "Patient records",
            "columns": _make_columns(100),
        }
        prompt = build_summary_prompt(meta)
        assert "patients" in prompt
        assert "Patient records" in prompt
        # All 100 columns present
        for i in range(100):
            assert f"col_{i}" in prompt

    def test_summary_prompt_is_lightweight(self):
        """Even for 200 columns, the summary prompt should be small."""
        meta = {
            "table_name": "wide_table",
            "columns": _make_columns(200),
        }
        prompt = build_summary_prompt(meta)
        # Should be under ~5KB (just names and types)
        assert len(prompt) < 5000

    def test_summary_prompt_no_values(self):
        """Summary prompt should not include top values."""
        meta = {
            "table_name": "tbl",
            "columns": _make_columns_with_values(10),
        }
        prompt = build_summary_prompt(meta)
        assert "val_" not in prompt


# ---------------------------------------------------------------------------
# Property extraction pass tests (Task 5.2)
# ---------------------------------------------------------------------------

class TestPropertyExtractionPass:
    def test_property_prompt_includes_entity_context(self):
        meta = {"table_name": "patients", "columns": _make_columns(5)}
        prompt = build_property_prompt(meta, meta["columns"], "Patient")
        assert "This table represents: Patient" in prompt

    def test_property_prompt_includes_batch_columns_only(self):
        all_cols = _make_columns(10)
        batch = all_cols[:5]
        meta = {"table_name": "tbl"}
        prompt = build_property_prompt(meta, batch, "Entity")
        for i in range(5):
            assert f"col_{i}" in prompt
        for i in range(5, 10):
            assert f"col_{i}" not in prompt


# ---------------------------------------------------------------------------
# Threshold behavior tests (Task 5.3)
# ---------------------------------------------------------------------------

class TestThresholdBehavior:
    def test_under_threshold_uses_single_call(self):
        mock_client = MagicMock(spec=LLMClient)
        mock_client.invoke.return_value = TableInterpretation(
            entity_name="Entity", properties=[]
        )

        engine = SemanticEngine(
            llm_client=mock_client, run_id="test", column_batch_size=25
        )
        meta = {
            "table_ref": "unity://cdm.clinical.tbl",
            "table_name": "tbl",
            "columns": _make_columns(40),  # under 50 threshold
            "sample_rows": [],
            "comment": None,
        }
        engine.interpret_table(meta)

        # Single call (no summary pass)
        assert mock_client.invoke.call_count == 1
        # Called with TableInterpretation schema
        args = mock_client.invoke.call_args
        assert args[0][1] == TableInterpretation

    def test_over_threshold_uses_two_pass(self):
        mock_client = MagicMock(spec=LLMClient)
        # First call: summary
        mock_client.invoke.side_effect = [
            TableSummary(entity_name="Patient", synonyms=["pt"]),
            _PropertyBatchResult(properties=[
                PropertyInterpretation(
                    column=f"col_{i}", name=f"Col {i}",
                    semantic_type="free_text",
                )
                for i in range(25)
            ]),
            _PropertyBatchResult(properties=[
                PropertyInterpretation(
                    column=f"col_{i}", name=f"Col {i}",
                    semantic_type="free_text",
                )
                for i in range(25, 50)
            ]),
            _PropertyBatchResult(properties=[
                PropertyInterpretation(
                    column=f"col_{i}", name=f"Col {i}",
                    semantic_type="free_text",
                )
                for i in range(50, 75)
            ]),
            _PropertyBatchResult(properties=[
                PropertyInterpretation(
                    column=f"col_{i}", name=f"Col {i}",
                    semantic_type="free_text",
                )
                for i in range(75, 80)
            ]),
        ]

        engine = SemanticEngine(
            llm_client=mock_client, run_id="test", column_batch_size=25
        )
        meta = {
            "table_ref": "unity://cdm.clinical.tbl",
            "table_name": "tbl",
            "columns": _make_columns(80),  # 80 >= 50 threshold
            "sample_rows": [],
            "comment": None,
        }
        assertions = engine.interpret_table(meta)

        # 1 summary + ceil(80/25) = 5 calls total
        assert mock_client.invoke.call_count == 5

        # Entity name from summary
        entity = [
            a for a in assertions
            if a.predicate == AssertionPredicate.HAS_ENTITY_NAME
        ]
        assert len(entity) == 1
        assert entity[0].payload["value"] == "Patient"

        # Properties from all batches
        props = [
            a for a in assertions
            if a.predicate == AssertionPredicate.HAS_PROPERTY_NAME
        ]
        assert len(props) == 80  # all columns covered


# ---------------------------------------------------------------------------
# Chunking math tests (Task 5.4)
# ---------------------------------------------------------------------------

class TestChunkingMath:
    def test_80_columns_batch_25(self):
        """80 / 25 = 4 batches: 25, 25, 25, 5"""
        columns = _make_columns(80)
        batch_size = 25
        batches = [
            columns[i:i + batch_size]
            for i in range(0, len(columns), batch_size)
        ]
        assert len(batches) == 4
        assert len(batches[0]) == 25
        assert len(batches[1]) == 25
        assert len(batches[2]) == 25
        assert len(batches[3]) == 5

    def test_50_columns_batch_25(self):
        """50 / 25 = 2 batches: 25, 25"""
        columns = _make_columns(50)
        batch_size = 25
        batches = [
            columns[i:i + batch_size]
            for i in range(0, len(columns), batch_size)
        ]
        assert len(batches) == 2
        assert all(len(b) == 25 for b in batches)


# ---------------------------------------------------------------------------
# Prompt compression tests (Task 6.1)
# ---------------------------------------------------------------------------

class TestPromptCompression:
    def test_top_values_truncated_to_max(self):
        meta = {
            "table_name": "tbl",
            "columns": [{
                "name": "status",
                "data_type": "STRING",
                "top_values": [{"value": f"v{i}"} for i in range(20)],
            }],
            "sample_rows": [],
        }
        prompt = build_interpretation_prompt(meta, max_sample_values=5)
        # Should include v0-v4 but not v5+
        assert "v0" in prompt
        assert "v4" in prompt
        assert "v5" not in prompt

    def test_columns_with_fewer_values(self):
        meta = {
            "table_name": "tbl",
            "columns": [{
                "name": "status",
                "data_type": "STRING",
                "top_values": [{"value": "a"}, {"value": "b"}],
            }],
            "sample_rows": [],
        }
        prompt = build_interpretation_prompt(meta, max_sample_values=5)
        assert "a" in prompt
        assert "b" in prompt


# ---------------------------------------------------------------------------
# Profiling skip tests (Tasks 6.2 & 6.3)
# ---------------------------------------------------------------------------

class TestProfilingSkip:
    def _make_connector(self, mock_connection, profiling):
        with patch(
            "sema.connectors.databricks.sql_connect"
        ) as mock_connect:
            mock_connect.return_value = mock_connection
            config = DatabricksConfig(
                host="https://test.databricks.com",
                token="dapi123",
                http_path="/sql/1.0/warehouses/test",
            )
            conn = DatabricksConnector(config=config, profiling=profiling)
            conn._connection = mock_connection
            return conn

    def test_temporal_skip_by_default(self):
        profiling = ProfilingConfig()
        assert profiling.skip_temporal_profiling is True

        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        connector = self._make_connector(conn, profiling)
        assert connector._should_skip_profiling("TIMESTAMP_NTZ") is True
        assert connector._should_skip_profiling("DATE") is True
        assert connector._should_skip_profiling("STRING") is False

    def test_temporal_skip_disabled(self):
        profiling = ProfilingConfig(skip_temporal_profiling=False)

        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        connector = self._make_connector(conn, profiling)
        assert connector._should_skip_profiling("TIMESTAMP_NTZ") is False
        assert connector._should_skip_profiling("DATE") is False

    def test_numeric_not_skipped_by_default(self):
        profiling = ProfilingConfig()
        assert profiling.skip_numeric_profiling is False

        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        connector = self._make_connector(conn, profiling)
        assert connector._should_skip_profiling("INT") is False
        assert connector._should_skip_profiling("DECIMAL(18,4)") is False

    def test_numeric_skipped_when_enabled(self):
        profiling = ProfilingConfig(skip_numeric_profiling=True)

        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        connector = self._make_connector(conn, profiling)
        assert connector._should_skip_profiling("INT") is True
        assert connector._should_skip_profiling("DECIMAL(18,4)") is True

    def test_string_always_profiled(self):
        profiling = ProfilingConfig(
            skip_temporal_profiling=True,
            skip_numeric_profiling=True,
        )

        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        connector = self._make_connector(conn, profiling)
        assert connector._should_skip_profiling("STRING") is False
        assert connector._should_skip_profiling("VARCHAR") is False


# ---------------------------------------------------------------------------
# Characterization: _interpret_two_pass
# ---------------------------------------------------------------------------

class TestInterpretTwoPassCharacterization:
    """Characterization tests capturing current behavior of _interpret_two_pass."""

    def test_two_pass_returns_assertions_for_wide_table(self):
        mock_client = MagicMock(spec=LLMClient)

        # First call: summary pass
        mock_client.invoke.side_effect = [
            TableSummary(
                entity_name="Patient",
                entity_description="Patient records",
                synonyms=["Subject"],
            ),
            # Batch 1: cols 0-4
            _PropertyBatchResult(properties=[
                PropertyInterpretation(
                    column=f"col_{i}", name=f"Col {i}",
                    semantic_type="free_text",
                )
                for i in range(5)
            ]),
            # Batch 2: cols 5-9
            _PropertyBatchResult(properties=[
                PropertyInterpretation(
                    column=f"col_{i}", name=f"Col {i}",
                    semantic_type="free_text",
                )
                for i in range(5, 10)
            ]),
            # Batch 3: cols 10-14
            _PropertyBatchResult(properties=[
                PropertyInterpretation(
                    column=f"col_{i}", name=f"Col {i}",
                    semantic_type="free_text",
                )
                for i in range(10, 15)
            ]),
        ]

        engine = SemanticEngine(
            llm_client=mock_client, run_id="test", column_batch_size=5,
        )
        table_metadata = {
            "table_ref": "unity://cat.sch.tbl",
            "table_name": "tbl",
            "columns": _make_columns(15),
            "sample_rows": [],
            "comment": None,
        }

        assertions = engine._interpret_two_pass(
            table_metadata, "unity://cat.sch.tbl"
        )

        # Returns a list of Assertion objects
        assert isinstance(assertions, list)
        assert all(isinstance(a, Assertion) for a in assertions)

        # Has HAS_ENTITY_NAME assertion
        entity_assertions = [
            a for a in assertions
            if a.predicate == AssertionPredicate.HAS_ENTITY_NAME
        ]
        assert len(entity_assertions) == 1
        assert entity_assertions[0].payload["value"] == "Patient"

        # Has HAS_PROPERTY_NAME for each column across all batches
        prop_assertions = [
            a for a in assertions
            if a.predicate == AssertionPredicate.HAS_PROPERTY_NAME
        ]
        assert len(prop_assertions) == 15

        # 1 summary call + 3 batch calls (15 / 5 = 3)
        assert mock_client.invoke.call_count == 4

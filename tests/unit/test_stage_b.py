"""Tests for Stage B: property classification with batching and recovery.

Covers: StageBColumnResult/StageBBatchResult/StageBResult schemas,
prompt construction, batching, bounded recovery, coverage computation,
pass/fail logic.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, call

import pytest

from sema.llm_client import LLMStageError
from sema.models.stages import (
    StageAResult,
    StageBBatchResult,
    StageBColumnResult,
    StageBCoverage,
    StageBResult,
    UnresolvedColumn,
)

pytestmark = pytest.mark.unit


# -- Fixtures ---------------------------------------------------------------

STAGE_A_CONTEXT = StageAResult(
    primary_entity="Somatic Mutation",
    grain_hypothesis="one row per variant call per sample",
    secondary_entity_hints=["gene", "protein change"],
    ambiguity_flags=[],
    confidence=0.88,
)

COLUMNS_5: list[dict[str, Any]] = [
    {"name": "patient_id", "data_type": "STRING", "comment": "Patient identifier"},
    {"name": "sample_id", "data_type": "STRING"},
    {"name": "Hugo_Symbol", "data_type": "STRING",
     "top_values": [{"value": "TP53"}, {"value": "KRAS"}]},
    {"name": "Variant_Classification", "data_type": "STRING",
     "top_values": [{"value": "Missense_Mutation"}, {"value": "Silent"}]},
    {"name": "t_alt_count", "data_type": "INT"},
]

TABLE_METADATA: dict[str, Any] = {
    "table_ref": "unity://catalog.schema.data_mutations",
    "table_name": "data_mutations",
    "comment": "Somatic mutation calls",
    "columns": COLUMNS_5,
}


def _col_result(name: str, **overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "column": name,
        "canonical_property_label": name.replace("_", " ").title(),
        "semantic_type": "identifier",
        "candidate_vocab_families": [],
        "entity_role": None,
        "grain_confirmation": None,
        "needs_stage_c": False,
        "ambiguity_notes": [],
        "evidence": [],
    }
    base.update(overrides)
    return base


def _batch_result(
    col_names: list[str], **overrides: Any,
) -> StageBBatchResult:
    cols = [StageBColumnResult(**_col_result(n)) for n in col_names]
    return StageBBatchResult(columns=cols, **overrides)


# -- 3.1 StageBColumnResult schema -----------------------------------------

class TestStageBColumnResultSchema:
    def test_valid_full_result(self) -> None:
        r = StageBColumnResult(**_col_result(
            "Hugo_Symbol",
            semantic_type="gene_identifier",
            candidate_vocab_families=["gene symbol namespace"],
            entity_role="secondary",
            needs_stage_c=False,
            evidence=["column name matches HUGO gene symbol pattern"],
        ))
        assert r.column == "Hugo_Symbol"
        assert r.semantic_type == "gene_identifier"
        assert r.candidate_vocab_families == ["gene symbol namespace"]

    def test_minimal_required_fields(self) -> None:
        r = StageBColumnResult(
            column="x", canonical_property_label="X",
            semantic_type="identifier",
        )
        assert r.column == "x"
        assert r.needs_stage_c is False
        assert r.candidate_vocab_families == []
        assert r.ambiguity_notes == []

    def test_needs_stage_c_flag(self) -> None:
        r = StageBColumnResult(
            column="dx_type_cd",
            canonical_property_label="Diagnosis Type",
            semantic_type="categorical",
            needs_stage_c=True,
        )
        assert r.needs_stage_c is True

    def test_confidence_field(self) -> None:
        r = StageBColumnResult(
            column="x", canonical_property_label="X",
            semantic_type="identifier", confidence=0.92,
        )
        assert r.confidence == 0.92

    def test_confidence_defaults_to_075(self) -> None:
        r = StageBColumnResult(
            column="x", canonical_property_label="X",
            semantic_type="identifier",
        )
        assert r.confidence == 0.75

    def test_synonyms_field(self) -> None:
        r = StageBColumnResult(
            column="x", canonical_property_label="X",
            semantic_type="identifier",
            synonyms=["alt_name", "other_name"],
        )
        assert r.synonyms == ["alt_name", "other_name"]

    def test_synonyms_default_empty(self) -> None:
        r = StageBColumnResult(
            column="x", canonical_property_label="X",
            semantic_type="identifier",
        )
        assert r.synonyms == []


# -- 3.2 StageBBatchResult schema ------------------------------------------

class TestStageBBatchResultSchema:
    def test_batch_with_columns(self) -> None:
        batch = _batch_result(["patient_id", "sample_id"])
        assert len(batch.columns) == 2
        assert batch.grain_correction is None

    def test_batch_with_grain_correction(self) -> None:
        batch = StageBBatchResult(
            columns=[StageBColumnResult(**_col_result("x"))],
            grain_correction="one row per sample, not per patient",
        )
        assert batch.grain_correction is not None

    def test_empty_batch(self) -> None:
        batch = StageBBatchResult(columns=[])
        assert len(batch.columns) == 0


# -- 3.3 UnresolvedColumn model --------------------------------------------

class TestUnresolvedColumn:
    def test_execution_failure(self) -> None:
        u = UnresolvedColumn(
            column="bad_col",
            reason="execution_failure",
            tier="critical",
        )
        assert u.reason == "execution_failure"
        assert u.tier == "critical"

    def test_semantic_unresolved(self) -> None:
        u = UnresolvedColumn(
            column="ambig_col",
            reason="semantic_unresolved",
            tier="peripheral",
        )
        assert u.reason == "semantic_unresolved"

    def test_invalid_reason_rejected(self) -> None:
        with pytest.raises(ValueError):
            UnresolvedColumn(
                column="x", reason="bad_reason", tier="critical",  # type: ignore[arg-type]
            )

    def test_invalid_tier_rejected(self) -> None:
        with pytest.raises(ValueError):
            UnresolvedColumn(
                column="x", reason="execution_failure", tier="unknown",  # type: ignore[arg-type]
            )


# -- 3.4 StageBResult model ------------------------------------------------

class TestStageBResultSchema:
    def test_success_result(self) -> None:
        r = StageBResult(
            status="B_SUCCESS",
            batch_results=[_batch_result(["a", "b", "c"])],
            raw_coverage=StageBCoverage(classified=3, total=3, pct=1.0),
            critical_coverage=StageBCoverage(classified=1, total=1, pct=1.0),
        )
        assert r.status == "B_SUCCESS"
        assert r.raw_coverage.pct == 1.0

    def test_partial_result(self) -> None:
        r = StageBResult(
            status="B_PARTIAL",
            batch_results=[_batch_result(["a", "b"])],
            raw_coverage=StageBCoverage(classified=2, total=3, pct=0.67),
            critical_coverage=StageBCoverage(classified=1, total=1, pct=1.0),
            unresolved_columns=[
                UnresolvedColumn(
                    column="c", reason="execution_failure", tier="peripheral",
                ),
            ],
        )
        assert r.status == "B_PARTIAL"
        assert len(r.unresolved_columns) == 1

    def test_failed_result(self) -> None:
        r = StageBResult(
            status="B_FAILED",
            raw_coverage=StageBCoverage(classified=0, total=5, pct=0.0),
            critical_coverage=StageBCoverage(classified=0, total=2, pct=0.0),
            unresolved_columns=[
                UnresolvedColumn(
                    column="x", reason="execution_failure", tier="critical",
                ),
            ],
        )
        assert r.status == "B_FAILED"

    def test_invalid_status_rejected(self) -> None:
        with pytest.raises(ValueError):
            StageBResult(
                status="INVALID",  # type: ignore[arg-type]
                raw_coverage=StageBCoverage(classified=0, total=0, pct=0.0),
                critical_coverage=StageBCoverage(classified=0, total=0, pct=0.0),
            )


# -- 3.5 Stage B prompt construction ---------------------------------------

class TestStageBPrompt:
    def test_prompt_includes_column_batch(self) -> None:
        from sema.engine.stage_utils import build_stage_b_prompt
        prompt = build_stage_b_prompt(
            TABLE_METADATA, COLUMNS_5[:2], STAGE_A_CONTEXT,
        )
        assert "patient_id" in prompt
        assert "sample_id" in prompt

    def test_prompt_includes_stage_a_context(self) -> None:
        from sema.engine.stage_utils import build_stage_b_prompt
        prompt = build_stage_b_prompt(
            TABLE_METADATA, COLUMNS_5, STAGE_A_CONTEXT,
        )
        assert "Somatic Mutation" in prompt
        assert "one row per variant call per sample" in prompt

    def test_prompt_includes_top_values(self) -> None:
        from sema.engine.stage_utils import build_stage_b_prompt
        prompt = build_stage_b_prompt(
            TABLE_METADATA, COLUMNS_5, STAGE_A_CONTEXT,
        )
        assert "TP53" in prompt

    def test_prompt_requests_stage_b_fields(self) -> None:
        from sema.engine.stage_utils import build_stage_b_prompt
        prompt = build_stage_b_prompt(
            TABLE_METADATA, COLUMNS_5, STAGE_A_CONTEXT,
        )
        assert "canonical_property_label" in prompt
        assert "semantic_type" in prompt
        assert "candidate_vocab_families" in prompt
        assert "needs_stage_c" in prompt

    def test_prompt_requests_synonyms_and_confidence(self) -> None:
        from sema.engine.stage_utils import build_stage_b_prompt
        prompt = build_stage_b_prompt(
            TABLE_METADATA, COLUMNS_5, STAGE_A_CONTEXT,
        )
        assert "synonyms" in prompt.lower()
        assert "confidence" in prompt.lower()

    def test_prompt_warns_against_specific_ontology(self) -> None:
        from sema.engine.stage_utils import build_stage_b_prompt
        prompt = build_stage_b_prompt(
            TABLE_METADATA, COLUMNS_5, STAGE_A_CONTEXT,
        )
        lower = prompt.lower()
        assert "do not name a specific ontology" in lower

    def test_prompt_includes_semantic_type_inventory(self) -> None:
        from sema.engine.stage_utils import build_stage_b_prompt
        prompt = build_stage_b_prompt(
            TABLE_METADATA, COLUMNS_5, STAGE_A_CONTEXT,
        )
        assert "identifier" in prompt
        assert "categorical" in prompt
        assert "temporal" in prompt

    def test_prompt_domain_slot_empty_by_default(self) -> None:
        from sema.engine.stage_utils import build_stage_b_prompt
        prompt = build_stage_b_prompt(
            TABLE_METADATA, COLUMNS_5, STAGE_A_CONTEXT,
        )
        # No domain bias when domain_context is None
        assert "domain" not in prompt.lower().split("entity context")[0]

    def test_prompt_domain_slot_accepted(self) -> None:
        from sema.engine.stage_utils import build_stage_b_prompt
        from sema.models.domain import DomainContext
        ctx = DomainContext(declared_domain="healthcare", domain_source="user")
        prompt = build_stage_b_prompt(
            TABLE_METADATA, COLUMNS_5, STAGE_A_CONTEXT,
            domain_context=ctx,
        )
        # Just verify the parameter is accepted
        assert "patient_id" in prompt


# -- 3.6 SemanticEngine.run_stage_b() with batching -----------------------

class TestRunStageB:
    def _make_engine(
        self, batch_responses: list[StageBBatchResult],
        batch_size: int = 25,
    ) -> Any:
        from sema.engine.semantic import SemanticEngine

        mock_client = MagicMock()
        mock_client.invoke.side_effect = batch_responses
        return SemanticEngine(
            llm_client=mock_client, run_id="test-run",
            column_batch_size=batch_size,
        )

    def test_single_batch_for_narrow_table(self) -> None:
        batch = _batch_result(["patient_id", "sample_id", "Hugo_Symbol",
                               "Variant_Classification", "t_alt_count"])
        engine = self._make_engine([batch])
        result = engine.run_stage_b(TABLE_METADATA, STAGE_A_CONTEXT)
        assert isinstance(result, StageBResult)
        assert result.status == "B_SUCCESS"
        assert result.raw_coverage.classified == 5
        assert engine._llm_client.invoke.call_count == 1

    def test_multiple_batches_for_wide_table(self) -> None:
        cols = [
            {"name": f"col_{i}", "data_type": "STRING"} for i in range(7)
        ]
        table = {**TABLE_METADATA, "columns": cols}
        batch1 = _batch_result(["col_0", "col_1", "col_2"])
        batch2 = _batch_result(["col_3", "col_4", "col_5"])
        batch3 = _batch_result(["col_6"])
        engine = self._make_engine([batch1, batch2, batch3], batch_size=3)
        result = engine.run_stage_b(table, STAGE_A_CONTEXT)
        assert result.status == "B_SUCCESS"
        assert result.raw_coverage.classified == 7
        assert engine._llm_client.invoke.call_count == 3

    def test_returns_stage_b_result_not_assertions(self) -> None:
        batch = _batch_result(["patient_id"])
        table = {**TABLE_METADATA, "columns": COLUMNS_5[:1]}
        engine = self._make_engine([batch])
        result = engine.run_stage_b(table, STAGE_A_CONTEXT)
        assert isinstance(result, StageBResult)
        assert not isinstance(result, list)

    def test_passes_correct_schema_to_client(self) -> None:
        batch = _batch_result(["patient_id"])
        table = {**TABLE_METADATA, "columns": COLUMNS_5[:1]}
        engine = self._make_engine([batch])
        engine.run_stage_b(table, STAGE_A_CONTEXT)
        call_args = engine._llm_client.invoke.call_args
        assert call_args[0][1] is StageBBatchResult

    def test_stage_name_includes_stage_b(self) -> None:
        batch = _batch_result(["patient_id"])
        table = {**TABLE_METADATA, "columns": COLUMNS_5[:1]}
        engine = self._make_engine([batch])
        engine.run_stage_b(table, STAGE_A_CONTEXT)
        call_kwargs = engine._llm_client.invoke.call_args[1]
        assert "stage_b" in call_kwargs["stage_name"].lower()

    def test_grain_correction_propagated(self) -> None:
        batch = StageBBatchResult(
            columns=[StageBColumnResult(**_col_result("x"))],
            grain_correction="actually per-sample",
        )
        table = {**TABLE_METADATA, "columns": [COLUMNS_5[0]]}
        engine = self._make_engine([batch])
        result = engine.run_stage_b(table, STAGE_A_CONTEXT)
        assert any(
            br.grain_correction is not None
            for br in result.batch_results
        )


# -- 3.7 Bounded recovery -------------------------------------------------

class TestBoundedRecovery:
    def _make_engine(self, side_effects: list[Any], batch_size: int = 25) -> Any:
        from sema.engine.semantic import SemanticEngine
        mock_client = MagicMock()
        mock_client.invoke.side_effect = side_effects
        return SemanticEngine(
            llm_client=mock_client, run_id="test-run",
            column_batch_size=batch_size,
        )

    def test_retry_on_execution_failure(self) -> None:
        """One retry max for execution failure."""
        error = LLMStageError(
            table_ref="test", stage_name="stage_b",
            step_errors=[("structured_output", ValueError("parse error"))],
        )
        good = _batch_result(["patient_id", "sample_id"])
        table = {**TABLE_METADATA, "columns": COLUMNS_5[:2]}
        engine = self._make_engine([error, good])
        result = engine.run_stage_b(table, STAGE_A_CONTEXT)
        assert result.status == "B_SUCCESS"
        assert engine._llm_client.invoke.call_count == 2

    def test_split_on_second_failure(self) -> None:
        """After retry fails, split batch into two smaller ones."""
        error = LLMStageError(
            table_ref="test", stage_name="stage_b",
            step_errors=[("structured_output", ValueError("bad"))],
        )
        half1 = _batch_result(["patient_id"])
        half2 = _batch_result(["sample_id"])
        table = {**TABLE_METADATA, "columns": COLUMNS_5[:2]}
        engine = self._make_engine([error, error, half1, half2])
        result = engine.run_stage_b(table, STAGE_A_CONTEXT)
        assert result.status == "B_SUCCESS"
        assert result.raw_coverage.classified == 2

    def test_no_unlimited_recursion(self) -> None:
        """Recovery stops after retry + split — no further retries on sub-batches."""
        error = LLMStageError(
            table_ref="test", stage_name="stage_b",
            step_errors=[("structured_output", ValueError("bad"))],
        )
        cols = [{"name": f"col_{i}", "data_type": "STRING"} for i in range(4)]
        table = {**TABLE_METADATA, "columns": cols}
        # All calls fail: initial, retry, split-left, split-right
        engine = self._make_engine([error, error, error, error], batch_size=25)
        result = engine.run_stage_b(table, STAGE_A_CONTEXT)
        # Should end up B_FAILED with bounded call count
        assert result.status == "B_FAILED"
        assert engine._llm_client.invoke.call_count <= 4


    def test_rescue_recovers_critical_column(self) -> None:
        """After retry+split fail, rescue call recovers critical columns."""
        error = LLMStageError(
            table_ref="test", stage_name="stage_b",
            step_errors=[("structured_output", ValueError("bad"))],
        )
        # patient_id is critical (matches _id pattern)
        cols = [
            {"name": "patient_id", "data_type": "STRING"},
            {"name": "notes", "data_type": "STRING"},
        ]
        table = {**TABLE_METADATA, "columns": cols}
        rescued = _batch_result(["patient_id"])
        # initial fail, retry fail, split: both halves fail, rescue succeeds
        engine = self._make_engine(
            [error, error, error, error, rescued], batch_size=25,
        )
        result = engine.run_stage_b(table, STAGE_A_CONTEXT)
        classified = [
            cr.column for br in result.batch_results for cr in br.columns
        ]
        assert "patient_id" in classified
        assert result.rescues_used == 1

    def test_recovery_counters_tracked(self) -> None:
        """Retries, splits, rescues are counted in StageBResult."""
        error = LLMStageError(
            table_ref="test", stage_name="stage_b",
            step_errors=[("structured_output", ValueError("bad"))],
        )
        good = _batch_result(["patient_id", "sample_id"])
        table = {**TABLE_METADATA, "columns": COLUMNS_5[:2]}
        engine = self._make_engine([error, good])
        result = engine.run_stage_b(table, STAGE_A_CONTEXT)
        assert result.retries_used == 1


# -- 3.8 Critical column identification -----------------------------------

class TestCriticalColumns:
    def test_grain_relevant_columns_are_critical(self) -> None:
        from sema.engine.stage_utils import identify_critical_columns
        cols = ["patient_id", "sample_id", "Hugo_Symbol", "some_flag"]
        critical = identify_critical_columns(cols, STAGE_A_CONTEXT)
        # Columns matching entity/key patterns should be tier 1
        assert "patient_id" in critical
        assert "sample_id" in critical

    def test_key_pattern_columns_are_critical(self) -> None:
        from sema.engine.stage_utils import identify_critical_columns
        cols = ["record_key", "entity_id", "mutation_pk", "notes"]
        stage_a = StageAResult(
            primary_entity="Record",
            grain_hypothesis="one row per record",
            confidence=0.9,
        )
        critical = identify_critical_columns(cols, stage_a)
        assert "record_key" in critical
        assert "entity_id" in critical

    def test_user_config_critical_columns(self) -> None:
        from sema.engine.stage_utils import identify_critical_columns
        cols = ["custom_col", "notes"]
        stage_a = StageAResult(
            primary_entity="Record",
            grain_hypothesis="one row per record",
            confidence=0.9,
        )
        critical = identify_critical_columns(
            cols, stage_a, user_critical={"custom_col"},
        )
        assert "custom_col" in critical

    def test_important_tier_for_columns_with_metadata(self) -> None:
        from sema.engine.stage_utils import classify_column_tier
        cols_meta = [
            {"name": "dx_code", "comment": "Diagnosis code", "top_values": None},
            {"name": "notes", "comment": None, "top_values": None},
        ]
        assert classify_column_tier("dx_code", set(), cols_meta) == "important"
        assert classify_column_tier("notes", set(), cols_meta) == "peripheral"

    def test_critical_tier_overrides_important(self) -> None:
        from sema.engine.stage_utils import classify_column_tier
        cols_meta = [
            {"name": "patient_id", "comment": "PK"},
        ]
        assert classify_column_tier(
            "patient_id", {"patient_id"}, cols_meta,
        ) == "critical"


# -- 3.9 Coverage computation ----------------------------------------------

class TestCoverageComputation:
    def test_full_coverage(self) -> None:
        from sema.engine.stage_utils import compute_b_coverage
        classified = ["a", "b", "c"]
        total = ["a", "b", "c"]
        cov = compute_b_coverage(classified, total)
        assert cov.classified == 3
        assert cov.total == 3
        assert cov.pct == 1.0

    def test_partial_coverage(self) -> None:
        from sema.engine.stage_utils import compute_b_coverage
        classified = ["a", "b"]
        total = ["a", "b", "c", "d"]
        cov = compute_b_coverage(classified, total)
        assert cov.classified == 2
        assert cov.total == 4
        assert cov.pct == 0.5

    def test_zero_coverage(self) -> None:
        from sema.engine.stage_utils import compute_b_coverage
        cov = compute_b_coverage([], ["a", "b"])
        assert cov.pct == 0.0

    def test_empty_total(self) -> None:
        from sema.engine.stage_utils import compute_b_coverage
        cov = compute_b_coverage([], [])
        assert cov.pct == 1.0


# -- 3.10 Pass/fail logic --------------------------------------------------

class TestPassFailLogic:
    def test_success_when_all_covered(self) -> None:
        from sema.engine.stage_utils import determine_b_status
        status = determine_b_status(
            raw_coverage=StageBCoverage(classified=10, total=10, pct=1.0),
            critical_coverage=StageBCoverage(classified=2, total=2, pct=1.0),
            unresolved=[],
        )
        assert status == "B_SUCCESS"

    def test_partial_above_threshold(self) -> None:
        from sema.engine.stage_utils import determine_b_status
        status = determine_b_status(
            raw_coverage=StageBCoverage(classified=8, total=10, pct=0.8),
            critical_coverage=StageBCoverage(classified=2, total=2, pct=1.0),
            unresolved=[
                UnresolvedColumn(
                    column="x", reason="execution_failure", tier="peripheral",
                ),
            ],
        )
        assert status == "B_PARTIAL"

    def test_failed_below_threshold(self) -> None:
        from sema.engine.stage_utils import determine_b_status
        status = determine_b_status(
            raw_coverage=StageBCoverage(classified=3, total=10, pct=0.3),
            critical_coverage=StageBCoverage(classified=2, total=2, pct=1.0),
            unresolved=[],
        )
        assert status == "B_FAILED"

    def test_failed_when_critical_missing(self) -> None:
        from sema.engine.stage_utils import determine_b_status
        status = determine_b_status(
            raw_coverage=StageBCoverage(classified=9, total=10, pct=0.9),
            critical_coverage=StageBCoverage(classified=1, total=2, pct=0.5),
            unresolved=[
                UnresolvedColumn(
                    column="pk", reason="execution_failure", tier="critical",
                ),
            ],
        )
        assert status == "B_FAILED"

    def test_success_with_no_critical_columns(self) -> None:
        from sema.engine.stage_utils import determine_b_status
        status = determine_b_status(
            raw_coverage=StageBCoverage(classified=5, total=5, pct=1.0),
            critical_coverage=StageBCoverage(classified=0, total=0, pct=1.0),
            unresolved=[],
        )
        assert status == "B_SUCCESS"

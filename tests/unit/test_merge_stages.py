"""Tests for merge step and pipeline integration (Section 4).

Covers: _merge_stage_outputs() ownership rules, B_PARTIAL handling,
StageStatus tracking, no VOCABULARY_MATCH from L2.
"""

from __future__ import annotations

from typing import Any

import pytest

from sema.models.assertions import AssertionPredicate
from sema.models.stages import (
    StageAResult,
    StageBBatchResult,
    StageBColumnResult,
    StageBCoverage,
    StageBResult,
    StageCResult,
    StageStatus,
    UnresolvedColumn,
)

pytestmark = pytest.mark.unit

TABLE_REF = "unity://catalog.schema.data_mutations"

STAGE_A = StageAResult(
    primary_entity="Somatic Mutation",
    grain_hypothesis="one row per variant call per sample",
    synonyms=["mutation call", "variant"],
    secondary_entity_hints=["gene", "protein change"],
    ambiguity_flags=[],
    confidence=0.88,
)

STAGE_A_NO_SYNONYMS = StageAResult(
    primary_entity="Somatic Mutation",
    grain_hypothesis="one row per variant call per sample",
    confidence=0.88,
)

STAGE_A_AMBIGUOUS = StageAResult(
    primary_entity="Patient Record",
    grain_hypothesis="one row per patient",
    synonyms=["patient data"],
    secondary_entity_hints=[],
    ambiguity_flags=["mixed granularity"],
    confidence=0.55,
)


def _col(name: str, **kw: Any) -> StageBColumnResult:
    defaults: dict[str, Any] = {
        "column": name,
        "canonical_property_label": name.replace("_", " ").title(),
        "semantic_type": "identifier",
        "candidate_vocab_families": [],
        "entity_role": None,
        "needs_stage_c": False,
        "ambiguity_notes": [],
        "evidence": [],
    }
    defaults.update(kw)
    return StageBColumnResult(**defaults)


def _stage_b_success(columns: list[StageBColumnResult]) -> StageBResult:
    return StageBResult(
        status="B_SUCCESS",
        batch_results=[StageBBatchResult(columns=columns)],
        raw_coverage=StageBCoverage(
            classified=len(columns), total=len(columns), pct=1.0,
        ),
        critical_coverage=StageBCoverage(
            classified=0, total=0, pct=1.0,
        ),
    )


# -- 4.1 StageStatus model ------------------------------------------------

class TestStageStatusSchema:
    def test_successful_full_pipeline(self) -> None:
        s = StageStatus(
            stage_a="success",
            stage_b_status="success",
            stage_b_raw_coverage=StageBCoverage(
                classified=5, total=5, pct=1.0,
            ),
            stage_b_critical_coverage=StageBCoverage(
                classified=2, total=2, pct=1.0,
            ),
            stage_c_triggered=True,
            stage_c_columns_requested=2,
            stage_c_columns_succeeded=2,
        )
        assert s.stage_a == "success"
        assert s.partial_output is False

    def test_partial_b_status(self) -> None:
        s = StageStatus(
            stage_a="success",
            stage_b_status="partial",
            stage_b_raw_coverage=StageBCoverage(
                classified=8, total=10, pct=0.8,
            ),
            stage_b_critical_coverage=StageBCoverage(
                classified=2, total=2, pct=1.0,
            ),
            stage_b_unresolved_columns=[
                UnresolvedColumn(
                    column="bad1", reason="execution_failure",
                    tier="peripheral",
                ),
            ],
            partial_output=True,
        )
        assert s.stage_b_status == "partial"
        assert len(s.stage_b_unresolved_columns) == 1

    def test_recovery_metrics(self) -> None:
        s = StageStatus(
            stage_a="success",
            stage_b_status="success",
            stage_b_raw_coverage=StageBCoverage(
                classified=5, total=5, pct=1.0,
            ),
            stage_b_critical_coverage=StageBCoverage(
                classified=0, total=0, pct=1.0,
            ),
            stage_b_retries_used=1,
            stage_b_splits_used=1,
        )
        assert s.stage_b_retries_used == 1
        assert s.stage_b_splits_used == 1


# -- 4.2–4.3 Merge function and ownership rules ---------------------------

class TestMergeStageOutputs:
    def test_emits_entity_name_from_a(self) -> None:
        from sema.engine.stage_utils import merge_stage_outputs
        b = _stage_b_success([
            _col("patient_id"), _col("Hugo_Symbol"),
        ])
        assertions = merge_stage_outputs(
            TABLE_REF, STAGE_A, b, run_id="test",
        )
        entity_a = [
            a for a in assertions
            if a.predicate == AssertionPredicate.HAS_ENTITY_NAME
        ]
        assert len(entity_a) == 1
        assert entity_a[0].payload["value"] == "Somatic Mutation"

    def test_emits_property_name_from_b(self) -> None:
        from sema.engine.stage_utils import merge_stage_outputs
        b = _stage_b_success([
            _col("Hugo_Symbol", canonical_property_label="Gene Symbol"),
        ])
        assertions = merge_stage_outputs(
            TABLE_REF, STAGE_A, b, run_id="test",
        )
        prop_a = [
            a for a in assertions
            if a.predicate == AssertionPredicate.HAS_PROPERTY_NAME
        ]
        assert len(prop_a) == 1
        assert prop_a[0].payload["value"] == "Gene Symbol"

    def test_emits_semantic_type_from_b(self) -> None:
        from sema.engine.stage_utils import merge_stage_outputs
        b = _stage_b_success([
            _col("Hugo_Symbol", semantic_type="gene_identifier"),
        ])
        assertions = merge_stage_outputs(
            TABLE_REF, STAGE_A, b, run_id="test",
        )
        type_a = [
            a for a in assertions
            if a.predicate == AssertionPredicate.HAS_SEMANTIC_TYPE
        ]
        assert len(type_a) == 1
        assert type_a[0].payload["value"] == "gene_identifier"

    def test_no_vocabulary_match_emitted(self) -> None:
        from sema.engine.stage_utils import merge_stage_outputs
        b = _stage_b_success([
            _col("Hugo_Symbol", candidate_vocab_families=["gene namespace"]),
        ])
        assertions = merge_stage_outputs(
            TABLE_REF, STAGE_A, b, run_id="test",
        )
        vocab_a = [
            a for a in assertions
            if a.predicate == AssertionPredicate.VOCABULARY_MATCH
        ]
        assert len(vocab_a) == 0

    def test_decoded_values_from_c_only(self) -> None:
        from sema.engine.stage_utils import merge_stage_outputs
        b = _stage_b_success([
            _col("gender", semantic_type="categorical", needs_stage_c=True),
        ])
        c_results = {
            "gender": StageCResult(
                column="gender",
                decoded_categories=[
                    {"raw": "M", "label": "Male"},
                    {"raw": "F", "label": "Female"},
                ],
                uncertainty=0.1,
            ),
        }
        assertions = merge_stage_outputs(
            TABLE_REF, STAGE_A, b, c_results=c_results, run_id="test",
        )
        decoded = [
            a for a in assertions
            if a.predicate == AssertionPredicate.HAS_DECODED_VALUE
        ]
        assert len(decoded) == 2
        raw_vals = {a.payload["raw"] for a in decoded}
        assert "M" in raw_vals
        assert "F" in raw_vals

    def test_no_decoded_values_without_c(self) -> None:
        from sema.engine.stage_utils import merge_stage_outputs
        b = _stage_b_success([
            _col("patient_id"),
        ])
        assertions = merge_stage_outputs(
            TABLE_REF, STAGE_A, b, run_id="test",
        )
        decoded = [
            a for a in assertions
            if a.predicate == AssertionPredicate.HAS_DECODED_VALUE
        ]
        assert len(decoded) == 0

    def test_emits_entity_aliases_from_a(self) -> None:
        from sema.engine.stage_utils import merge_stage_outputs
        b = _stage_b_success([_col("patient_id")])
        assertions = merge_stage_outputs(
            TABLE_REF, STAGE_A, b, run_id="test",
        )
        aliases = [
            a for a in assertions
            if (a.predicate == AssertionPredicate.HAS_ALIAS
                and a.subject_ref == TABLE_REF)
        ]
        assert len(aliases) == 2
        vals = [a.payload["value"] for a in aliases]
        assert "mutation call" in vals
        assert "variant" in vals
        assert aliases[0].payload["is_preferred"] is True

    def test_no_entity_aliases_when_a_has_none(self) -> None:
        from sema.engine.stage_utils import merge_stage_outputs
        b = _stage_b_success([_col("patient_id")])
        assertions = merge_stage_outputs(
            TABLE_REF, STAGE_A_NO_SYNONYMS, b, run_id="test",
        )
        aliases = [
            a for a in assertions
            if (a.predicate == AssertionPredicate.HAS_ALIAS
                and a.subject_ref == TABLE_REF)
        ]
        assert len(aliases) == 0

    def test_emits_property_aliases_from_b(self) -> None:
        from sema.engine.stage_utils import merge_stage_outputs
        b = _stage_b_success([
            _col("Hugo_Symbol", synonyms=["HGNC symbol", "gene name"]),
        ])
        assertions = merge_stage_outputs(
            TABLE_REF, STAGE_A_NO_SYNONYMS, b, run_id="test",
        )
        col_aliases = [
            a for a in assertions
            if (a.predicate == AssertionPredicate.HAS_ALIAS
                and a.subject_ref != TABLE_REF)
        ]
        assert len(col_aliases) == 2
        vals = [a.payload["value"] for a in col_aliases]
        assert "HGNC symbol" in vals

    def test_entity_aliases_dropped_on_grain_correction(self) -> None:
        from sema.engine.stage_utils import merge_stage_outputs
        b = StageBResult(
            status="B_SUCCESS",
            batch_results=[StageBBatchResult(
                columns=[_col("sample_id")],
                grain_correction="actually per-sample",
            )],
            raw_coverage=StageBCoverage(
                classified=1, total=1, pct=1.0,
            ),
            critical_coverage=StageBCoverage(
                classified=0, total=0, pct=1.0,
            ),
        )
        assertions = merge_stage_outputs(
            TABLE_REF, STAGE_A_AMBIGUOUS, b, run_id="test",
        )
        entity_aliases = [
            a for a in assertions
            if (a.predicate == AssertionPredicate.HAS_ALIAS
                and a.subject_ref == TABLE_REF)
        ]
        # A's aliases should be dropped when B corrects
        assert len(entity_aliases) == 0

    def test_b_grain_correction_updates_grain_in_payload(self) -> None:
        from sema.engine.stage_utils import merge_stage_outputs
        b = StageBResult(
            status="B_SUCCESS",
            batch_results=[StageBBatchResult(
                columns=[_col("sample_id"), _col("mutation_id")],
                grain_correction="one row per sample, not per patient",
            )],
            raw_coverage=StageBCoverage(
                classified=2, total=2, pct=1.0,
            ),
            critical_coverage=StageBCoverage(
                classified=0, total=0, pct=1.0,
            ),
        )
        assertions = merge_stage_outputs(
            TABLE_REF, STAGE_A_AMBIGUOUS, b, run_id="test",
        )
        entity_a = [
            a for a in assertions
            if a.predicate == AssertionPredicate.HAS_ENTITY_NAME
        ]
        assert len(entity_a) == 1
        # Grain in payload should reflect B's correction
        assert entity_a[0].payload["grain"] == (
            "one row per sample, not per patient"
        )


# -- 4.4 Merge for B_PARTIAL ----------------------------------------------

class TestMergePartial:
    def test_partial_only_emits_classified_columns(self) -> None:
        from sema.engine.stage_utils import merge_stage_outputs
        b = StageBResult(
            status="B_PARTIAL",
            batch_results=[StageBBatchResult(
                columns=[_col("patient_id"), _col("Hugo_Symbol")],
            )],
            raw_coverage=StageBCoverage(
                classified=2, total=3, pct=0.67,
            ),
            critical_coverage=StageBCoverage(
                classified=1, total=1, pct=1.0,
            ),
            unresolved_columns=[
                UnresolvedColumn(
                    column="bad_col",
                    reason="execution_failure",
                    tier="peripheral",
                ),
            ],
        )
        assertions = merge_stage_outputs(
            TABLE_REF, STAGE_A, b, run_id="test",
        )
        prop_cols = {
            a.subject_ref.split(".")[-1]
            for a in assertions
            if a.predicate == AssertionPredicate.HAS_PROPERTY_NAME
        }
        assert "patient_id" in prop_cols
        assert "Hugo_Symbol" in prop_cols
        assert "bad_col" not in prop_cols


# -- 4.10 Full A→B→merge integration test ---------------------------------

class TestFullMergeIntegration:
    def test_correct_assertion_set(self) -> None:
        from sema.engine.stage_utils import merge_stage_outputs
        b = _stage_b_success([
            _col("patient_id", semantic_type="identifier",
                 entity_role="foreign_key"),
            _col("Hugo_Symbol", semantic_type="gene_identifier",
                 canonical_property_label="Gene Symbol",
                 candidate_vocab_families=["gene symbol namespace"]),
            _col("Variant_Classification", semantic_type="categorical",
                 canonical_property_label="Variant Classification",
                 needs_stage_c=True),
        ])
        assertions = merge_stage_outputs(
            TABLE_REF, STAGE_A, b, run_id="test",
        )
        # 1 entity + 2 entity aliases + 3 props + 3 sem types = 9
        assert len(assertions) == 9
        preds = [a.predicate for a in assertions]
        assert preds.count(AssertionPredicate.HAS_ENTITY_NAME) == 1
        assert preds.count(AssertionPredicate.HAS_ALIAS) == 2
        assert preds.count(AssertionPredicate.HAS_PROPERTY_NAME) == 3
        assert preds.count(AssertionPredicate.HAS_SEMANTIC_TYPE) == 3
        assert preds.count(AssertionPredicate.VOCABULARY_MATCH) == 0

    def test_all_assertions_have_correct_metadata(self) -> None:
        from sema.engine.stage_utils import merge_stage_outputs
        b = _stage_b_success([_col("patient_id")])
        assertions = merge_stage_outputs(
            TABLE_REF, STAGE_A, b, run_id="merge-test",
        )
        for a in assertions:
            assert a.source == "llm_interpretation"
            assert a.run_id == "merge-test"


# -- 4.11 No VOCABULARY_MATCH from L2 path --------------------------------

class TestNoVocabMatch:
    def test_vocab_families_not_materialized(self) -> None:
        from sema.engine.stage_utils import merge_stage_outputs
        b = _stage_b_success([
            _col("dx_code",
                 candidate_vocab_families=["diagnosis coding system"]),
            _col("Hugo_Symbol",
                 candidate_vocab_families=["gene symbol namespace"]),
        ])
        assertions = merge_stage_outputs(
            TABLE_REF, STAGE_A, b, run_id="test",
        )
        vocab = [
            a for a in assertions
            if a.predicate == AssertionPredicate.VOCABULARY_MATCH
        ]
        assert len(vocab) == 0


# -- 4.7 Enriched VocabColumnContext from staged output --------------------

class TestEnrichedVocabContext:
    def test_enrichment_version_set_to_one(self) -> None:
        from sema.engine.stage_utils import build_enriched_vocab_context
        col = _col(
            "Hugo_Symbol",
            semantic_type="gene_identifier",
            candidate_vocab_families=["gene symbol namespace"],
            entity_role="secondary",
        )
        ctx = build_enriched_vocab_context(col, STAGE_A, "data_mutations")
        assert ctx._enrichment_version == 1

    def test_new_fields_accessible(self) -> None:
        from sema.engine.stage_utils import build_enriched_vocab_context
        col = _col(
            "Hugo_Symbol",
            canonical_property_label="Gene Symbol",
            semantic_type="gene_identifier",
            candidate_vocab_families=["gene symbol namespace"],
            entity_role="secondary",
            ambiguity_notes=["could be alias or official symbol"],
        )
        ctx = build_enriched_vocab_context(col, STAGE_A, "data_mutations")
        assert ctx.candidate_vocab_families == ["gene symbol namespace"]
        assert ctx.grain_hypothesis == "one row per variant call per sample"
        assert ctx.entity_role == "secondary"
        assert ctx.ambiguity_notes == ["could be alias or official symbol"]

    def test_legacy_fields_populated(self) -> None:
        from sema.engine.stage_utils import build_enriched_vocab_context
        col = _col("Hugo_Symbol", canonical_property_label="Gene Symbol",
                    semantic_type="gene_identifier")
        ctx = build_enriched_vocab_context(col, STAGE_A, "data_mutations")
        assert ctx.column_name == "Hugo_Symbol"
        assert ctx.entity_name == "Somatic Mutation"
        assert ctx.semantic_type == "gene_identifier"
        assert ctx.property_name == "Gene Symbol"
        assert ctx.table_name == "data_mutations"

    def test_domain_context_passed_through(self) -> None:
        from sema.engine.stage_utils import build_enriched_vocab_context
        from sema.models.domain import DomainContext
        dc = DomainContext(declared_domain="healthcare", domain_source="user")
        col = _col("Hugo_Symbol")
        ctx = build_enriched_vocab_context(
            col, STAGE_A, "data_mutations", domain_context=dc,
        )
        assert ctx.domain_context.declared_domain == "healthcare"


# -- 4.13 B_PARTIAL excludes unresolved from VocabColumnContext ------------

class TestPartialExcludesUnresolved:
    def test_unresolved_columns_excluded_from_vocab_contexts(self) -> None:
        from sema.engine.stage_utils import build_enriched_vocab_context
        b = StageBResult(
            status="B_PARTIAL",
            batch_results=[StageBBatchResult(
                columns=[_col("patient_id"), _col("Hugo_Symbol")],
            )],
            raw_coverage=StageBCoverage(
                classified=2, total=3, pct=0.67,
            ),
            critical_coverage=StageBCoverage(
                classified=1, total=1, pct=1.0,
            ),
            unresolved_columns=[
                UnresolvedColumn(
                    column="bad_col",
                    reason="execution_failure",
                    tier="peripheral",
                ),
            ],
        )
        unresolved_names = {u.column for u in b.unresolved_columns}
        contexts = []
        for batch in b.batch_results:
            for col in batch.columns:
                if col.column not in unresolved_names:
                    contexts.append(
                        build_enriched_vocab_context(
                            col, STAGE_A, "data_mutations",
                        )
                    )
        ctx_names = {c.column_name for c in contexts}
        assert "patient_id" in ctx_names
        assert "Hugo_Symbol" in ctx_names
        assert "bad_col" not in ctx_names

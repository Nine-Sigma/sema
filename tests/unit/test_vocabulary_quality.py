import json
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.unit

from sema.engine.vocabulary import (
    VocabColumnContext,
    VocabularyEngine,
)
from sema.models.assertions import AssertionPredicate


class TestHierarchyGating:
    def test_numeric_column_skipped(self):
        ctx = VocabColumnContext(semantic_type="numeric")
        engine = VocabularyEngine(run_id="test-run")
        assertions = engine.process_column(
            "unity://tbl.col", ["2", "25", "3", "30"], context=ctx,
        )
        parent_of = [a for a in assertions if a.predicate == AssertionPredicate.PARENT_OF]
        assert len(parent_of) == 0

    def test_temporal_column_skipped(self):
        ctx = VocabColumnContext(semantic_type="temporal")
        engine = VocabularyEngine(run_id="test-run")
        assertions = engine.process_column(
            "unity://tbl.col", ["2020", "2020-01"], context=ctx,
        )
        parent_of = [a for a in assertions if a.predicate == AssertionPredicate.PARENT_OF]
        assert len(parent_of) == 0

    def test_categorical_icd10_runs_hierarchy(self):
        ctx = VocabColumnContext(semantic_type="categorical")
        engine = VocabularyEngine(run_id="test-run")
        assertions = engine.process_column(
            "unity://tbl.col", ["C18", "C18.0", "C18.1"], context=ctx,
        )
        parent_of = [a for a in assertions if a.predicate == AssertionPredicate.PARENT_OF]
        assert len(parent_of) >= 1

    def test_categorical_unknown_vocab_skipped(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content=json.dumps({"vocabulary": "CustomSystem", "confidence": 0.6})
        )
        ctx = VocabColumnContext(semantic_type="categorical")
        engine = VocabularyEngine(llm=mock_llm, run_id="test-run")
        assertions = engine.process_column(
            "unity://tbl.col", ["AA", "AAB", "AB"], context=ctx,
        )
        parent_of = [a for a in assertions if a.predicate == AssertionPredicate.PARENT_OF]
        assert len(parent_of) == 0

    def test_no_context_skips_hierarchy(self):
        engine = VocabularyEngine(run_id="test-run")
        assertions = engine.process_column(
            "unity://tbl.col", ["C18", "C18.0", "C18.1"],
        )
        parent_of = [a for a in assertions if a.predicate == AssertionPredicate.PARENT_OF]
        assert len(parent_of) == 0

    def test_numeric_regression_no_false_edges(self):
        ctx = VocabColumnContext(semantic_type="numeric")
        engine = VocabularyEngine(run_id="test-run")
        values = ["3", "30", "8", "80", "180"]
        assertions = engine.process_column(
            "unity://tbl.box_size", values, context=ctx,
        )
        parent_of = [a for a in assertions if a.predicate == AssertionPredicate.PARENT_OF]
        assert len(parent_of) == 0


class TestContextEnrichment:
    def test_process_column_backward_compat(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content=json.dumps({"vocabulary": "OncoTree", "confidence": 0.7})
        )
        engine = VocabularyEngine(llm=mock_llm, run_id="test-run")
        assertions = engine.process_column(
            "unity://cdm.clinical.tbl.col", ["CRC", "BRCA"]
        )
        vocab = [a for a in assertions if a.predicate == AssertionPredicate.VOCABULARY_MATCH]
        assert len(vocab) == 1

    def test_prompt_includes_entity_name(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content=json.dumps({"vocabulary": "OncoTree", "confidence": 0.7})
        )
        ctx = VocabColumnContext(
            column_name="dx_type_cd",
            table_name="diagnoses",
            entity_name="Diagnosis",
        )
        engine = VocabularyEngine(llm=mock_llm, run_id="test-run")
        engine.detect_vocabulary("unity://cdm.tbl.col", ["CRC", "BRCA"], context=ctx)
        prompt_sent = mock_llm.invoke.call_args[0][0]
        assert "Diagnosis" in prompt_sent

    def test_prompt_includes_vocab_guess(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content=json.dumps({"vocabulary": "OncoTree", "confidence": 0.7})
        )
        ctx = VocabColumnContext(vocabulary_guess="OncoTree")
        engine = VocabularyEngine(llm=mock_llm, run_id="test-run")
        engine.detect_vocabulary("unity://cdm.tbl.col", ["CRC", "BRCA"], context=ctx)
        prompt_sent = mock_llm.invoke.call_args[0][0]
        assert "OncoTree" in prompt_sent

    def test_agreement_boost(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content=json.dumps({"vocabulary": "OncoTree", "confidence": 0.6})
        )
        ctx = VocabColumnContext(vocabulary_guess="OncoTree")
        engine = VocabularyEngine(llm=mock_llm, run_id="test-run")
        assertions = engine.process_column(
            "unity://cdm.tbl.col", ["CRC", "BRCA"], context=ctx,
        )
        vocab = [a for a in assertions if a.predicate == AssertionPredicate.VOCABULARY_MATCH]
        assert len(vocab) == 1
        assert vocab[0].confidence >= 0.7

    def test_agreement_boost_normalized(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content=json.dumps({"vocabulary": "AJCC Staging", "confidence": 0.6})
        )
        ctx = VocabColumnContext(vocabulary_guess="AJCC 8th Edition")
        engine = VocabularyEngine(llm=mock_llm, run_id="test-run")
        assertions = engine.process_column(
            "unity://cdm.tbl.col", ["Stage I", "Stage II"], context=ctx,
        )
        vocab = [a for a in assertions if a.predicate == AssertionPredicate.VOCABULARY_MATCH]
        assert len(vocab) == 1
        assert vocab[0].confidence >= 0.7

    def test_no_agreement_no_boost(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content=json.dumps({"vocabulary": "OncoTree", "confidence": 0.6})
        )
        ctx = VocabColumnContext(vocabulary_guess="ICD-10")
        engine = VocabularyEngine(llm=mock_llm, run_id="test-run")
        assertions = engine.process_column(
            "unity://cdm.tbl.col", ["CRC", "BRCA"], context=ctx,
        )
        vocab = [a for a in assertions if a.predicate == AssertionPredicate.VOCABULARY_MATCH]
        assert len(vocab) == 1
        assert vocab[0].confidence == 0.5


class TestConfidenceCalibration:
    def test_llm_confidence_not_self_reported(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content=json.dumps({"vocabulary": "OncoTree", "confidence": 0.95})
        )
        engine = VocabularyEngine(llm=mock_llm, run_id="test-run")
        assertions = engine.detect_vocabulary(
            "unity://tbl.col", ["CRC", "BRCA"],
        )
        vocab = [a for a in assertions if a.predicate == AssertionPredicate.VOCABULARY_MATCH]
        assert len(vocab) == 1
        assert vocab[0].confidence != 0.95

    def test_agreement_raises_confidence(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content=json.dumps({"vocabulary": "OncoTree", "confidence": 0.9})
        )
        ctx = VocabColumnContext(vocabulary_guess="OncoTree")
        engine = VocabularyEngine(llm=mock_llm, run_id="test-run")
        assertions = engine.detect_vocabulary(
            "unity://tbl.col", ["CRC", "BRCA"], context=ctx,
        )
        vocab = assertions[0]
        no_ctx_llm = MagicMock()
        no_ctx_llm.invoke.return_value = MagicMock(
            content=json.dumps({"vocabulary": "OncoTree", "confidence": 0.9})
        )
        engine2 = VocabularyEngine(llm=no_ctx_llm, run_id="test-run")
        assertions2 = engine2.detect_vocabulary(
            "unity://tbl.col", ["CRC", "BRCA"],
        )
        no_ctx_vocab = assertions2[0]
        assert vocab.confidence > no_ctx_vocab.confidence

    def test_column_name_heuristic(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content=json.dumps({"vocabulary": "OncoTree", "confidence": 0.9})
        )
        ctx_with_code = VocabColumnContext(column_name="dx_code")
        ctx_no_code = VocabColumnContext(column_name="notes")
        engine = VocabularyEngine(llm=mock_llm, run_id="test-run")
        a1 = engine.detect_vocabulary("unity://tbl.col", ["CRC", "BRCA"], context=ctx_with_code)
        mock_llm.invoke.return_value = MagicMock(
            content=json.dumps({"vocabulary": "OncoTree", "confidence": 0.9})
        )
        a2 = engine.detect_vocabulary("unity://tbl.col", ["CRC", "BRCA"], context=ctx_no_code)
        assert a1[0].confidence > a2[0].confidence

    def test_pattern_match_confidence_unchanged(self):
        engine = VocabularyEngine(run_id="test-run")
        assertions = engine.detect_vocabulary(
            "unity://tbl.col", ["C18.0", "C18.1", "C34.1"],
        )
        vocab = [a for a in assertions if a.predicate == AssertionPredicate.VOCABULARY_MATCH]
        assert vocab[0].confidence >= 0.9
        assert vocab[0].source == "pattern_match"

    def test_synonym_confidence_with_vocab(self):
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [
            MagicMock(content=json.dumps({"vocabulary": "OncoTree", "confidence": 0.7})),
            MagicMock(content=json.dumps({
                "synonyms": [{"term": "Colorectal", "synonyms": ["CRC"]}],
            })),
        ]
        ctx = VocabColumnContext(semantic_type="categorical")
        engine = VocabularyEngine(llm=mock_llm, run_id="test-run")
        assertions = engine.process_column(
            "unity://tbl.col", ["CRC", "BRCA"],
            [{"raw": "CRC", "label": "Colorectal"}],
            context=ctx,
        )
        alias = [a for a in assertions if a.predicate == AssertionPredicate.HAS_ALIAS]
        assert len(alias) >= 1
        assert alias[0].confidence == 0.75

    def test_synonym_confidence_without_vocab(self):
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [
            MagicMock(content=json.dumps({"vocabulary": None, "confidence": 0.0})),
            MagicMock(content=json.dumps({
                "synonyms": [{"term": "Active", "synonyms": ["enabled"]}],
            })),
        ]
        engine = VocabularyEngine(llm=mock_llm, run_id="test-run")
        assertions = engine.process_column(
            "unity://tbl.col", ["active"], [{"raw": "active", "label": "Active"}],
        )
        alias = [a for a in assertions if a.predicate == AssertionPredicate.HAS_ALIAS]
        assert len(alias) >= 1
        assert alias[0].confidence == 0.65

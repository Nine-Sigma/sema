import json
import pytest
from unittest.mock import MagicMock
from datetime import datetime, timezone

pytestmark = pytest.mark.unit

from sema.engine.vocabulary import (
    VocabularyEngine,
    detect_vocabulary_pattern,
    infer_hierarchy,
)
from sema.models.assertions import Assertion, AssertionPredicate


class TestPatternDetection:
    def test_icd10_format(self):
        values = ["C18.0", "C18.1", "C34.1", "C50.9"]
        result = detect_vocabulary_pattern(values)
        assert result is not None
        assert result["vocabulary"] == "ICD-10"
        assert result["confidence"] >= 0.9

    def test_icd10_with_letters(self):
        values = ["E11.9", "I10", "J45.0"]
        result = detect_vocabulary_pattern(values)
        assert result is not None
        assert result["vocabulary"] == "ICD-10"

    def test_ajcc_staging(self):
        values = ["Stage I", "Stage IA", "Stage II", "Stage IIIA", "Stage IV"]
        result = detect_vocabulary_pattern(values)
        assert result is not None
        assert result["vocabulary"] == "AJCC Staging"
        assert result["confidence"] >= 0.9

    def test_ajcc_staging_lowercase(self):
        values = ["stage i", "stage ii", "stage iii"]
        result = detect_vocabulary_pattern(values)
        assert result is not None
        assert result["vocabulary"] == "AJCC Staging"

    def test_tnm_codes(self):
        values = ["T1N0M0", "T2N1M0", "T3N2M1"]
        result = detect_vocabulary_pattern(values)
        assert result is not None
        assert "TNM" in result["vocabulary"]

    def test_no_pattern_for_generic_strings(self):
        values = ["active", "inactive", "pending", "completed"]
        result = detect_vocabulary_pattern(values)
        assert result is None

    def test_no_pattern_for_numbers(self):
        values = ["1", "2", "3", "4", "5"]
        result = detect_vocabulary_pattern(values)
        assert result is None

    def test_empty_values(self):
        result = detect_vocabulary_pattern([])
        assert result is None


class TestHierarchyInference:
    def test_stage_hierarchy(self):
        values = ["Stage I", "Stage IA", "Stage IA1", "Stage IB", "Stage II", "Stage III", "Stage IIIA", "Stage IIIB"]
        hierarchy = infer_hierarchy(values)
        # Stage I should be parent of Stage IA, Stage IB
        assert ("Stage I", "Stage IA") in hierarchy
        assert ("Stage I", "Stage IB") in hierarchy
        # Stage IA should be parent of Stage IA1
        assert ("Stage IA", "Stage IA1") in hierarchy
        # Stage III should be parent of Stage IIIA, Stage IIIB
        assert ("Stage III", "Stage IIIA") in hierarchy
        assert ("Stage III", "Stage IIIB") in hierarchy

    def test_no_hierarchy_for_flat_values(self):
        values = ["Male", "Female", "Unknown"]
        hierarchy = infer_hierarchy(values)
        assert len(hierarchy) == 0

    def test_prefix_based_hierarchy(self):
        values = ["C18", "C18.0", "C18.1", "C18.2", "C19"]
        hierarchy = infer_hierarchy(values)
        assert ("C18", "C18.0") in hierarchy
        assert ("C18", "C18.1") in hierarchy
        assert ("C18", "C18.2") in hierarchy

    def test_empty_values(self):
        hierarchy = infer_hierarchy([])
        assert len(hierarchy) == 0


class TestLLMVocabularyDetection:
    def test_llm_fallback_for_ambiguous_values(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content=json.dumps({"vocabulary": "OncoTree", "confidence": 0.7})
        )
        engine = VocabularyEngine(llm=mock_llm, run_id="test-run")
        values = ["CRC", "BRCA", "NSCLC", "MEL"]
        assertions = engine.detect_vocabulary("unity://cdm.clinical.tbl.col", values)
        vocab_assertions = [a for a in assertions if a.predicate == AssertionPredicate.VOCABULARY_MATCH]
        assert len(vocab_assertions) == 1
        assert vocab_assertions[0].payload["value"] == "OncoTree"
        assert vocab_assertions[0].source == "llm_interpretation"

    def test_pattern_match_takes_priority(self):
        mock_llm = MagicMock()
        engine = VocabularyEngine(llm=mock_llm, run_id="test-run")
        values = ["C18.0", "C18.1", "C34.1"]
        assertions = engine.detect_vocabulary("unity://cdm.clinical.tbl.col", values)
        vocab_assertions = [a for a in assertions if a.predicate == AssertionPredicate.VOCABULARY_MATCH]
        assert len(vocab_assertions) == 1
        assert vocab_assertions[0].source == "pattern_match"
        assert vocab_assertions[0].confidence >= 0.9
        mock_llm.invoke.assert_not_called()

    def test_llm_failure_returns_empty(self):
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("LLM failed")
        engine = VocabularyEngine(llm=mock_llm, run_id="test-run")
        values = ["CRC", "BRCA"]
        assertions = engine.detect_vocabulary("unity://cdm.clinical.tbl.col", values)
        vocab_assertions = [a for a in assertions if a.predicate == AssertionPredicate.VOCABULARY_MATCH]
        assert len(vocab_assertions) == 0


class TestSynonymExpansion:
    def test_synonym_assertions_for_decoded_terms(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content=json.dumps({
                "synonyms": [
                    {"term": "Colorectal Cancer", "synonyms": ["colon cancer", "bowel cancer", "CRC"]},
                    {"term": "Breast Cancer", "synonyms": ["breast carcinoma", "BRCA"]},
                ]
            })
        )
        engine = VocabularyEngine(llm=mock_llm, run_id="test-run")
        terms = [{"code": "CRC", "label": "Colorectal Cancer"}, {"code": "BRCA", "label": "Breast Cancer"}]
        assertions = engine.expand_synonyms("unity://cdm.clinical.tbl.col", terms)
        syn_assertions = [a for a in assertions if a.predicate == AssertionPredicate.HAS_SYNONYM]
        assert len(syn_assertions) >= 4  # at least 2 per term


class TestFullVocabularyPipeline:
    def test_process_column_emits_all_assertion_types(self):
        mock_llm = MagicMock()
        # vocab detection returns OncoTree
        mock_llm.invoke.side_effect = [
            MagicMock(content=json.dumps({"vocabulary": "OncoTree", "confidence": 0.7})),
            MagicMock(content=json.dumps({
                "synonyms": [
                    {"term": "Colorectal Cancer", "synonyms": ["colon cancer"]},
                ]
            })),
        ]
        engine = VocabularyEngine(llm=mock_llm, run_id="test-run")

        values = ["CRC", "BRCA"]
        decoded = [{"raw": "CRC", "label": "Colorectal Cancer"}, {"raw": "BRCA", "label": "Breast Cancer"}]
        assertions = engine.process_column("unity://cdm.clinical.tbl.col", values, decoded)

        predicates = {a.predicate for a in assertions}
        assert AssertionPredicate.VOCABULARY_MATCH in predicates
        assert AssertionPredicate.HAS_SYNONYM in predicates

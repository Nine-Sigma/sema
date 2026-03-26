"""Tests for engines using LLMClient (tasks 4.1, 4.2, 4.3)."""
import json
import pytest
from unittest.mock import MagicMock

pytestmark = pytest.mark.unit

from sema.engine.semantic import (
    SemanticEngine,
    TableInterpretation,
    PropertyInterpretation,
)
from sema.engine.vocabulary import VocabularyEngine
from sema.llm_client import (
    LLMClient,
    LLMStageError,
    VocabularyDetection,
    SynonymExpansion,
)
from sema.models.assertions import AssertionPredicate


# ---------------------------------------------------------------------------
# SemanticEngine with LLMClient (Task 4.1)
# ---------------------------------------------------------------------------

class TestSemanticEngineWithLLMClient:
    def test_produces_correct_assertions(self):
        mock_client = MagicMock(spec=LLMClient)
        mock_client.invoke.return_value = TableInterpretation(
            entity_name="Cancer Diagnosis",
            entity_description="Diagnosis records",
            synonyms=["dx"],
            properties=[
                PropertyInterpretation(
                    column="dx_type_cd",
                    name="Diagnosis Type",
                    semantic_type="categorical",
                    confidence=0.9,
                    synonyms=["cancer type"],
                    decoded_values=[{"raw": "CRC", "label": "Colorectal Cancer"}],
                    vocabulary_guess="OncoTree",
                ),
            ],
        )

        engine = SemanticEngine(llm_client=mock_client, run_id="test-run")
        sample = {
            "table_ref": "unity://cdm.clinical.tbl",
            "table_name": "tbl",
            "columns": [{"name": "dx_type_cd", "data_type": "STRING"}],
            "sample_rows": [],
            "comment": None,
        }

        assertions = engine.interpret_table(sample)

        entity = [a for a in assertions if a.predicate == AssertionPredicate.HAS_ENTITY_NAME]
        assert len(entity) == 1
        assert entity[0].payload["value"] == "Cancer Diagnosis"

        props = [a for a in assertions if a.predicate == AssertionPredicate.HAS_PROPERTY_NAME]
        assert len(props) == 1
        assert props[0].payload["value"] == "Diagnosis Type"

        decoded = [a for a in assertions if a.predicate == AssertionPredicate.HAS_DECODED_VALUE]
        assert len(decoded) == 1
        assert decoded[0].payload["raw"] == "CRC"

        vocab = [a for a in assertions if a.predicate == AssertionPredicate.VOCABULARY_MATCH]
        assert len(vocab) == 1
        assert vocab[0].payload["value"] == "OncoTree"

    def test_llm_stage_error_propagates(self):
        mock_client = MagicMock(spec=LLMClient)
        mock_client.invoke.side_effect = LLMStageError(
            table_ref="unity://cdm.clinical.tbl",
            stage_name="L2 semantic",
            step_errors=[("structured_output", ValueError("fail"))],
        )

        engine = SemanticEngine(llm_client=mock_client, run_id="test-run")
        sample = {
            "table_ref": "unity://cdm.clinical.tbl",
            "table_name": "tbl",
            "columns": [],
            "sample_rows": [],
            "comment": None,
        }

        with pytest.raises(LLMStageError) as exc_info:
            engine.interpret_table(sample)
        assert exc_info.value.stage_name == "L2 semantic"


# ---------------------------------------------------------------------------
# VocabularyEngine with LLMClient (Task 4.2)
# ---------------------------------------------------------------------------

class TestVocabularyEngineWithLLMClient:
    def test_detect_vocabulary_via_llm_client(self):
        mock_client = MagicMock(spec=LLMClient)
        mock_client.invoke.return_value = VocabularyDetection(
            vocabulary="OncoTree", confidence=0.8
        )

        engine = VocabularyEngine(llm_client=mock_client, run_id="test-run")
        assertions = engine.detect_vocabulary(
            "unity://cdm.clinical.tbl.col", ["CRC", "BRCA", "NSCLC"]
        )

        vocab = [a for a in assertions if a.predicate == AssertionPredicate.VOCABULARY_MATCH]
        assert len(vocab) == 1
        assert vocab[0].payload["value"] == "OncoTree"
        assert vocab[0].source == "llm_interpretation"

    def test_expand_synonyms_via_llm_client(self):
        mock_client = MagicMock(spec=LLMClient)
        mock_client.invoke.return_value = SynonymExpansion(
            synonyms=[
                {"term": "Colorectal Cancer", "synonyms": ["colon cancer", "CRC"]},
            ]
        )

        engine = VocabularyEngine(llm_client=mock_client, run_id="test-run")
        terms = [{"code": "CRC", "label": "Colorectal Cancer"}]
        assertions = engine.expand_synonyms("unity://cdm.clinical.tbl.col", terms)

        aliases = [a for a in assertions if a.predicate == AssertionPredicate.HAS_ALIAS]
        assert len(aliases) == 2
        values = {a.payload["value"] for a in aliases}
        assert "colon cancer" in values
        assert "CRC" in values
        # First alias should be preferred
        assert aliases[0].payload["is_preferred"] is True

    def test_llm_stage_error_propagates_from_detect(self):
        mock_client = MagicMock(spec=LLMClient)
        mock_client.invoke.side_effect = LLMStageError(
            table_ref="unity://cdm.clinical.tbl.col",
            stage_name="L3 vocabulary",
            step_errors=[("plain_invoke", ValueError("fail"))],
        )

        engine = VocabularyEngine(llm_client=mock_client, run_id="test-run")

        with pytest.raises(LLMStageError):
            engine.detect_vocabulary("unity://cdm.clinical.tbl.col", ["CRC"])

    def test_llm_stage_error_propagates_from_expand(self):
        mock_client = MagicMock(spec=LLMClient)
        mock_client.invoke.side_effect = LLMStageError(
            table_ref="col",
            stage_name="L3 vocabulary",
            step_errors=[("plain_invoke", ValueError("fail"))],
        )

        engine = VocabularyEngine(llm_client=mock_client, run_id="test-run")
        terms = [{"code": "X", "label": "Something"}]

        with pytest.raises(LLMStageError):
            engine.expand_synonyms("col", terms)

    def test_pattern_match_still_takes_priority(self):
        mock_client = MagicMock(spec=LLMClient)
        engine = VocabularyEngine(llm_client=mock_client, run_id="test-run")

        values = ["C18.0", "C18.1", "C34.1"]
        assertions = engine.detect_vocabulary("col", values)

        vocab = [a for a in assertions if a.predicate == AssertionPredicate.VOCABULARY_MATCH]
        assert len(vocab) == 1
        assert vocab[0].source == "pattern_match"
        mock_client.invoke.assert_not_called()


# ---------------------------------------------------------------------------
# Legitimate empty results (Task 4.3)
# ---------------------------------------------------------------------------

class TestLegitimateEmptyResults:
    def test_no_llm_no_pattern_returns_empty(self):
        engine = VocabularyEngine(run_id="test-run")
        assertions = engine.detect_vocabulary("col", ["active", "inactive"])
        assert assertions == []

    def test_no_terms_for_synonym_expansion(self):
        mock_client = MagicMock(spec=LLMClient)
        engine = VocabularyEngine(llm_client=mock_client, run_id="test-run")
        assertions = engine.expand_synonyms("col", [])
        assert assertions == []
        mock_client.invoke.assert_not_called()

    def test_empty_values_for_vocabulary(self):
        mock_client = MagicMock(spec=LLMClient)
        engine = VocabularyEngine(llm_client=mock_client, run_id="test-run")
        assertions = engine.detect_vocabulary("col", [])
        assert assertions == []
        mock_client.invoke.assert_not_called()

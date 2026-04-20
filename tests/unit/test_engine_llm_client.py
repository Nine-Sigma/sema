"""Tests for VocabularyEngine + LLMClient integration."""
import pytest
from unittest.mock import MagicMock

pytestmark = pytest.mark.unit

from sema.engine.vocabulary import VocabularyEngine
from sema.llm_client import (
    LLMClient,
    LLMStageError,
    VocabularyDetection,
    SynonymExpansion,
)
from sema.models.assertions import AssertionPredicate


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

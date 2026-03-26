import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

pytestmark = pytest.mark.unit

from sema.engine.semantic import (
    SemanticEngine,
    TableInterpretation,
    PropertyInterpretation,
    build_interpretation_prompt,
)
from sema.models.assertions import Assertion, AssertionPredicate

FIXTURES = Path(__file__).parent.parent / "fixtures"


@pytest.fixture
def sample_metadata():
    with open(FIXTURES / "sample_table_metadata.json") as f:
        return json.load(f)


@pytest.fixture
def expected_response():
    with open(FIXTURES / "expected_llm_response.json") as f:
        return json.load(f)


@pytest.fixture
def mock_llm(expected_response):
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content=json.dumps(expected_response))
    return llm


@pytest.fixture
def engine(mock_llm):
    return SemanticEngine(llm=mock_llm, run_id="test-run")


class TestPromptConstruction:
    def test_prompt_includes_table_name(self, sample_metadata):
        prompt = build_interpretation_prompt(sample_metadata)
        assert "cancer_diagnosis" in prompt

    def test_prompt_includes_comment(self, sample_metadata):
        prompt = build_interpretation_prompt(sample_metadata)
        assert "Cancer diagnosis records" in prompt

    def test_prompt_includes_column_names_and_types(self, sample_metadata):
        prompt = build_interpretation_prompt(sample_metadata)
        assert "dx_type_cd" in prompt
        assert "STRING" in prompt
        assert "DATE" in prompt

    def test_prompt_includes_top_values(self, sample_metadata):
        prompt = build_interpretation_prompt(sample_metadata)
        assert "CRC" in prompt
        assert "BRCA" in prompt

    def test_prompt_includes_sample_rows(self, sample_metadata):
        prompt = build_interpretation_prompt(sample_metadata)
        assert "P12345" in prompt or "Stage IIIA" in prompt


class TestResponseParsing:
    def test_parse_valid_response(self, expected_response):
        interp = TableInterpretation.model_validate(expected_response)
        assert interp.entity_name == "Cancer Diagnosis"
        assert len(interp.properties) == 4

    def test_property_has_semantic_type(self, expected_response):
        interp = TableInterpretation.model_validate(expected_response)
        dx_prop = next(p for p in interp.properties if p.column == "dx_type_cd")
        assert dx_prop.semantic_type == "categorical"

    def test_property_has_decoded_values(self, expected_response):
        interp = TableInterpretation.model_validate(expected_response)
        dx_prop = next(p for p in interp.properties if p.column == "dx_type_cd")
        assert len(dx_prop.decoded_values) == 4
        assert dx_prop.decoded_values[0]["raw"] == "CRC"
        assert dx_prop.decoded_values[0]["label"] == "Colorectal Cancer"

    def test_property_has_synonyms(self, expected_response):
        interp = TableInterpretation.model_validate(expected_response)
        dx_prop = next(p for p in interp.properties if p.column == "dx_type_cd")
        assert "cancer type" in dx_prop.synonyms


class TestAssertionEmission:
    def test_emits_entity_name_assertion(self, engine, sample_metadata):
        assertions = engine.interpret_table(sample_metadata)
        entity_assertions = [a for a in assertions
                            if a.predicate == AssertionPredicate.HAS_ENTITY_NAME]
        assert len(entity_assertions) == 1
        assert entity_assertions[0].payload["value"] == "Cancer Diagnosis"
        assert entity_assertions[0].source == "llm_interpretation"

    def test_emits_property_name_assertions(self, engine, sample_metadata):
        assertions = engine.interpret_table(sample_metadata)
        prop_assertions = [a for a in assertions
                          if a.predicate == AssertionPredicate.HAS_PROPERTY_NAME]
        assert len(prop_assertions) == 4
        names = {a.payload["value"] for a in prop_assertions}
        assert "Diagnosis Type" in names
        assert "Date of Diagnosis" in names

    def test_emits_semantic_type_assertions(self, engine, sample_metadata):
        assertions = engine.interpret_table(sample_metadata)
        type_assertions = [a for a in assertions
                          if a.predicate == AssertionPredicate.HAS_SEMANTIC_TYPE]
        assert len(type_assertions) == 4
        dx_type = next(a for a in type_assertions if "dx_type_cd" in a.subject_ref)
        assert dx_type.payload["value"] == "categorical"

    def test_emits_decoded_value_assertions(self, engine, sample_metadata):
        assertions = engine.interpret_table(sample_metadata)
        decoded = [a for a in assertions
                  if a.predicate == AssertionPredicate.HAS_DECODED_VALUE]
        assert len(decoded) > 0
        crc = next(a for a in decoded if a.payload.get("raw") == "CRC")
        assert crc.payload["label"] == "Colorectal Cancer"

    def test_emits_alias_assertions(self, engine, sample_metadata):
        assertions = engine.interpret_table(sample_metadata)
        aliases = [a for a in assertions
                   if a.predicate == AssertionPredicate.HAS_ALIAS]
        assert len(aliases) > 0
        entity_aliases = [a for a in aliases
                          if a.subject_ref == sample_metadata["table_ref"]]
        assert len(entity_aliases) > 0
        # First alias should be preferred
        assert entity_aliases[0].payload["is_preferred"] is True

    def test_all_assertions_have_llm_source(self, engine, sample_metadata):
        assertions = engine.interpret_table(sample_metadata)
        for a in assertions:
            assert a.source == "llm_interpretation"
            assert a.run_id == "test-run"


class TestLLMFailureHandling:
    def test_invalid_json_returns_empty(self):
        bad_llm = MagicMock()
        bad_llm.invoke.return_value = MagicMock(content="not valid json {{{")
        engine = SemanticEngine(llm=bad_llm, run_id="test-run")
        sample = {"table_ref": "unity://cdm.clinical.tbl", "table_name": "tbl",
                  "columns": [], "sample_rows": [], "comment": None}
        assertions = engine.interpret_table(sample)
        assert assertions == []

    def test_llm_timeout_returns_empty(self):
        bad_llm = MagicMock()
        bad_llm.invoke.side_effect = TimeoutError("LLM timed out")
        engine = SemanticEngine(llm=bad_llm, run_id="test-run")
        sample = {"table_ref": "unity://cdm.clinical.tbl", "table_name": "tbl",
                  "columns": [], "sample_rows": [], "comment": None}
        assertions = engine.interpret_table(sample)
        assert assertions == []

    def test_partial_response_handled(self):
        partial_llm = MagicMock()
        partial_llm.invoke.return_value = MagicMock(
            content=json.dumps({
                "entity_name": "Partial",
                "entity_description": None,
                "synonyms": [],
                "properties": [],
            })
        )
        engine = SemanticEngine(llm=partial_llm, run_id="test-run")
        sample = {"table_ref": "unity://cdm.clinical.tbl", "table_name": "tbl",
                  "columns": [], "sample_rows": [], "comment": None}
        assertions = engine.interpret_table(sample)
        entity_assertions = [a for a in assertions
                            if a.predicate == AssertionPredicate.HAS_ENTITY_NAME]
        assert len(entity_assertions) == 1


class TestInterpretationToAssertionsCharacterization:
    """Characterization tests capturing current behavior of _interpretation_to_assertions."""

    def test_converts_full_interpretation_to_assertions(self):
        engine = SemanticEngine(run_id="test-run")

        interpretation = TableInterpretation(
            entity_name="Patient",
            entity_description="Patient demographic records",
            synonyms=["Subject"],
            properties=[
                PropertyInterpretation(
                    column="dx_code",
                    name="Diagnosis Code",
                    semantic_type="categorical",
                    vocabulary_guess="ICD-10",
                    decoded_values=[
                        {"raw": "C18", "label": "Colorectal Cancer"},
                        {"raw": "C50", "label": "Breast Cancer"},
                    ],
                ),
                PropertyInterpretation(
                    column="birth_date",
                    name="Date of Birth",
                    semantic_type="temporal",
                ),
            ],
        )
        table_ref = "unity://cat.sch.patients"

        assertions = engine._interpretation_to_assertions(interpretation, table_ref)

        # All results are Assertion objects
        assert isinstance(assertions, list)
        assert all(isinstance(a, Assertion) for a in assertions)

        predicates = [a.predicate for a in assertions]

        # HAS_ENTITY_NAME for the entity
        assert AssertionPredicate.HAS_ENTITY_NAME in predicates
        entity_a = [a for a in assertions if a.predicate == AssertionPredicate.HAS_ENTITY_NAME]
        assert len(entity_a) == 1
        assert entity_a[0].payload["value"] == "Patient"

        # HAS_ALIAS for entity-level alias (replaces HAS_SYNONYM)
        assert AssertionPredicate.HAS_ALIAS in predicates
        syn_a = [a for a in assertions if a.predicate == AssertionPredicate.HAS_ALIAS]
        assert any(a.payload["value"] == "Subject" for a in syn_a)
        # First alias should be marked preferred
        entity_aliases = [
            a for a in syn_a
            if a.subject_ref == table_ref
        ]
        assert entity_aliases[0].payload["is_preferred"] is True

        # HAS_PROPERTY_NAME for each property
        prop_a = [a for a in assertions if a.predicate == AssertionPredicate.HAS_PROPERTY_NAME]
        assert len(prop_a) == 2
        prop_names = {a.payload["value"] for a in prop_a}
        assert "Diagnosis Code" in prop_names
        assert "Date of Birth" in prop_names

        # HAS_SEMANTIC_TYPE for each property
        type_a = [a for a in assertions if a.predicate == AssertionPredicate.HAS_SEMANTIC_TYPE]
        assert len(type_a) == 2

        # HAS_DECODED_VALUE for decoded values on dx_code
        decoded_a = [a for a in assertions if a.predicate == AssertionPredicate.HAS_DECODED_VALUE]
        assert len(decoded_a) == 2
        raw_values = {a.payload["raw"] for a in decoded_a}
        assert "C18" in raw_values
        assert "C50" in raw_values

        # VOCABULARY_MATCH for the property with vocabulary_guess
        vocab_a = [a for a in assertions if a.predicate == AssertionPredicate.VOCABULARY_MATCH]
        assert len(vocab_a) == 1
        assert vocab_a[0].payload["value"] == "ICD-10"

        # All assertions have correct source and run_id
        for a in assertions:
            assert a.source == "llm_interpretation"
            assert a.run_id == "test-run"

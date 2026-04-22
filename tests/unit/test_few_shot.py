"""Tests for few-shot example library (tasks 8.1–8.6)."""
import pytest

pytestmark = pytest.mark.unit


class TestFewShotStorage:
    """Task 8.1: structured storage, selectable by domain+stage key."""

    def test_lookup_healthcare_stage_a(self) -> None:
        from sema.engine.few_shot import get_examples

        examples = get_examples(domain="healthcare", stage="A")
        assert len(examples) >= 3
        assert len(examples) <= 5

    def test_lookup_healthcare_stage_b(self) -> None:
        from sema.engine.few_shot import get_examples

        examples = get_examples(domain="healthcare", stage="B")
        assert len(examples) >= 8
        assert len(examples) <= 12

    def test_lookup_healthcare_stage_c(self) -> None:
        from sema.engine.few_shot import get_examples

        examples = get_examples(domain="healthcare", stage="C")
        assert len(examples) >= 6
        assert len(examples) <= 10

    def test_unknown_domain_returns_empty(self) -> None:
        from sema.engine.few_shot import get_examples

        examples = get_examples(domain="financial", stage="A")
        assert examples == []

    def test_none_domain_returns_empty(self) -> None:
        from sema.engine.few_shot import get_examples

        examples = get_examples(domain=None, stage="A")
        assert examples == []


class TestHealthcareStageAExamples:
    """Task 8.2: 5 healthcare Stage A examples."""

    def test_example_has_input_and_output(self) -> None:
        from sema.engine.few_shot import get_examples

        examples = get_examples(domain="healthcare", stage="A")
        for ex in examples:
            assert "input" in ex
            assert "output" in ex
            assert "table_name" in ex["input"]
            assert "primary_entity" in ex["output"]
            assert "grain_hypothesis" in ex["output"]

    def test_covers_required_tables(self) -> None:
        from sema.engine.few_shot import get_examples

        examples = get_examples(domain="healthcare", stage="A")
        table_names = {ex["input"]["table_name"] for ex in examples}
        assert "patient" in table_names
        assert "sample" in table_names
        assert "mutation" in table_names


class TestHealthcareStageBExamples:
    """Task 8.3: 12 healthcare Stage B column examples."""

    def test_example_has_input_and_output(self) -> None:
        from sema.engine.few_shot import get_examples

        examples = get_examples(domain="healthcare", stage="B")
        for ex in examples:
            assert "input" in ex
            assert "output" in ex
            assert "column" in ex["input"]
            assert "semantic_type" in ex["output"]
            assert "needs_stage_c" in ex["output"]

    def test_uses_semantic_family_not_specific_ontology(self) -> None:
        from sema.engine.few_shot import get_examples

        examples = get_examples(domain="healthcare", stage="B")
        for ex in examples:
            families = ex["output"].get("candidate_vocab_families", [])
            for f in families:
                # Should NOT contain specific ontology names
                # unless the column explicitly identifies them
                col_name = ex["input"]["column"].lower()
                if "icd" not in col_name and "snomed" not in col_name:
                    assert "ICD-10" not in f
                    assert "SNOMED CT" not in f


class TestHealthcareStageCExamples:
    """Task 8.4: 8 healthcare Stage C value decoding examples."""

    def test_example_has_input_and_output(self) -> None:
        from sema.engine.few_shot import get_examples

        examples = get_examples(domain="healthcare", stage="C")
        for ex in examples:
            assert "input" in ex
            assert "output" in ex
            assert "column" in ex["input"]
            assert "values" in ex["input"]
            assert "decoded_categories" in ex["output"]


class TestFewShotInjection:
    """Task 8.5: injection into prompt builders."""

    def test_format_stage_a_examples(self) -> None:
        from sema.engine.few_shot import format_examples

        block = format_examples(domain="healthcare", stage="A")
        assert len(block) > 0
        assert "patient" in block.lower()
        assert "primary_entity" in block

    def test_format_stage_b_examples(self) -> None:
        from sema.engine.few_shot import format_examples

        block = format_examples(domain="healthcare", stage="B")
        assert len(block) > 0
        assert "semantic_type" in block

    def test_format_unknown_domain_falls_back_to_generic(self) -> None:
        from sema.engine.few_shot import format_examples

        block = format_examples(domain="geology", stage="A")
        assert len(block) > 0
        assert "primary_entity" in block

    def test_format_none_domain_returns_generic(self) -> None:
        from sema.engine.few_shot import format_examples

        block = format_examples(domain=None, stage="A")
        assert len(block) > 0
        assert "primary_entity" in block


class TestGenericFewShot:
    """Generic base layer — industry-agnostic archetypes."""

    def test_generic_stage_a_has_archetypes(self) -> None:
        from sema.engine.few_shot import get_examples

        examples = get_examples(domain="generic", stage="A")
        assert len(examples) >= 4
        entities = {ex["output"]["primary_entity"] for ex in examples}
        assert {"Event", "Order", "Product"} & entities

    def test_generic_stage_b_covers_column_archetypes(self) -> None:
        from sema.engine.few_shot import get_examples

        examples = get_examples(domain="generic", stage="B")
        sem_types = {ex["output"]["semantic_type"] for ex in examples}
        assert {
            "identifier", "temporal", "numeric",
            "categorical", "boolean",
        } <= sem_types

    def test_generic_stage_c_has_decoding_patterns(self) -> None:
        from sema.engine.few_shot import get_examples

        examples = get_examples(domain="generic", stage="C")
        assert len(examples) >= 3

    def test_compose_prepends_generic_to_domain(self) -> None:
        from sema.engine.few_shot import compose_examples, get_examples

        generic = get_examples(domain="generic", stage="A")
        healthcare = get_examples(domain="healthcare", stage="A")
        composed = compose_examples(domain="healthcare", stage="A")
        assert len(composed) == len(generic) + len(healthcare)
        assert composed[:len(generic)] == generic
        assert composed[len(generic):] == healthcare


class TestHoldoutDisjointness:
    """Task 8.6: no overlap between few-shot source tables and holdout."""

    def test_no_overlap_with_holdout(self) -> None:
        import yaml
        from pathlib import Path
        from sema.engine.few_shot import get_examples

        holdout_path = (
            Path(__file__).resolve().parents[2]
            / "showcase" / "cbioportal_to_omop"
            / "slices" / "holdout.yaml"
        )
        if not holdout_path.exists():
            pytest.skip("holdout.yaml not found")

        with open(holdout_path) as f:
            holdout = yaml.safe_load(f)

        holdout_tables = {
            t["table_name"] for t in holdout.get("tables", [])
        }

        for stage in ("A", "B", "C"):
            examples = get_examples(domain="healthcare", stage=stage)
            for ex in examples:
                source_table = ex["input"].get("table_name", "")
                assert source_table not in holdout_tables, (
                    f"Few-shot example from '{source_table}' "
                    f"overlaps with holdout set"
                )

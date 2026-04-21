"""Tests for domain-aware prompt composition layers (tasks 7.1–7.6)."""
import pytest

from sema.models.domain import DomainContext, DomainCandidate

pytestmark = pytest.mark.unit


def _healthcare_ctx(
    confidence: float = 0.85,
    source: str = "user",
) -> DomainContext:
    return DomainContext(
        declared_domain="healthcare",
        domain_confidence=confidence,
        domain_source=source,  # type: ignore[arg-type]
    )


def _financial_ctx() -> DomainContext:
    return DomainContext(
        declared_domain="financial",
        domain_confidence=1.0,
        domain_source="user",
    )


def _conflict_ctx() -> DomainContext:
    """Declared healthcare, but profiler says financial with high confidence."""
    return DomainContext(
        declared_domain="healthcare",
        detected_domain="financial",
        domain_confidence=0.72,
        domain_source="user",
        alternate_domains=[
            DomainCandidate(domain="financial", confidence=0.72),
        ],
    )


def _low_confidence_ctx() -> DomainContext:
    return DomainContext(
        detected_domain="healthcare",
        domain_confidence=0.3,
        domain_source="profiler",
    )


class TestDomainBiasHeader:
    """Task 7.1: domain bias header with conflict handling."""

    def test_healthcare_header(self) -> None:
        from sema.engine.domain_prompts import build_domain_bias_header

        header = build_domain_bias_header(_healthcare_ctx())
        assert "healthcare" in header.lower()
        assert len(header) > 0

    def test_financial_header(self) -> None:
        from sema.engine.domain_prompts import build_domain_bias_header

        header = build_domain_bias_header(_financial_ctx())
        assert "financial" in header.lower()

    def test_no_domain_returns_empty(self) -> None:
        from sema.engine.domain_prompts import build_domain_bias_header

        header = build_domain_bias_header(None)
        assert header == ""

    def test_default_context_returns_empty(self) -> None:
        from sema.engine.domain_prompts import build_domain_bias_header

        header = build_domain_bias_header(DomainContext())
        assert header == ""

    def test_low_confidence_profiler_returns_empty(self) -> None:
        from sema.engine.domain_prompts import build_domain_bias_header

        header = build_domain_bias_header(_low_confidence_ctx())
        assert header == ""

    def test_user_declared_always_applies(self) -> None:
        from sema.engine.domain_prompts import build_domain_bias_header

        ctx = DomainContext(
            declared_domain="healthcare",
            domain_confidence=0.2,
            domain_source="user",
        )
        header = build_domain_bias_header(ctx)
        assert "healthcare" in header.lower()

    def test_conflict_mentions_both_domains(self) -> None:
        from sema.engine.domain_prompts import build_domain_bias_header

        header = build_domain_bias_header(_conflict_ctx())
        assert "healthcare" in header.lower()
        assert "financial" in header.lower()


class TestSemanticTypeInventory:
    """Tasks 7.2 and 7.3: domain-specific and generic inventories."""

    def test_healthcare_inventory(self) -> None:
        from sema.engine.domain_prompts import (
            get_semantic_type_inventory,
        )

        inv = get_semantic_type_inventory(_healthcare_ctx())
        assert "patient identifier" in inv.lower()
        assert "diagnosis" in inv.lower()
        assert "biomarker" in inv.lower()
        assert "therapy" in inv.lower() or "drug" in inv.lower()
        assert "outcome" in inv.lower() or "survival" in inv.lower()

    def test_generic_inventory(self) -> None:
        from sema.engine.domain_prompts import (
            get_semantic_type_inventory,
        )

        inv = get_semantic_type_inventory(None)
        assert "identifier" in inv.lower()
        assert "categorical" in inv.lower()
        assert "temporal" in inv.lower()
        assert "numeric" in inv.lower()
        assert "free_text" in inv.lower()

    def test_no_domain_uses_generic(self) -> None:
        from sema.engine.domain_prompts import (
            get_semantic_type_inventory,
        )

        inv_none = get_semantic_type_inventory(None)
        inv_default = get_semantic_type_inventory(DomainContext())
        assert inv_none == inv_default

    def test_unknown_domain_uses_generic(self) -> None:
        from sema.engine.domain_prompts import (
            get_semantic_type_inventory,
        )

        ctx = DomainContext(
            declared_domain="geology",
            domain_confidence=1.0,
            domain_source="user",
        )
        inv = get_semantic_type_inventory(ctx)
        # Unknown domain falls back to generic
        assert "identifier" in inv.lower()
        assert "categorical" in inv.lower()


class TestVocabFamilyHints:
    """Task 7.4: vocabulary family hints for healthcare domain."""

    def test_healthcare_hints(self) -> None:
        from sema.engine.domain_prompts import (
            build_vocab_family_hints,
        )

        hints = build_vocab_family_hints(_healthcare_ctx())
        assert len(hints) > 0
        lower = hints.lower()
        assert "omop" in lower or "snomed" in lower or "gene" in lower

    def test_no_domain_returns_empty(self) -> None:
        from sema.engine.domain_prompts import (
            build_vocab_family_hints,
        )

        hints = build_vocab_family_hints(None)
        assert hints == ""

    def test_default_context_returns_empty(self) -> None:
        from sema.engine.domain_prompts import (
            build_vocab_family_hints,
        )

        hints = build_vocab_family_hints(DomainContext())
        assert hints == ""


class TestPromptComposition:
    """Task 7.5: wiring domain layers into Stage A and Stage B prompts."""

    def test_stage_a_with_domain(self) -> None:
        from sema.engine.stage_utils import build_stage_a_prompt

        meta = {
            "table_name": "patient",
            "columns": [
                {"name": "patient_id", "data_type": "STRING"},
                {"name": "gender", "data_type": "STRING"},
            ],
        }
        prompt = build_stage_a_prompt(
            meta, domain_context=_healthcare_ctx(),
        )
        assert "healthcare" in prompt.lower()

    def test_stage_a_without_domain(self) -> None:
        from sema.engine.stage_utils import build_stage_a_prompt

        meta = {
            "table_name": "patient",
            "columns": [
                {"name": "patient_id", "data_type": "STRING"},
            ],
        }
        prompt = build_stage_a_prompt(meta, domain_context=None)
        assert "healthcare" not in prompt.lower()

    def test_stage_b_with_domain(self) -> None:
        from sema.engine.stage_utils import build_stage_b_prompt
        from sema.models.stages import StageAResult

        meta = {"table_name": "patient"}
        stage_a = StageAResult(
            primary_entity="Patient",
            grain_hypothesis="one row per patient",
            confidence=0.9,
        )
        batch = [{"name": "gender", "data_type": "STRING"}]
        prompt = build_stage_b_prompt(
            meta, batch, stage_a,
            domain_context=_healthcare_ctx(),
        )
        assert "healthcare" in prompt.lower()
        # Should use healthcare inventory, not generic
        assert "patient identifier" in prompt.lower()

    def test_stage_b_without_domain_uses_generic_inventory(self) -> None:
        from sema.engine.stage_utils import build_stage_b_prompt
        from sema.models.stages import StageAResult

        meta = {"table_name": "orders"}
        stage_a = StageAResult(
            primary_entity="Order",
            grain_hypothesis="one row per order",
            confidence=0.9,
        )
        batch = [{"name": "order_id", "data_type": "INT"}]
        prompt = build_stage_b_prompt(
            meta, batch, stage_a, domain_context=None,
        )
        # Generic inventory terms
        assert "identifier" in prompt.lower()
        assert "categorical" in prompt.lower()

    def test_stage_b_includes_vocab_hints_when_domain(self) -> None:
        from sema.engine.stage_utils import build_stage_b_prompt
        from sema.models.stages import StageAResult

        meta = {"table_name": "patient"}
        stage_a = StageAResult(
            primary_entity="Patient",
            grain_hypothesis="one row per patient",
            confidence=0.9,
        )
        batch = [{"name": "gender", "data_type": "STRING"}]
        prompt = build_stage_b_prompt(
            meta, batch, stage_a,
            domain_context=_healthcare_ctx(),
        )
        # Should contain vocab family hints
        assert "vocabulary" in prompt.lower() or "ontolog" in prompt.lower()

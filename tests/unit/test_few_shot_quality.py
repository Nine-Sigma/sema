"""Tests for few-shot example quality: synonyms coverage and token cost."""
from __future__ import annotations

import pytest

from sema.engine.few_shot import format_examples, get_examples
from sema.engine.few_shot_healthcare import HEALTHCARE_STAGE_B

pytestmark = pytest.mark.unit


class TestStageBSynonymsCoverage:
    def test_most_b_examples_include_synonyms(self) -> None:
        """B examples must teach the LLM that synonyms is a live field.

        Without non-empty synonyms in examples, the LLM drops aliases.
        Caught empirically on step 4 dev slice (52 aliases regression).
        """
        with_synonyms = sum(
            1 for ex in HEALTHCARE_STAGE_B
            if ex["output"].get("synonyms")
        )
        assert with_synonyms >= 6, (
            f"Only {with_synonyms}/{len(HEALTHCARE_STAGE_B)} B examples "
            f"show non-empty synonyms — LLM will learn to drop them."
        )

    def test_synonyms_present_for_identifiers_and_domain_terms(
        self,
    ) -> None:
        """Synonyms should cover identifier and domain-specific columns."""
        by_col = {
            ex["input"]["column"]: ex["output"]
            for ex in HEALTHCARE_STAGE_B
        }
        for col in ("patient_id", "hugo_symbol", "tmb", "msi_type"):
            syns = by_col[col].get("synonyms", [])
            assert syns, f"{col} example must show synonyms"


class TestFewShotFormatCompact:
    def test_uses_compact_json_without_indent(self) -> None:
        """Compact JSON reduces prompt tokens by ~25-30%."""
        block = format_examples("healthcare", "B")
        assert block, "sanity: block must not be empty"
        assert '\n  "' not in block, (
            "Examples should use compact JSON, not indented — "
            "found multi-line JSON structure which wastes tokens."
        )

    def test_block_stays_under_token_budget(self) -> None:
        """Stage B composed (generic + healthcare) must stay under 2100 tokens.

        Budget raised from the 1200 healthcare-only ceiling after adding the
        8-example generic base layer. Target ~90 tokens/example at compact
        JSON, 20 composed examples, plus framing.
        """
        block = format_examples("healthcare", "B")
        approx_tokens = len(block) // 4
        assert approx_tokens <= 2100, (
            f"Stage B few-shot block is {approx_tokens} tokens — "
            f"budget is 2100."
        )

"""Tests for domain_affinity on VocabPattern and cross-domain disambiguation."""

import pytest

from sema.engine.vocabulary import (
    VocabColumnContext,
    VOCABULARY_PATTERNS,
    detect_vocabulary_pattern,
)
from sema.engine.vocab_pattern import detect_first_match

pytestmark = pytest.mark.unit


class TestCPTvsZIPDisambiguation:
    r"""CPT and ZIP both match ^\d{5}$ — domain_affinity disambiguates."""

    def test_five_digit_in_healthcare_is_cpt(self) -> None:
        values = ["99213", "99214", "99215"]
        context = VocabColumnContext(
            column_name="procedure_code",
            table_name="claims",
        )
        result = detect_first_match(
            VOCABULARY_PATTERNS, values, context,
            warehouse_domains={"healthcare": 0.8},
        )
        assert result is not None
        assert result["vocabulary"] == "CPT"

    def test_five_digit_in_real_estate_is_zip(self) -> None:
        values = ["10001", "90210", "60601"]
        context = VocabColumnContext(
            column_name="zip_code",
            table_name="properties",
        )
        result = detect_first_match(
            VOCABULARY_PATTERNS, values, context,
            warehouse_domains={"real_estate": 0.8},
        )
        assert result is not None
        assert result["vocabulary"] == "ZIP"

    def test_no_profile_falls_back_to_context_keywords(self) -> None:
        """Without warehouse profile, context_keywords disambiguate."""
        values = ["99213", "99214"]
        context = VocabColumnContext(
            column_name="procedure_code",
            table_name="billing",
        )
        # No warehouse_domains -> domain_affinity ignored
        result = detect_first_match(
            VOCABULARY_PATTERNS, values, context,
            warehouse_domains=None,
        )
        assert result is not None
        # CPT matches because "billing" is in its context_keywords
        assert result["vocabulary"] == "CPT"

    def test_zip_context_without_profile(self) -> None:
        values = ["10001", "90210"]
        context = VocabColumnContext(
            column_name="postal_code",
            table_name="address",
        )
        result = detect_first_match(
            VOCABULARY_PATTERNS, values, context,
            warehouse_domains=None,
        )
        assert result is not None
        assert result["vocabulary"] == "ZIP"


class TestNewPatterns:
    def test_naics_in_financial_domain(self) -> None:
        values = ["5112", "5121", "5191"]
        context = VocabColumnContext(
            column_name="naics_code",
            table_name="companies",
        )
        result = detect_first_match(
            VOCABULARY_PATTERNS, values, context,
            warehouse_domains={"financial": 0.7},
        )
        assert result is not None
        assert result["vocabulary"] == "NAICS"

    def test_isin_pattern(self) -> None:
        values = ["US0378331005", "GB0002634946"]
        context = VocabColumnContext(
            column_name="isin_code",
            table_name="securities",
        )
        result = detect_first_match(
            VOCABULARY_PATTERNS, values, context,
            warehouse_domains={"financial": 0.8},
        )
        assert result is not None
        assert result["vocabulary"] == "ISIN"

    def test_iso_4217_currency(self) -> None:
        values = ["USD", "EUR", "GBP"]
        context = VocabColumnContext(
            column_name="currency_code",
            table_name="transactions",
        )
        result = detect_first_match(
            VOCABULARY_PATTERNS, values, context,
        )
        assert result is not None
        assert result["vocabulary"] == "ISO-4217"

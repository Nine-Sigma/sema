"""Tests for defensive sanitization of Stage B column field output."""
from __future__ import annotations

import pytest

from sema.engine.stage_utils import sanitize_column_name

pytestmark = pytest.mark.unit


class TestSanitizeColumnName:
    def test_strips_parenthesized_type_suffix(self) -> None:
        assert sanitize_column_name("BIOTYPE (STRING)") == "BIOTYPE"

    def test_strips_bracket_suffix(self) -> None:
        assert sanitize_column_name("age [INT]") == "age"

    def test_strips_colon_type_suffix(self) -> None:
        assert sanitize_column_name("patient_id: VARCHAR") == "patient_id"

    def test_preserves_clean_column_names(self) -> None:
        assert sanitize_column_name("patient_id") == "patient_id"
        assert sanitize_column_name("Hugo_Symbol") == "Hugo_Symbol"

    def test_strips_trailing_whitespace(self) -> None:
        assert sanitize_column_name("  BIOTYPE  ") == "BIOTYPE"

    def test_keeps_internal_underscores_and_case(self) -> None:
        assert sanitize_column_name("OS_STATUS") == "OS_STATUS"
        assert sanitize_column_name("AJCC_PATHOLOGIC_TUMOR_STAGE") == (
            "AJCC_PATHOLOGIC_TUMOR_STAGE"
        )

    def test_empty_and_malformed(self) -> None:
        assert sanitize_column_name("") == ""
        assert sanitize_column_name("   ") == ""
        assert sanitize_column_name("(STRING)") == ""

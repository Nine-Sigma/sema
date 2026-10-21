"""Inline code lists: the versioned lexicon pattern over description and notes text."""

from __future__ import annotations

import re

import pytest

pytestmark = pytest.mark.unit

# Design Decision 8: the one S2 pattern, its label group named by the code list
# display role, since a fixture column is called label (source-literal guard).
S2_PATTERN = r"(?P<code>[^\s=]+)\s*=\s*(?P<display>[^=\n]+?)(?=\s+[^\s=]+\s*=|\n|$)"
BOROUGHS = "1 = Manhattan\n2 = Bronx\n3 = Brooklyn\n4 = Queens"


def _inline(text: str) -> dict[str, str] | None:
    from sema.intake.table_schema_utils import inline_values

    return inline_values(text)


def _source(pattern: object) -> str:
    return str(getattr(pattern, "pattern", pattern))


def test_borough_list_on_line_breaks() -> None:
    assert _inline(BOROUGHS) == {
        "1": "Manhattan",
        "2": "Bronx",
        "3": "Brooklyn",
        "4": "Queens",
    }


def test_single_letter_codes_without_spaces() -> None:
    assert _inline("P=partial E=entire") == {"P": "partial", "E": "entire"}


def test_quoted_flag_prose_is_not_a_list() -> None:
    assert _inline("‘Y’ or ‘N’ field") is None


def test_format_mask_is_not_a_list() -> None:
    assert _inline("CCCNNNNNNNNNNNNN") is None


def test_one_pair_is_not_a_list() -> None:
    assert _inline("A = active") is None


def test_order_is_preserved() -> None:
    values = _inline("Z = last\nA = first\nM = middle")

    assert values is not None
    assert list(values.items()) == [("Z", "last"), ("A", "first"), ("M", "middle")]


def test_comma_and_pipe_separated_list_yields_nothing_under_the_s2_patterns() -> None:
    assert _inline("1, Manhattan | 2, Bronx") is None


def test_empty_text_yields_nothing() -> None:
    assert _inline("") is None


def test_lexicon_holds_exactly_the_s2_pattern_with_its_version() -> None:
    from sema.intake import lexicon

    assert [_source(pattern) for pattern in lexicon.PATTERNS] == [S2_PATTERN]
    assert isinstance(lexicon.PATTERNS_VERSION, str)
    assert lexicon.PATTERNS_VERSION


def test_lexicon_carries_its_version_and_reference_phrases() -> None:
    from sema.intake import lexicon

    assert isinstance(lexicon.LEXICON_VERSION, str)
    assert lexicon.LEXICON_VERSION
    phrases = [re.compile(_source(p), re.IGNORECASE) for p in lexicon.REFERENCE_PHRASES]
    assert any(p.search("for details see Other Thing dataset") for p in phrases)

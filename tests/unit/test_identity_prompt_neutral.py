"""Done-when 3 (N5): the proposer's prompt names no source.

Every string the prompt module holds (its instructions and its examples) is split into
tokens; no identifier-shaped token (one that holds `_` or a digit) may be a source literal: a
name a source profile, an intake fixture or a connector fixture holds. The literal guard
cannot see one identifier inside a long prompt string, so this test reads each token.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests.unit.source_literals_env import REPO_ROOT, source_texts
from tests.unit.test_connector_no_source_literals import CONNECTOR_FIXTURES, source_literals

pytestmark = pytest.mark.unit

PROMPT_MODULE = REPO_ROOT / "src" / "sema" / "engine" / "identity_prompt.py"
IDENTIFIER = re.compile(r"\w+")


def identifier_tokens(source: str) -> set[str]:
    """Tokens of the module's strings that hold an underscore or a digit, lowercased."""
    return {
        token.casefold()
        for text in source_texts(source)
        for token in IDENTIFIER.findall(text.text)
        if "_" in token or any(ch.isdigit() for ch in token)
    }


def literal_tokens() -> set[str]:
    return {key[0] for key in source_literals(modules=CONNECTOR_FIXTURES) if len(key) == 1}


def source_literal_tokens(source: str) -> set[str]:
    return identifier_tokens(source) & literal_tokens()


def _prompt_source() -> str:
    assert PROMPT_MODULE.is_file(), f"no proposer prompt module at {PROMPT_MODULE}"
    return PROMPT_MODULE.read_text(encoding="utf-8")


def test_the_prompt_states_the_proposal_fields() -> None:
    tokens = identifier_tokens(_prompt_source())

    assert {"entity_key", "latest_by", "row_key"} <= tokens


def test_no_identifier_of_the_prompt_is_a_source_literal() -> None:
    assert source_literal_tokens(_prompt_source()) == set()


def test_a_planted_source_literal_in_a_prompt_is_found(tmp_path: Path) -> None:
    planted = max((t for t in literal_tokens() if "_" in t), key=len)
    source = f'PROMPT = "Name the entity_key of the relation, e.g. {planted} and row_key."\n'

    assert source_literal_tokens(source) == {planted}

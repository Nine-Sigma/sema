"""Domain-neutral words the intake rules match, and how text is normalized first.

Normalization is English-only: case fold after splitting case changes,
non-alphanumerics become spaces, whitespace collapses, and a trailing plural
``s`` is stripped from each token.
"""

from __future__ import annotations

import re

HEADER_KEY_TOKENS: dict[str, frozenset[str]] = {
    "name": frozenset({"name", "title"}),
    "description": frozenset({"description", "summary"}),
}

_CASE_CHANGE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_NON_ALPHANUMERIC = re.compile(r"[^0-9a-z]+")
_MIN_PLURAL_LENGTH = 4


def normalize(text: str) -> str:
    split = _CASE_CHANGE.sub(" ", text).casefold()
    words = _NON_ALPHANUMERIC.sub(" ", split).split()
    return " ".join(_singular(word) for word in words)


def _singular(word: str) -> str:
    plural = len(word) >= _MIN_PLURAL_LENGTH and word.endswith("s")
    return word[:-1] if plural and not word.endswith("ss") else word

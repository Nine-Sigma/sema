"""A governed value whose term has no label builds into the context and renders as its code.

A graph read returns `label: None` for a term with no label, so the value candidate carries
`None`, never the code and never an empty string.
"""

from __future__ import annotations

from typing import Any

import pytest

from sema.consumers.nl2sql.prompting import build_sql_prompt
from sema.models.context import SemanticCandidateSet
from sema.pipeline.context import prune_to_sco
from tests.support.connector_fixture_context import (
    LABELLED_REGIONS,
    REGION_COLUMN,
    UNLABELLED_REGION,
    VALUES_QUESTION,
    lines_naming,
)

pytestmark = pytest.mark.unit


def _value(code: str, label: str | None, ambiguous: bool) -> dict[str, Any]:
    table, column = REGION_COLUMN
    return {"type": "value", "property_name": column, "column": column, "table": table,
            "code": code, "label": label, "ambiguous": ambiguous, "status": "auto",
            "confidence": 0.9, "confidence_policy": "semantic"}


def _candidates(ambiguous: bool) -> SemanticCandidateSet:
    members = [*LABELLED_REGIONS, (UNLABELLED_REGION, None)]
    return SemanticCandidateSet(
        query=VALUES_QUESTION,
        candidates=[_value(code, label, ambiguous) for code, label in members],
    )


@pytest.mark.parametrize("ambiguous", [False, True], ids=["governed", "ambiguous"])
def test_a_value_with_no_label_builds_and_renders_its_code_with_no_label(
    ambiguous: bool,
) -> None:
    sco = prune_to_sco(_candidates(ambiguous))
    text = build_sql_prompt(sco, VALUES_QUESTION)

    (rendered,) = lines_naming(text, UNLABELLED_REGION)
    assert f"'{UNLABELLED_REGION}' (no label)" in rendered, text
    for code, label in LABELLED_REGIONS:
        assert f"'{code}' ({label})" in rendered, text

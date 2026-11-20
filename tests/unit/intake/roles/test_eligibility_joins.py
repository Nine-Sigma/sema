"""Declared joins are scoped to the sentence that names the other entity.

The prose is synthetic so the sentence rule is tested apart from any fixture.
"""

from __future__ import annotations

import pytest

from sema.intake.roles.eligibility_text_utils import (
    JOIN_WITHOUT_COLUMNS,
    Dictionary,
    HeaderReading,
    declared_joins,
)

pytestmark = pytest.mark.unit

OWNER, OTHER = "Alpha Ledger", "Beta Ledger"
PROSE_KEY = "Notes"
PROPERTIES = ("Entry Ref", "Batch Day", "Shared Extra", "Owner Only")
SHARED = PROPERTIES[:3]
LINK = "Each row links back to the Beta Ledger records using the entry ref and batch day fields"
ASIDE = "The shared extra field is an alternate identifier kept for older lookups"


Opened = tuple[str, tuple[str, ...], tuple[str, ...]]


def _joins(prose: str) -> tuple[list[tuple[str, ...]], list[Opened]]:
    owner = Dictionary(
        relation="alpha_columns", header="alpha_header", entity=OWNER, aliases=(OWNER,),
        properties=tuple((name, "") for name in PROPERTIES),
    )
    other = Dictionary(
        relation="beta_columns", header=None, entity=OTHER, aliases=(OTHER,),
        properties=tuple((name, "") for name in SHARED),
    )
    header = HeaderReading(
        relation="alpha_header", name=OWNER, description=None,
        values={PROSE_KEY: prose}, prose=(PROSE_KEY,),
    )
    joins, open_joins = declared_joins([owner, other], {"alpha_header": header})
    assert all((j.from_entity, j.to_entity) == (OWNER, OTHER) for j in joins)
    return [j.columns for j in joins], [(a.subject, a.members, a.hints) for a in open_joins]


@pytest.mark.parametrize(
    "boundary", [". ", "? ", "! ", ".\n", "\n", "\r\n", ".  "],
    ids=["period", "question", "exclamation", "period-newline", "newline", "crlf", "spaces"],
)
def test_a_shared_column_named_in_another_sentence_is_not_a_join_column(boundary: str) -> None:
    columns, open_joins = _joins(f"{LINK}{boundary}{ASIDE}.")

    assert columns == [("Entry Ref", "Batch Day")]
    assert open_joins == []


def test_punctuation_not_followed_by_whitespace_does_not_end_the_sentence() -> None:
    columns, _ = _joins("Rows link to Beta Ledger (format v2.1) by entry ref and shared extra.")

    assert columns == [("Entry Ref", "Shared Extra")]


def test_every_sentence_naming_the_other_entity_contributes_its_columns() -> None:
    prose = (
        "Rows link to Beta Ledger by entry ref. Batch day is kept for audit only. "
        "Beta Ledger also holds shared extra."
    )
    columns, _ = _joins(prose)

    assert columns == [("Entry Ref", "Shared Extra")]


def test_a_sentence_naming_the_entity_but_no_shared_column_is_an_ambiguity() -> None:
    columns, open_joins = _joins("Rows relate to the Beta Ledger. Entry ref is unique per row.")

    assert columns == []
    assert open_joins == [(f"alpha_columns.join.{OTHER}", SHARED, (JOIN_WITHOUT_COLUMNS,))]


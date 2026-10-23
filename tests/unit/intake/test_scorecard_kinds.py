"""The scorer's ``terms``, ``mappings``, and ``retirements`` kinds (S5 decision 15; bug-848).

Pinned contract, over invented imports:

- ``terms`` scores each manifest term by existence and display, one cell per
  ``(vocabulary, code)``: a term only the expected import writes is ``missing``, one both
  write with the same display is ``equal``, a different display is ``wrong``. A term with no
  definition and no external refs therefore still scores.
- ``mappings`` scores the authored SSSOM mappings, one cell per
  ``(subject_id, predicate_id, object_id)``; ``retirements`` scores the authored retirements,
  one cell per ``(kind, vocabulary, code)``. A record whose other fields differ is one
  ``wrong`` cell; a record whose identity differs is ``missing`` beside ``extra``; record
  order never moves a cell.
- The three kinds are in every scorecard, ``render`` prints a row for each, and the overall
  numbers count their cells.
"""

from __future__ import annotations

from typing import Any

import pytest

from sema.intake.scorecard import render, score
from tests.unit.intake.scorecard_kinds_env import (
    LEXICAL,
    MAPPINGS_KIND,
    NEW_KINDS,
    NO_STAGING,
    RETIREMENTS_KIND,
    TERMS_KIND,
    Change,
    add,
    authored,
    changed,
    counts,
    drop,
    edit,
    manifest,
    mapping,
    moved,
    oracle_mappings,
    oracle_retirements,
    oracle_terms,
    retirement,
    reverse,
    score_authored,
    score_terms,
    term,
)

pytestmark = pytest.mark.unit

TERM_CASES: list[Any] = [
    pytest.param(None, (3, 0, 0, 0), id="identical"),
    pytest.param(drop(1), (2, 1, 0, 0), id="dropped"),
    pytest.param(edit(0, display="Falling"), (2, 0, 1, 0), id="display"),
    pytest.param(add(term("HGH", "High")), (3, 0, 0, 1), id="extra"),
    pytest.param(reverse, (3, 0, 0, 0), id="order"),
]


@pytest.mark.parametrize(("change", "expected"), TERM_CASES)
def test_terms_score_by_existence_and_display_per_vocabulary_and_code(
    change: Change | None, expected: tuple[int, int, int, int]
) -> None:
    card = score_terms(changed(oracle_terms(), change))

    assert counts(card.kinds[TERMS_KIND]) == expected


def test_a_dropped_inline_term_lowers_the_overall_recall() -> None:
    card = score_terms(changed(oracle_terms(), drop(0)))

    assert moved(card) == {TERMS_KIND: (2, 1, 0, 0)}
    assert (card.precision, card.recall) == pytest.approx((1.0, 2 / 3))


def test_a_term_definition_alone_does_not_move_the_terms_kind() -> None:
    card = score_terms(changed(oracle_terms(), edit(0, definition="Water falling back")))

    assert counts(card.kinds[TERMS_KIND]) == (3, 0, 0, 0)


MAPPING_CASES: list[Any] = [
    pytest.param(None, (3, 0, 0, 0), id="identical"),
    pytest.param(drop(2), (2, 1, 0, 0), id="dropped"),
    pytest.param(edit(0, mapping_justification=LEXICAL), (2, 0, 1, 0), id="justification"),
    pytest.param(edit(1, object_id="ocean:0009"), (2, 1, 0, 1), id="identity"),
    pytest.param(
        add(mapping("tide:HGH", "skos:broadMatch", "ocean:0004")), (3, 0, 0, 1), id="new"
    ),
    pytest.param(reverse, (3, 0, 0, 0), id="order"),
]


@pytest.mark.parametrize(("change", "expected"), MAPPING_CASES)
def test_authored_mappings_score_one_cell_per_subject_predicate_and_object(
    change: Change | None, expected: tuple[int, int, int, int]
) -> None:
    evidence = authored(changed(oracle_mappings(), change), oracle_retirements())

    card = score_authored(evidence)

    assert moved(card) == {MAPPINGS_KIND: expected, RETIREMENTS_KIND: (3, 0, 0, 0)}


RETIREMENT_CASES: list[Any] = [
    pytest.param(None, (3, 0, 0, 0), id="identical"),
    pytest.param(drop(0), (2, 1, 0, 0), id="dropped"),
    pytest.param(edit(1, replaced_by="EBB"), (2, 0, 1, 0), id="replaced-by"),
    pytest.param(edit(0, code="NEAP2"), (2, 1, 0, 1), id="identity"),
    pytest.param(add(retirement("LOW", "EBB")), (3, 0, 0, 1), id="new"),
    pytest.param(reverse, (3, 0, 0, 0), id="order"),
]


@pytest.mark.parametrize(("change", "expected"), RETIREMENT_CASES)
def test_authored_retirements_score_one_cell_per_kind_vocabulary_and_code(
    change: Change | None, expected: tuple[int, int, int, int]
) -> None:
    evidence = authored(oracle_mappings(), changed(oracle_retirements(), change))

    card = score_authored(evidence)

    assert moved(card) == {MAPPINGS_KIND: (3, 0, 0, 0), RETIREMENTS_KIND: expected}


def test_the_new_kinds_are_in_every_scorecard() -> None:
    card = score(manifest(()), manifest(()), NO_STAGING, NO_STAGING)

    assert set(NEW_KINDS) <= set(card.kinds)


def _row(text: str, name: str) -> list[str]:
    (line,) = [line for line in text.splitlines() if line.split()[:1] == [name]]
    return line.split()


def test_render_prints_each_new_kind_and_the_overall_counts_them() -> None:
    terms = changed(oracle_terms(), drop(0))
    mappings = changed(oracle_mappings(), edit(0, mapping_justification=LEXICAL))
    retirements = changed(oracle_retirements(), add(retirement("LOW", "EBB")))
    card = score(
        manifest(oracle_terms()), manifest(terms), NO_STAGING, NO_STAGING,
        expected_authored=authored(oracle_mappings(), oracle_retirements()),
        authored=authored(mappings, retirements),
    )

    text = render(card)

    expected = {
        TERMS_KIND: (2, 1, 0, 0), MAPPINGS_KIND: (2, 0, 1, 0), RETIREMENTS_KIND: (3, 0, 0, 1),
    }
    for name, numbers in expected.items():
        assert tuple(int(cell) for cell in _row(text, name)[3:7]) == numbers, name
    assert tuple(int(cell) for cell in _row(text, "overall")[3:7]) == (7, 1, 1, 1)
    assert (card.precision, card.recall) == pytest.approx((7 / 9, 7 / 9))

"""Records sharing a key are numbered the same whatever order their values were built in."""

from __future__ import annotations

import pytest

from sema.intake.scorecard import score
from tests.unit.intake.scorecard_env import REGISTRY, manifest, ref, term

TERM_REFS = "term_external_refs"

pytestmark = pytest.mark.unit


def _refs_printed_by_insertion_order() -> tuple[dict[str, str], dict[str, str]]:
    first = (REGISTRY, "a")
    seconds = ((REGISTRY, f"z{n}") for n in range(1000))
    second = next(z for z in seconds if repr(frozenset((first, z))) != repr(frozenset((z, first))))
    return ref(*first), ref(*second)


def test_terms_sharing_a_key_number_their_ref_sets_by_content_not_insertion_order() -> None:
    first, second = _refs_printed_by_insertion_order()
    middle = (ref(REGISTRY, "m"), ref(REGISTRY, "n"))
    expected = manifest(terms=[term("K", "x", None, first, second), term("K", "y", None, *middle)])
    built = manifest(terms=[term("K", "x", None, second, first), term("K", "y", None, *middle)])

    card = score(expected, built, "", "")

    assert (card.kinds[TERM_REFS].equal, card.kinds[TERM_REFS].wrong) == (2, 0)

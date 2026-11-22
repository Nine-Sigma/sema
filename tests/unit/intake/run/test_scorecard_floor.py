"""The floor check of the accuracy scorer: each deviation from a leg's floor, either way."""

from __future__ import annotations

import pytest

from sema.intake.run.scorecard import Scorecard
from tests.unit.intake.scorecard_env import Floor, floor_misses

pytestmark = pytest.mark.unit


FLOOR = Floor(precision=0.9, recall=0.8, ignored=1)


def test_floor_misses_is_empty_when_the_leg_holds_its_floor() -> None:
    held = Scorecard(kinds={}, precision=0.9, recall=0.8)

    assert floor_misses(held, 1, FLOOR) == ()


@pytest.mark.parametrize(
    ("precision", "recall", "ignored", "named"),
    [
        (0.89, 0.8, 1, ("precision below",)),
        (0.9, 0.79, 1, ("recall below",)),
        (0.9, 0.8, 2, ("ignored above",)),
        (0.91, 0.8, 1, ("precision above",)),
        (0.9, 0.81, 1, ("recall above",)),
        (0.9, 0.8, 0, ("ignored below",)),
        (0.5, 0.5, 4, ("precision below", "recall below", "ignored above")),
    ],
)
def test_floor_misses_names_each_deviation_either_way(
    precision: float, recall: float, ignored: int, named: tuple[str, ...]
) -> None:
    card = Scorecard(kinds={}, precision=precision, recall=recall)

    misses = floor_misses(card, ignored, FLOOR)

    assert tuple(" ".join(miss.split()[0:4:3]) for miss in misses) == named

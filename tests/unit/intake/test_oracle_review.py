"""A floor row exists only over a reviewed oracle (S5 decision 13; generality-gate spec).

Pinned contract, all in ``scorecard_env``:

- Every oracle file starts with three comment lines: ``# review_status: <status>`` (one of
  ``reviewed_correct``, ``not_reviewed``, ``known_wrong``), ``# reviewed_by: <name>``, and
  ``# reviewed_on: <YYYY-MM-DD>``. ``oracle_review(path)`` returns a frozen record with
  ``status``, ``reviewed_by``, and ``reviewed_on``, and raises ``ValueError`` naming a line
  that is missing.
- ``ORACLES`` maps each scored source to its oracle path. ``unreviewed_floors(baseline,
  oracles)`` returns one message per floor row over an oracle that is not
  ``reviewed_correct``, naming the source and the status; over the committed ``BASELINE`` it
  returns nothing.
- ``UNREVIEWED`` lists every source whose oracle is ``not_reviewed`` or ``known_wrong``; none
  of them has a ``BASELINE`` row (``test_no_hint`` runs them as a strict expected failure
  named ``unreviewed``).

A held-out oracle is read only through the parser, and a failure there names the exception
type only.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from tests.unit.intake.scorecard_env import BASELINE, Floor

pytestmark = pytest.mark.unit

REVIEWED, NOT_REVIEWED, KNOWN_WRONG = "reviewed_correct", "not_reviewed", "known_wrong"
STATUSES = (REVIEWED, NOT_REVIEWED, KNOWN_WRONG)
REVIEWER, REVIEWED_ON = "A Reviewer", "2026-10-23"
HEADER_FIELDS = ("review_status", "reviewed_by", "reviewed_on")
BODY = "# An invented oracle.\nconcept_source:\n  spec_version: 2\n"
FLOOR = Floor(precision=1.0, recall=1.0, ignored=0)
CAP = "spec_cap"
BASELINE_SOURCES = sorted({source for source, _ in BASELINE})


def _env() -> ModuleType:
    return importlib.import_module("tests.unit.intake.scorecard_env")


def _header(status: str, *, without: str | None = None) -> str:
    values = dict(zip(HEADER_FIELDS, (status, REVIEWER, REVIEWED_ON), strict=True))
    lines = [f"# {name}: {value}" for name, value in values.items() if name != without]
    return "\n".join(lines) + "\n" + BODY


def _oracle(directory: Path, name: str, text: str) -> Path:
    path = directory / f"{name}.spec.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def _status(source: str, path: Path) -> str:
    """The header status; any failure names the exception type only (held-out rule)."""
    parse = _env().oracle_review
    try:
        status: str = parse(path).status
    except Exception as error:  # noqa: BLE001 - only the type may leave an oracle read
        failure = type(error).__name__
    else:
        return status
    pytest.fail(f"{source}: reading the oracle header raised {failure}", pytrace=False)


@pytest.mark.parametrize("status", STATUSES)
def test_oracle_review_reads_each_status_from_the_header(tmp_path: Path, status: str) -> None:
    path = _oracle(tmp_path, "invented", _header(status))

    review = _env().oracle_review(path)

    assert (review.status, review.reviewed_by, str(review.reviewed_on)) == (
        status, REVIEWER, REVIEWED_ON,
    )


@pytest.mark.parametrize("missing", HEADER_FIELDS)
def test_oracle_review_names_a_missing_header_line(tmp_path: Path, missing: str) -> None:
    path = _oracle(tmp_path, "invented", _header(REVIEWED, without=missing))
    parse = _env().oracle_review

    with pytest.raises(ValueError, match=missing):
        parse(path)


def test_the_review_record_is_frozen(tmp_path: Path) -> None:
    review = _env().oracle_review(_oracle(tmp_path, "invented", _header(REVIEWED)))

    with pytest.raises((AttributeError, TypeError, ValueError)):
        review.status = KNOWN_WRONG


def test_oracles_names_an_existing_oracle_for_every_baseline_source() -> None:
    oracles: dict[str, Path] = _env().ORACLES

    assert set(BASELINE_SOURCES) <= set(oracles), sorted(set(BASELINE_SOURCES) - set(oracles))
    assert [name for name, path in oracles.items() if not Path(path).is_file()] == []


@pytest.mark.parametrize("source", BASELINE_SOURCES)
def test_every_baseline_source_has_a_reviewed_correct_oracle(source: str) -> None:
    path = Path(_env().ORACLES[source])

    assert _status(source, path) == REVIEWED, source


def _synthetic(tmp_path: Path) -> tuple[dict[tuple[str, str], Floor], dict[str, Path]]:
    oracles = {
        f"source_{status}": _oracle(tmp_path, f"source_{status}", _header(status))
        for status in STATUSES
    }
    return {(name, CAP): FLOOR for name in oracles}, oracles


def test_a_floor_over_an_unreviewed_oracle_fails_the_guard_naming_source_and_status(
    tmp_path: Path,
) -> None:
    baseline, oracles = _synthetic(tmp_path)

    messages: tuple[str, ...] = _env().unreviewed_floors(baseline, oracles)

    for status in (NOT_REVIEWED, KNOWN_WRONG):
        source = f"source_{status}"
        assert [m for m in messages if source in m and status in m], (source, messages)
    assert not [m for m in messages if f"source_{REVIEWED}" in m], messages


def test_the_guard_passes_floors_over_reviewed_oracles_only(tmp_path: Path) -> None:
    baseline, oracles = _synthetic(tmp_path)
    reviewed = {key: floor for key, floor in baseline.items() if key[0].endswith(REVIEWED)}

    assert _env().unreviewed_floors(reviewed, oracles) == ()


def _guarded(baseline: Any, oracles: Any) -> tuple[str, ...]:
    guard = _env().unreviewed_floors
    try:
        found: tuple[str, ...] = guard(baseline, oracles)
    except Exception as error:  # noqa: BLE001 - only the type may leave an oracle read
        failure = type(error).__name__
    else:
        return found
    pytest.fail(f"the floor guard raised {failure}", pytrace=False)


def test_the_committed_baseline_holds_no_floor_over_an_unreviewed_oracle() -> None:
    assert _guarded(BASELINE, _env().ORACLES) == ()


def test_unreviewed_sources_are_listed_and_have_no_floor_row() -> None:
    env = _env()
    unreviewed: tuple[str, ...] = env.UNREVIEWED
    statuses = {source: _status(source, Path(path)) for source, path in env.ORACLES.items()}

    assert sorted(unreviewed) == sorted(s for s, found in statuses.items() if found != REVIEWED)
    floored = sorted(set(unreviewed) & set(BASELINE_SOURCES))
    assert floored == [], floored
    assert {statuses[source] for source in unreviewed} <= {NOT_REVIEWED, KNOWN_WRONG}

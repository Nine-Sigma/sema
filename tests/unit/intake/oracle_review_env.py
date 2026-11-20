"""The review status every oracle header states, and the guard over floor rows (S5 decision 13).

An oracle's leading comment block carries ``# review_status: <status>``,
``# reviewed_by: <name>``, and ``# reviewed_on: <YYYY-MM-DD>``. Reading stops at the first
non-comment line, so a held-out oracle's body never enters a test.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from itertools import takewhile
from pathlib import Path
from typing import Any

REVIEWED_CORRECT = "reviewed_correct"
STATUSES = (REVIEWED_CORRECT, "not_reviewed", "known_wrong")
HEADER_FIELDS = ("review_status", "reviewed_by", "reviewed_on")
HEADER_LINE = re.compile(r"#\s*(\w+):\s*(.*?)\s*$")
FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "intake"
ORACLES: dict[str, Path] = {
    "a1": FIXTURES / "partner_sl" / "standard_library.spec.yaml",
    "a2": FIXTURES / "tabular" / "codes_a2.spec.yaml",
    "a3": FIXTURES / "semsql" / "ontology.spec.yaml",
    "a7_csv": FIXTURES / "omop" / "cdm_field_level.spec.yaml",
    "a10": FIXTURES / "acris" / "acris.spec.yaml",
    "heldout3": FIXTURES / "heldout3" / "heldout3.spec.yaml",
    "heldout4": FIXTURES / "heldout4" / "heldout4.spec.yaml",
}


@dataclass(frozen=True)
class OracleReview:
    status: str
    reviewed_by: str
    reviewed_on: date


def oracle_review(path: Path) -> OracleReview:
    """The review record of the oracle at ``path``, from its leading comment block only."""
    found: dict[str, str] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in takewhile(lambda text: text.startswith("#"), handle):
            match = HEADER_LINE.match(line)
            if match is not None:
                found.setdefault(match.group(1), match.group(2))
    missing = [name for name in HEADER_FIELDS if not found.get(name)]
    if missing:
        raise ValueError(f"oracle header lacks {', '.join(missing)}")
    status = found["review_status"]
    if status not in STATUSES:
        raise ValueError(f"review_status {status!r} is not one of {', '.join(STATUSES)}")
    return OracleReview(
        status=status,
        reviewed_by=found["reviewed_by"],
        reviewed_on=date.fromisoformat(found["reviewed_on"]),
    )


def unreviewed_floors(
    baseline: Mapping[tuple[str, str], Any], oracles: Mapping[str, Path]
) -> tuple[str, ...]:
    """One message per floor row whose oracle is missing or not ``reviewed_correct``."""
    messages: list[str] = []
    for source, cap_id in baseline:
        if source not in oracles:
            messages.append(f"{source}-{cap_id}: floor row with no oracle")
            continue
        status = oracle_review(oracles[source]).status
        if status != REVIEWED_CORRECT:
            messages.append(f"{source}-{cap_id}: floor row over a {status} oracle")
    return tuple(messages)


def _unreviewed(oracles: Mapping[str, Path]) -> tuple[str, ...]:
    return tuple(
        source
        for source, path in oracles.items()
        if oracle_review(path).status != REVIEWED_CORRECT
    )


UNREVIEWED: tuple[str, ...] = _unreviewed(ORACLES)

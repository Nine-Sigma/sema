"""Write ``ambiguity_a8.csv``, the A8 role-ambiguity fixture, byte for byte reproducibly.

Run: ``uv run python tests/fixtures/intake/tabular/build_ambiguity_a8.py``.

Forty rows over four columns, built so the role predicates alone cannot choose:

- ``id`` and ``code``: integers, unique and non-null, so both are eligible for both
  identifier roles and, not being text, for no text role.
- ``label`` and ``description``: text with a mean length near 80, above the definition
  floor and inside the label bounds, so both are eligible for both the name and the
  definition roles. Each repeats one value, so neither is unique and neither is eligible
  for an identifier role.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

OUT = Path(__file__).with_name("ambiguity_a8.csv")
ROWS = 40
HEADER = ("id", "code", "label", "description")
FIRST_CODE = 1001
SUBJECTS = ("river", "valley", "harbour", "meadow", "summit", "orchard", "quarry", "lagoon")
QUALITIES = ("northern", "southern", "eastern", "western", "upper", "lower", "inner", "outer")


def label(n: int) -> str:
    quality, subject = QUALITIES[n % len(QUALITIES)], SUBJECTS[(n // 2) % len(SUBJECTS)]
    return f"Item {n:02d}: the {quality} {subject} entry kept as a long readable heading text"


def description(n: int) -> str:
    subject, quality = SUBJECTS[n % len(SUBJECTS)], QUALITIES[(n // 3) % len(QUALITIES)]
    return f"Entry {n:02d} records a {subject} in the {quality} part, written as a narrative note"


def rows() -> list[tuple[int, int, str, str]]:
    built = [(n, FIRST_CODE + n - 1, label(n), description(n)) for n in range(1, ROWS + 1)]
    last_id, last_code, _, last_description = built[-1]
    built[-1] = (last_id, last_code, built[-2][2], last_description)
    first_id, first_code, first_label, _ = built[1]
    built[1] = (first_id, first_code, first_label, built[0][3])
    return built


def build() -> bytes:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(HEADER)
    writer.writerows(rows())
    return buffer.getvalue().encode("utf-8")


if __name__ == "__main__":
    OUT.write_bytes(build())

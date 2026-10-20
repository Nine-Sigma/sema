"""Pure helpers for the tabular loader: header detection and column typing.

A header is the first row, within the first ``HEADER_SCAN_ROWS``, where at least
``HEADER_MIN_STRING_RATIO`` of the cells are non-empty strings, those strings are
unique within the row ignoring case, and none of them recurs in its own column in
the window. Text that parses as a number is not a header string: CSV cells arrive
as text.

A column is numeric only when every text cell survives the round trip: ``1.10``
and ``007`` stay text, because a code column must keep each code's identity.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from datetime import date, datetime
from typing import Any

HEADER_MIN_STRING_RATIO = 0.8
HEADER_SCAN_ROWS = 50
HEADER_TESTS = (
    f"at least {HEADER_MIN_STRING_RATIO:.0%} of cells are non-empty strings",
    "the strings are unique within the row, ignoring case",
    f"none recurs in its column within the first {HEADER_SCAN_ROWS} rows",
)
EMPTY_COLUMN_PREFIX = "_empty_"

_INTEGER_TEXT = re.compile(r"^-?(0|[1-9]\d*)$")
_BIGINT_RANGE = range(-(2**63), 2**63)
_BOOL_TEXT = {"true": True, "false": False}


def detect_header(rows: list[list[Any]]) -> int | None:
    """1-based row number of the header, or ``None`` when no row passes."""
    window = rows[:HEADER_SCAN_ROWS]
    width = max((len(row) for row in window), default=0)
    if width == 0:
        return None
    counts = [Counter[str]() for _ in range(width)]
    for row in window:
        for index, cell in enumerate(row):
            if (text := cell_text(cell)) is not None:
                counts[index][text] += 1
    for number, row in enumerate(window, start=1):
        if _is_header(row, width, counts):
            return number
    return None


def _is_header(row: list[Any], width: int, counts: list[Counter[str]]) -> bool:
    strings = {
        index: text
        for index, cell in enumerate(row)
        if isinstance(cell, str)
        and (text := cell_text(cell)) is not None
        and not _parses_float(text)
    }
    if len(strings) < HEADER_MIN_STRING_RATIO * width - 1e-9:
        return False
    if len({text.casefold() for text in strings.values()}) != len(strings):
        return False
    return all(counts[index][text] == 1 for index, text in strings.items())


def cell_text(cell: Any) -> str | None:
    if cell is None:
        return None
    text = str(cell).strip()
    return text or None


def column_names(header: list[Any], width: int) -> list[str]:
    padded = list(header) + [None] * (width - len(header))
    return [
        cell_text(cell) or f"{EMPTY_COLUMN_PREFIX}{index}"
        for index, cell in enumerate(padded, start=1)
    ]


def column_type(values: list[Any]) -> str:
    """DuckDB type for a column: typed only when every non-null cell agrees."""
    present = [value for value in values if cell_text(value) is not None]
    if not present:
        return "VARCHAR"
    for name, test in _TYPE_TESTS:
        if all(test(value) for value in present):
            return name
    return "VARCHAR"


def _is_bool(value: Any) -> bool:
    return isinstance(value, bool) or (
        isinstance(value, str) and value.strip().lower() in _BOOL_TEXT
    )


def _is_int(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return value in _BIGINT_RANGE
    if not isinstance(value, str) or not _INTEGER_TEXT.match(value.strip()):
        return False
    return int(value) in _BIGINT_RANGE


def _is_float(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return value in _BIGINT_RANGE
    if isinstance(value, float):
        return True
    return isinstance(value, str) and _float_round_trips(value.strip())


def _float_round_trips(text: str) -> bool:
    try:
        number = float(text)
    except ValueError:
        return False
    return math.isfinite(number) and repr(number) == text


def _parses_float(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


def _is_date(value: Any) -> bool:
    if isinstance(value, datetime):
        return value.time() == datetime.min.time()
    if isinstance(value, date):
        return True
    return isinstance(value, str) and _parses_date(value)


def _parses_date(text: str) -> bool:
    try:
        date.fromisoformat(text.strip())
    except ValueError:
        return False
    return len(text.strip()) == 10


def _is_timestamp(value: Any) -> bool:
    return isinstance(value, (datetime, date))


_TYPE_TESTS = (
    ("BOOLEAN", _is_bool),
    ("BIGINT", _is_int),
    ("DOUBLE", _is_float),
    ("DATE", _is_date),
    ("TIMESTAMP", _is_timestamp),
)


def convert(value: Any, duckdb_type: str) -> Any:
    """A cell as the Python value for its column type; text columns get ``str``."""
    if cell_text(value) is None:
        return None
    if duckdb_type == "BOOLEAN":
        return value if isinstance(value, bool) else _BOOL_TEXT[value.strip().lower()]
    if duckdb_type == "BIGINT":
        return int(value)
    if duckdb_type == "DOUBLE":
        return float(value)
    if duckdb_type == "DATE":
        return _as_date(value)
    if duckdb_type == "TIMESTAMP":
        return value
    return _as_text(value)


def _as_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value).strip())


def _as_text(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return str(value)

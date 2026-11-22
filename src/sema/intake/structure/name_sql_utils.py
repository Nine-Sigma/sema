"""Compare a text column with normalized names by counts only.

``listed_names_sql`` normalizes each value inside SQL exactly as ``lexicon.normalize`` does,
using only ``regexp_extract``, ``LOWER``, ``SUBSTR``, ``LENGTH``, ``TRIM``, ``NULLIF``, and
``CONCAT_WS``, which the staging and warehouse dialects share, and returns three counts: the
distinct normalized values, how many of those are among the names, and the rows it does not
compare.

Why it is exact. A trimmed value made of printable ASCII splits into maximal alphanumeric
chunks; ``CHUNK`` with ``i`` repeats can only match whole chunks, because each repeat must end
on a separator. Inside a chunk, ``normalize``'s case-change split leaves pieces of the form
upper run then lower-or-digit run, and every piece after the first starts upper, so the piece
patterns cannot split a run. Each piece is lowered (``casefold`` on ASCII), singularized by
the same length and suffix test, and joined in order. A value outside printable ASCII, or with
more words than any name has (so it equals none of them), is counted apart, not compared.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass

from sema.intake.land.landing import LOADER_COLUMN_PREFIX
from sema.intake.roles.lexicon import MIN_PLURAL_LENGTH
from sema.intake.land.profile_utils import TEXT_TYPE, sql_string

ALNUM = "A-Za-z0-9"
LEADING = f"^[^{ALNUM}]*"
CHUNK = f"[{ALNUM}]+[^{ALNUM}]+"
FIRST_PIECE = "^([A-Z]+[a-z0-9]*|[a-z0-9]+)"
LATER_PIECE = "[A-Z]+[a-z0-9]+"
PRINTABLE_ASCII = "[^ -~]"


@dataclass(frozen=True)
class NameListing:
    distinct: int
    listed: int
    uncompared: int

    def lists_exactly(self, name_count: int) -> bool:
        """Every value normalizes to one of the names and every name is listed."""
        return self.uncompared == 0 and self.distinct == self.listed == name_count


def listed_names_sql(relation_sql: str, column: str, names: Collection[str]) -> str:
    """Counts comparing the trimmed, normalized text of ``column`` with ``names``.

    ``names`` are already normalized; loader-added and blank values are skipped, as a
    schema-text read skips them.
    """
    width = max([1, *(len(name.split()) for name in names)])
    chunks = ", ".join(
        f"regexp_extract(v, {sql_string(_chunk(i))}, 1) AS c{i}" for i in range(width + 1)
    )
    pieces = ", ".join(
        f"LOWER(regexp_extract(c{i}, {sql_string(_piece(j))}, 1)) AS p{i}_{j}"
        for i in range(width)
        for j in range(width - i + 1)
    )
    words = ", ".join(
        f"NULLIF({_singular(f'p{i}_{j}')}, '')" for i in range(width) for j in range(width - i)
    )
    overflow = [f"c{width} <> ''", *(f"p{i}_{width - i} <> ''" for i in range(width))]
    apart = " OR ".join([f"regexp_extract(v, {sql_string(PRINTABLE_ASCII)}, 0) <> ''", *overflow])
    listed = ", ".join(sql_string(name) for name in sorted(names)) or "NULL"
    prefix = sql_string(LOADER_COLUMN_PREFIX)
    texts = (
        f"SELECT v FROM (SELECT TRIM(CAST({column} AS {TEXT_TYPE})) AS v FROM {relation_sql}"
        f" WHERE {column} IS NOT NULL) t"
        f" WHERE v <> '' AND SUBSTR(v, 1, {len(LOADER_COLUMN_PREFIX)}) <> {prefix}"
    )
    normalized = (
        f"SELECT CONCAT_WS(' ', {words}) AS n, CASE WHEN {apart} THEN 1 ELSE 0 END AS o"
        f" FROM (SELECT v, c{width}, {pieces} FROM (SELECT v, {chunks} FROM ({texts}) s) c) p"
    )
    return (
        "SELECT COUNT(DISTINCT CASE WHEN o = 0 THEN n END),"
        f" COUNT(DISTINCT CASE WHEN o = 0 AND n IN ({listed}) THEN n END),"
        f" COUNT(CASE WHEN o = 1 THEN 1 END) FROM ({normalized}) w"
    )


def _chunk(index: int) -> str:
    repeat = f"(?:{CHUNK}){{{index}}}" if index else ""
    return f"{LEADING}{repeat}([{ALNUM}]+)"


def _piece(index: int) -> str:
    if index == 0:
        return FIRST_PIECE
    repeat = f"(?:{LATER_PIECE}){{{index - 1}}}" if index > 1 else ""
    return f"^[A-Z]*[a-z0-9]+{repeat}([A-Z]+[a-z0-9]*)"


def _singular(word: str) -> str:
    last = f"SUBSTR({word}, LENGTH({word}), 1)"
    before = f"SUBSTR({word}, LENGTH({word}) - 1, 1)"
    plural = f"LENGTH({word}) >= {MIN_PLURAL_LENGTH} AND {last} = 's' AND {before} <> 's'"
    return f"CASE WHEN {plural} THEN SUBSTR({word}, 1, LENGTH({word}) - 1) ELSE {word} END"

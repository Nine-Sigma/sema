"""The SQL name comparison counts exactly what ``lexicon.normalize`` would compare."""

from __future__ import annotations

import random
from collections.abc import Collection, Iterator, Sequence

import duckdb
import pytest

from sema.intake.lexicon import normalize
from sema.intake.name_sql_utils import NameListing, listed_names_sql

pytestmark = pytest.mark.unit

EDGE_VALUES = (
    "Kind Code", "kind-label", "KindNote", "HTMLParser", "aBcDE", "ABCdEF", "1A", "a1B2",
    "abcDe", "class", "bus", "cars", "Classes", "IDs", "ID_s", "sss", "ssss", "assS",
    "  padded name  ", "--", "oneTwoThree Four-five", "A B C D E F G",
)
CORPUS_ALPHABET = "aAbBsSzZ09_- ./"
CORPUS_SIZE = 300
CORPUS_SEED = 7


@pytest.fixture
def con() -> Iterator[duckdb.DuckDBPyConnection]:
    connection = duckdb.connect(":memory:")
    connection.execute("CREATE TABLE t (v VARCHAR)")
    yield connection
    connection.close()


def _listing(
    con: duckdb.DuckDBPyConnection, values: Sequence[str | None], names: Collection[str]
) -> NameListing:
    con.execute("DELETE FROM t")
    con.executemany("INSERT INTO t VALUES (?)", [[value] for value in values])
    row = con.execute(listed_names_sql("t", '"v"', names)).fetchall()[0]
    return NameListing(*(int(value) for value in row))


def _corpus() -> list[str]:
    rng = random.Random(CORPUS_SEED)
    return [
        "".join(rng.choice(CORPUS_ALPHABET) for _ in range(rng.randint(1, 16)))
        for _ in range(CORPUS_SIZE)
    ]


@pytest.mark.parametrize("value", EDGE_VALUES)
def test_a_value_compares_equal_to_its_python_normalization(
    con: duckdb.DuckDBPyConnection, value: str
) -> None:
    name = normalize(value.strip())

    assert _listing(con, [value], {name}) == NameListing(1, 1, 0)
    assert _listing(con, [value], {f"{name} other"}).listed == 0


def test_a_printable_ascii_corpus_compares_as_python_normalizes(
    con: duckdb.DuckDBPyConnection,
) -> None:
    mismatched = [
        value
        for value in _corpus()
        if value.strip()
        and _listing(con, [value], {normalize(value.strip())}) != NameListing(1, 1, 0)
    ]

    assert mismatched == []


def test_names_listed_exactly_by_a_column(con: duckdb.DuckDBPyConnection) -> None:
    names = {"kind code", "kind label", "kind note"}

    listing = _listing(con, ["Kind Code", "kind_label", "KindNote"], names)

    assert listing.lists_exactly(3)
    assert not listing.lists_exactly(4)


def test_a_value_with_more_words_than_any_name_is_not_compared(
    con: duckdb.DuckDBPyConnection,
) -> None:
    listing = _listing(con, ["kind code extra", "kindCodeExtra"], {"kind code"})

    assert listing == NameListing(0, 0, 2)


def test_text_outside_printable_ascii_is_not_compared(con: duckdb.DuckDBPyConnection) -> None:
    listing = _listing(con, ["kind\tcode", "kïnd code"], {"kind code"})

    assert listing.uncompared == 2
    assert not listing.lists_exactly(1)


def test_null_blank_and_loader_values_are_skipped(con: duckdb.DuckDBPyConnection) -> None:
    listing = _listing(con, [None, "   ", "_sema_locator", "kind code"], {"kind code"})

    assert listing == NameListing(1, 1, 0)

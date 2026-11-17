"""The connector fixture's joins as discovery-scorecard cells, and a reader for a rendered card.

A join is keyed as the contract's oracle file format keys it: (relation, columns, referenced
relation, referenced columns), each column list its sorted names joined by `+`.
"""

from __future__ import annotations

from collections.abc import Iterable

from tests.support.connector_fixture import (
    EXPECTED_DETECTED_JOINS,
    MAIN_TABLES,
    UNDETECTABLE_DECLARED_JOIN,
    FixtureJoin,
)

Key = tuple[str, ...]


def join_key(table: str, columns: Iterable[str], referenced: str,
             referenced_columns: Iterable[str]) -> Key:
    return (table, "+".join(sorted(columns)), referenced, "+".join(sorted(referenced_columns)))


def _single(join: FixtureJoin) -> Key:
    return join_key(join.referencing_table, [join.referencing_column],
                    join.referenced_table or "", [join.referenced_column or ""])


DECLARED_JOIN_KEYS = frozenset(
    join_key(t.name, k.columns, k.referenced_table, k.referenced_columns)
    for t in MAIN_TABLES for k in t.declared_keys
)
DETECTED_JOIN_KEYS = frozenset(_single(j) for j in EXPECTED_DETECTED_JOINS)
UNDETECTED_JOIN_KEY = _single(UNDETECTABLE_DECLARED_JOIN)
COMPOSITE_JOIN_KEYS = frozenset(k for k in DECLARED_JOIN_KEYS if "+" in k[1])


def card_rows(markdown: str) -> dict[str, dict[str, str]]:
    """Each row of a rendered markdown card, its cells by header name, keyed by its first cell."""
    table = [
        [cell.strip() for cell in line.strip().strip("|").split("|")]
        for line in markdown.splitlines() if line.strip().startswith("|")
    ]
    header, body = table[0], [r for r in table[1:] if not set("".join(r)) <= set("-: ")]
    return {row[0]: dict(zip(header, row, strict=True)) for row in body}

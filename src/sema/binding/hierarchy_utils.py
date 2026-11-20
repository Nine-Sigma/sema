"""Hierarchy forms projected to (child, parent) pairs.

Each projection returns the pairs plus the implicit nodes it saw: a parent the
rows imply but never list as a code. Implicit nodes get no edge; the importer
logs them.
"""

from __future__ import annotations

from collections.abc import Iterable

from sema.binding.importer_utils import Record, as_text

Pairs = set[tuple[str, str]]


def parent_pairs_from_parent_column(
    rows: Iterable[Record], child_column: str, parent_column: str
) -> tuple[Pairs, list[str]]:
    pairs: Pairs = set()
    for row in rows:
        child, parent = as_text(row[child_column]), as_text(row[parent_column])
        if child is not None and parent is not None:
            pairs.add((child, parent))
    return pairs, []


def parent_pairs_from_levels(
    rows: Iterable[Record], level_columns: list[str]
) -> tuple[Pairs, list[str]]:
    """Adjacent non-null levels pair up; a null level ends the row's path."""
    pairs: Pairs = set()
    for row in rows:
        parent: str | None = None
        for column in level_columns:
            node = as_text(row[column])
            if node is None:
                break
            if parent is not None:
                pairs.add((node, parent))
            parent = node
    return pairs, []


def parent_pairs_from_paths(
    rows: Iterable[Record], path_column: str, separator: str
) -> tuple[Pairs, list[str]]:
    codes = {code for row in rows if (code := as_text(row[path_column])) is not None}
    pairs: Pairs = set()
    implicit: set[str] = set()
    for code in codes:
        parent, sep, _ = code.rpartition(separator)
        if not sep or not parent:
            continue
        if parent in codes:
            pairs.add((code, parent))
        else:
            implicit.add(parent)
    return pairs, sorted(implicit)

"""Invented relations for the gate-or-finding tests (S5 decisions 3, 18, 22, 23).

Every relation, column, and value is made up and lands in one in-memory DuckDB schema,
``fd``. Each source isolates one constant: the relation passes every gate of its template
and sits on the wrong side of exactly one measured floor, or fails exactly one gate.
Names that do not exist before task 3.3 are read lazily (``getattr``), so a missing one
fails its own test.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from tests.unit.intake.eligibility_env import memory_source
from tests.unit.intake.structure.shapes.shapes_env import Source

SCHEMA = "fd"
Rows = Sequence[tuple[object, ...]]

# Four groups over six rows (ratio 0.667, above the 0.5 floor); codes repeat across groups,
# so the group scopes them, and the label is a function of the group.
DENSE = "dense"
DENSE_ROWS: Rows = (
    ("G1", "Alpha set", "C1", "One"), ("G1", "Alpha set", "C2", "Two"),
    ("G2", "Beta set", "C1", "Three"), ("G3", "Gamma set", "C1", "Four"),
    ("G4", "Delta set", "C1", "Five"), ("G4", "Delta set", "C2", "Six"),
)
DENSE_COLUMNS = ("set_key", "set_label", "item_code", "item_label")
DENSE_RATIO = 4 / 6
# The same rows with a null group key: a gate, not a finding.
HOLED_ROWS: Rows = (*DENSE_ROWS[:-1], (None, "Delta set", "C2", "Six"))
# Four entities over six rows (ratio 0.667, above the 0.5 cap); names recur across them.
FIELDS = "field_list"
FIELD_ROWS: Rows = (
    ("T1", "id"), ("T1", "name"), ("T2", "id"), ("T3", "id"), ("T4", "id"), ("T4", "name"),
)
FIELD_COLUMNS = ("table_label", "field_label")
FIELD_RATIO = 4 / 6
# A code table: two groups, every code unique (property ratio 1.0, above the 0.98 floor).
KINDS = "kinds"
KIND_COLUMNS = ("kind_group", "kind_code", "kind_label", "kind_note")
KIND_ROWS: Rows = tuple(
    (f"Grp {i % 2}", f"K{i:02d}", f"Kind label {i}", f"Kind note number {i}") for i in range(1, 7)
)
# A code list: four of twenty labels empty (0.2, above the 0.05 display floor).
SPARSE = "sparse"
SPARSE_ROWS: Rows = tuple(
    (f"B{i:02d}", None if i <= 4 else f"Marrow {i}") for i in range(1, 21)
)
SPARSE_COLUMNS = ("item_code", "item_label")
SPARSE_RATE = 4 / 20
# A concept list whose definitions are short (mean 5, below the 12.0 floor).
TERSE = "terse"
TERSE_ROWS: Rows = tuple((f"Z{i:02d}", f"Name {i:02d}", f"Df {i:02d}") for i in range(1, 21))
TERSE_COLUMNS = ("concept_id", "concept_name", "definition")
TERSE_MEAN = 5.0
OPTIONS_ROWS: Rows = (("yes", "Accepted"), ("no", "Refused"))
OPTIONS_COLUMNS = ("option_value", "option_text")
# A dictionary group (key column naming the code table's columns, beside a header sheet).
DICTIONARY, HEADER = "spec__fields", "spec__about"
DICTIONARY_ROWS: Rows = tuple((name, f"The {name} of a kind", "text") for name in KIND_COLUMNS)
DICTIONARY_COLUMNS = ("field_name", "field_description", "field_type")
HEADER_ROWS: Rows = (("title", "Kinds of things"), ("owner", "A team"))
HEADER_COLUMNS = ("key", "value")


def _literal(cell: object) -> str:
    return "NULL" if cell is None else "'" + str(cell).replace("'", "''") + "'"


def create(name: str, columns: Sequence[str], rows: Rows) -> str:
    values = ", ".join("(" + ", ".join(_literal(c) for c in row) + ")" for row in rows)
    return (
        f"CREATE TABLE {SCHEMA}.{name} AS SELECT * FROM (VALUES {values})"
        f" t({', '.join(columns)})"
    )


@contextmanager
def landed(root: Path, *statements: str) -> Iterator[Source]:
    con, source = memory_source((f"CREATE SCHEMA {SCHEMA}", *statements), SCHEMA, root)
    try:
        yield source
    finally:
        con.close()


def options_groups(identical: int, groups: int) -> list[str]:
    """``groups`` files of a data sheet and an options sheet; the first ``identical`` options
    sheets hold the same rows, the rest differ."""
    statements = []
    for index in range(1, groups + 1):
        data = tuple((f"D{index}{j}", f"Datum {index} {j}") for j in range(1, 4))
        statements.append(create(f"g{index}__data", ("datum_code", "datum_label"), data))
        options = OPTIONS_ROWS if index <= identical else (("maybe", f"Option {index}"),)
        statements.append(create(f"g{index}__options", OPTIONS_COLUMNS, options))
    return statements


def qualified(name: str) -> str:
    return f"{SCHEMA}.{name}"


def findings_of(item: Any) -> list[tuple[str, Any]]:
    """``(name, value)`` of every finding on a binding or eligibility (``findings`` field)."""
    return [(found.name, found.value) for found in getattr(item, "findings")]


def stated(item: Any, name: str) -> Any:
    """A measured value named ``name`` on a binding: a finding, else a slot of that name."""
    values = [value for found, value in findings_of(item) if found == name]
    if not values and name in getattr(item, "slots", {}):
        values = [item.slots[name]]
    assert values, (name, item)
    return values[0]


def mentions(item: Any, *parts: str) -> bool:
    text = json.dumps(item.model_dump(mode="json"))
    return all(part in text for part in parts)

"""The connector fixtures' facts as discovery-scorecard cells, and a reader for a rendered card.

A join is keyed as the contract's oracle file format keys it: (relation, columns, referenced
relation, referenced columns), each column list its sorted names joined by `+`. The coded
fixture's claims are keyed (relation, column), an observed value (relation, column, value).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

from tests.support.connector_fixture import (
    EXPECTED_DETECTED_JOINS,
    MAIN_TABLES,
    UNDETECTABLE_DECLARED_JOIN,
    FixtureJoin,
)
from tests.support.connector_fixture_codes import CODE_LISTS, ROWS, column_values

Key = tuple[str, ...]

DICTIONARY_ONLY = "dictionary_only"
# The coded fixture's verified bindings, read off its rows and dictionary: a declared key, a
# name the data agrees with, or a stated domain, each contained at the floor.
BINDING_CLAIMS: dict[Key, str] = {
    ("panel", "finish_code"): "finish", ("panel", "grit_code"): "grit",
    ("fitting", "tone_code"): "tone", ("fitting", "sheen_code"): "sheen",
    ("fitting", "glaze_code"): "glaze", ("panel", "crew_ref"): "crew",
    ("panel", "coat_kind"): "coat_kinds", ("panel", "seal_kind"): "seal_kinds",
    ("fitting", "seal_kind"): "seal_kinds", ("panel", "bevel_kind"): "bevel_kinds",
}
# A code relation's one text column, else one determined sibling, else the dictionary's labels.
LABEL_CLAIMS: dict[Key, object] = {
    ("panel", "finish_code"): ["finish", "finish_label"],
    ("fitting", "tone_code"): ["fitting", "tone_name"],
    ("fitting", "glaze_code"): ["glaze", "glaze_title"],
    ("panel", "crew_ref"): ["crew", "crew_name"],
    **{key: DICTIONARY_ONLY for key, name in BINDING_CLAIMS.items() if name in CODE_LISTS},
}


def listed_codes(code_list: str) -> set[str]:
    """A code list's codes: the dictionary's, or its code relation's first column."""
    if code_list in CODE_LISTS:
        return set(CODE_LISTS[code_list])
    return {str(row[0]) for row in ROWS[code_list]}


def present_values(key: Key) -> list[str]:
    return [str(value) for value in column_values(*key) if value is not None]


COVERAGE_CLAIMS: dict[Key, float] = {
    key: sum(v in listed_codes(name) for v in present_values(key)) / len(present_values(key))
    for key, name in BINDING_CLAIMS.items()
}
OBSERVED_VALUE_CLAIMS: dict[Key, int] = {
    (*key, value): rows
    for key in BINDING_CLAIMS for value, rows in Counter(present_values(key)).items()
}
CODED_CLAIMS: dict[str, dict[Key, object]] = {
    "column_to_concept_binding": dict(BINDING_CLAIMS), "label_source": LABEL_CLAIMS,
    "mapping_coverage": dict(COVERAGE_CLAIMS), "observed_values": dict(OBSERVED_VALUE_CLAIMS),
}


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
DECLARED_JOIN_VALUES = {
    join_key(t.name, k.columns, k.referenced_table, k.referenced_columns):
        "one_to_one" if set(k.columns) == set(t.primary_key) else "many_to_one"
    for t in MAIN_TABLES for k in t.declared_keys
}
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

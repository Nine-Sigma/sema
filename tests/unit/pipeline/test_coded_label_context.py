"""Done-when 2: a question naming a label renders the column, the code and the label.

The context is built by the core query path (retrieval, then pruning to a context object,
then the SQL prompt the model reads) over a stand-in for the build graph. The stand-in holds
the contract shape for every label location at once: each coded column links to a value
set whose member `Term`s carry their labels, whether the label came from a sibling column,
a code relation reached by a join, or a dictionary only. The model is never called.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import pytest

from sema.consumers.nl2sql.prompting import build_sql_prompt
from sema.graph.queries import CypherQueries
from sema.pipeline.context import prune_to_sco
from sema.pipeline.retrieval import RetrievalEngine
from tests.support.connector_fixture_codes import (
    CODE_LISTS,
    CODES_CATALOG,
    CODES_SCHEMA,
    FINISH,
    TONE_NAMES,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.xfail(
        strict=True,
        reason="unit 5 N3 (mechanism) blocked; see unit-5/STATE.md Waiting on Dean 2",
    ),
]


@dataclass(frozen=True)
class Member:
    table: str
    column: str
    vocabulary: str
    code: str
    label: str


MEMBERS = (
    *(Member("fitting", "tone_code", "tone", c, label) for c, label in TONE_NAMES.items()),
    *(Member("panel", "finish_code", "finish", c, label) for c, label in FINISH.items()),
    *(Member("panel", "coat_kind", "coat_kinds", c, label)
      for c, label in CODE_LISTS["coat_kinds"].items()),
)


class _Record:
    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def data(self) -> dict[str, Any]:
        return self._data


class _Session:
    """Answers the term reads of the query path from `MEMBERS`; every other read is empty."""

    def __enter__(self) -> _Session:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def run(self, query: str, **params: Any) -> Iterator[_Record]:
        if query == CypherQueries.lexical_search_terms():
            token = params["token"]
            return iter([
                _Record({"node": {
                    "code": m.code, "label": m.label, "vocabulary_name": m.vocabulary,
                    "confidence": 0.95, "status": "auto",
                }})
                for m in MEMBERS if token in m.label.lower()
            ])
        if query == CypherQueries.find_value_sets_for_term():
            vocabulary = params.get("vocabulary_name")
            return iter([
                _Record({
                    "value_set_name": f"{m.table}_{m.column}_values", "column_name": m.column,
                    "table_name": m.table, "schema_name": CODES_SCHEMA, "catalog": CODES_CATALOG,
                })
                for m in MEMBERS
                if m.code == params["code"] and vocabulary in (None, m.vocabulary)
            ])
        return iter([])


class _Driver:
    def session(self, **_: Any) -> _Session:
        return _Session()


LOCATIONS = {
    "sibling label column": ("How many fittings are Cobalt?", "tone_code", "T-BL", "Cobalt"),
    "label relation by join": ("Count the Brushed panels", "finish_code", "BRU", "Brushed"),
    "dictionary only": ("Which panels carry a Lacquer coat?", "coat_kind", "CK-C", "Lacquer"),
}


@pytest.mark.parametrize("location", list(LOCATIONS))
def test_the_context_for_a_label_renders_its_column_code_and_label(location: str) -> None:
    question, column, code, label = LOCATIONS[location]
    candidates = RetrievalEngine(driver=_Driver()).retrieve(question)
    prompt = build_sql_prompt(prune_to_sco(candidates), question)
    column_lines = [line for line in prompt.splitlines() if column in line and code in line]
    assert column_lines, f"no line of the context names {column} with {code}:\n{prompt}"
    assert any(label in line for line in column_lines), (
        f"the context names {column} and {code} without the label {label!r}: {column_lines}"
    )

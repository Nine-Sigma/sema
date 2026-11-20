"""Neutral graph cases for the four context defect classes, over the connector fixture's relations.

`CaseGraph` stands in for the graph the core query path reads. It answers each read
`RetrievalEngine` makes from the facts declared here and returns no rows for any other read,
so a test states a graph and a question and reads the context the product renders. Vector
hits are what a nearest-neighbour index returns: the closest nodes for any question, whether
or not the question names them. Entity and property names follow the fixture's mocked model,
which names an entity after its table and a property after its column.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from sema.graph.queries import CypherQueries
from sema.pipeline.retrieval import RetrievalEngine
from tests.support.connector_fixture import CATALOG, MAIN_SCHEMA, MAIN_TABLES

TERM_INDEX = "term_embedding_index"
ENTITY_INDEX = "entity_embedding_index"
PROPERTY_INDEX = "property_embedding_index"
_VECTOR_INDEXES = (ENTITY_INDEX, PROPERTY_INDEX, TERM_INDEX,
                   "alias_embedding_index", "metric_embedding_index")
_TOP_K_RANGE = range(1, 51)
_CONFIDENCE = 0.9


@dataclass(frozen=True)
class CaseTerm:
    code: str
    label: str
    parent: str | None = None


GOODS_VOCABULARY = "goods_family"
GOODS_TERMS = (
    CaseTerm("CT-T", "Tools"),
    CaseTerm("CT-A", "Hand tools", "CT-T"),
    CaseTerm("CT-B", "Power tools", "CT-T"),
    CaseTerm("CT-M", "Materials"),
    CaseTerm("CT-C", "Paint", "CT-M"),
    CaseTerm("CT-D", "Glue", "CT-M"),
)
GOODS_COLUMN = ("product", "category_code")
UNASKED_HIERARCHY_QUESTION = "How many transfers were made in March?"
ASKED_HIERARCHY_QUESTION = "How many products are hand tools?"

REGION_COLUMN = ("customer", "region_code")
LABELLED_REGIONS = (("RG-N", "North"), ("RG-S", "South"))
UNLABELLED_REGION = "RG-W"
VALUES_QUESTION = "How many customers live in the North?"


@dataclass(frozen=True)
class CaseColumn:
    table: str
    column: str
    definition: str


NAMED_COLUMN = CaseColumn("transfer", "amount", "the sum moved from one account to another")
DEFINED_COLUMN = CaseColumn(
    "price_period", "unit_price",
    "the amount a product costs per unit while its price period is valid",
)
NAMED_TERM_QUESTION = "What is the total amount of all transfers?"
DEFINITION_ONLY_QUESTION = "What does a product cost?"
PROPERTY_SCORES: dict[str, dict[CaseColumn, float]] = {
    NAMED_TERM_QUESTION: {DEFINED_COLUMN: 0.93, NAMED_COLUMN: 0.71},
    DEFINITION_ONLY_QUESTION: {DEFINED_COLUMN: 0.88, NAMED_COLUMN: 0.40},
}
ENTITY_HITS: dict[str, tuple[str, ...]] = {
    NAMED_TERM_QUESTION: (NAMED_COLUMN.table, DEFINED_COLUMN.table),
}
# One score for every entity hit, so no score gate keeps one relation and drops another.
ENTITY_SCORE = 0.8
CLOSEST_SCORE = 0.95

RESULT_COLUMNS = ("region_code", "customer_count")
SHAPE_QUESTION = "How many customers are in each region?"
SHAPE_SQL = (
    f"SELECT region_code, COUNT(*) AS customer_count "
    f"FROM {CATALOG}.{MAIN_SCHEMA}.customer GROUP BY region_code"
)

CATEGORICAL = {GOODS_COLUMN, REGION_COLUMN}
DEFINITIONS = {(c.table, c.column): c.definition for c in (NAMED_COLUMN, DEFINED_COLUMN)}


def entity_name(table: str) -> str:
    return table.replace("_", " ").title()


def property_name(column: str) -> str:
    return column.replace("_", " ")


def _members(table: str, column: str) -> list[tuple[str, str, str]]:
    """(code, label, vocabulary) of each value-set member; an unlabelled member has label ''."""
    if (table, column) == GOODS_COLUMN:
        return [(t.code, t.label, GOODS_VOCABULARY) for t in GOODS_TERMS if t.parent]
    if (table, column) == REGION_COLUMN:
        return [(c, label, "") for c, label in (*LABELLED_REGIONS, (UNLABELLED_REGION, ""))]
    return []


class _Record:
    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def data(self) -> dict[str, Any]:
        return self._data


class CaseGraph:
    """A driver whose sessions answer the core query path's reads from the declared facts."""

    def __init__(self, vector_hits: Mapping[str, list[tuple[dict[str, Any], float]]]) -> None:
        self._vector = {
            CypherQueries.vector_search(index, k): vector_hits.get(index, [])
            for index in _VECTOR_INDEXES for k in _TOP_K_RANGE
        }

    def session(self, **_: Any) -> CaseGraph:
        return self

    def __enter__(self) -> CaseGraph:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def run(self, query: str, **params: Any) -> Iterator[_Record]:
        if query in self._vector:
            return iter([_Record({"node": n, "score": s}) for n, s in self._vector[query]])
        answer = _READS.get(query)
        return iter([_Record(row) for row in answer(params)] if answer else [])


def _token_hits(nodes: list[dict[str, Any]], field: str, token: str) -> list[dict[str, Any]]:
    return [{"node": n} for n in nodes if token in str(n[field]).lower()]


def term_node(term: CaseTerm) -> dict[str, Any]:
    return {"code": term.code, "label": term.label, "vocabulary_name": GOODS_VOCABULARY,
            "confidence": _CONFIDENCE, "status": "auto"}


def entity_node(table: str) -> dict[str, Any]:
    return {"name": entity_name(table), "confidence": _CONFIDENCE, "status": "auto"}


def property_node(table: str, column: str) -> dict[str, Any]:
    node = {"name": property_name(column), "entity_name": entity_name(table),
            "confidence": _CONFIDENCE, "status": "auto"}
    definition = DEFINITIONS.get((table, column))
    return {**node, "description": definition} if definition else node


def _tables() -> dict[str, tuple[str, ...]]:
    return {t.name: t.column_names for t in MAIN_TABLES}


def _physical(params: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {"table_name": table, "schema_name": MAIN_SCHEMA, "catalog": CATALOG, "columns": [
            {"property": property_name(c), "column": c, "data_type": "string",
             "semantic_type": "categorical" if (table, c) in CATEGORICAL else "free_text"}
            for c in columns
        ]}
        for table, columns in _tables().items() if entity_name(table) == params["entity_name"]
    ]


def _ancestors(params: Mapping[str, Any]) -> list[dict[str, Any]]:
    by_code = {t.code: t for t in GOODS_TERMS}
    rows, term = [], by_code.get(params["code"])
    while term is not None and term.parent and params.get("vocabulary_name") in (
        None, GOODS_VOCABULARY,
    ):
        parent = by_code[term.parent]
        rows.append({"code": parent.code, "label": parent.label,
                     "vocabulary_name": GOODS_VOCABULARY, "parent_code": params["code"]})
        term = parent
    return rows


def _value_sets_for_term(params: Mapping[str, Any]) -> list[dict[str, Any]]:
    table, column = GOODS_COLUMN
    return [
        {"value_set_name": f"{table}_{column}_values", "column_name": column,
         "table_name": table, "schema_name": MAIN_SCHEMA, "catalog": CATALOG}
        for code, _, vocabulary in _members(table, column)
        if code == params["code"] and params.get("vocabulary_name") in (None, vocabulary)
    ]


def _column_members(params: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [{"code": code, "label": label, "vocabulary_name": vocabulary or None}
            for code, label, vocabulary in _members(params["table_name"], params["column_name"])]


def _lexical_properties(params: Mapping[str, Any]) -> list[dict[str, Any]]:
    nodes = [property_node(t, c) for t, columns in _tables().items() for c in columns]
    return _token_hits(nodes, "name", params["token"])


_READS: dict[str, Any] = {
    CypherQueries.lexical_search_terms(): lambda p: _token_hits(
        [term_node(t) for t in GOODS_TERMS], "label", p["token"]),
    CypherQueries.lexical_search_entities(): lambda p: _token_hits(
        [entity_node(t) for t in _tables()], "name", p["token"]),
    CypherQueries.lexical_search_properties(): _lexical_properties,
    CypherQueries.resolve_physical_mapping(): _physical,
    CypherQueries.expand_ancestry(): _ancestors,
    CypherQueries.find_value_sets_for_term(): _value_sets_for_term,
    CypherQueries.find_value_set_members_by_column(): _column_members,
}


class CaseEmbedder:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.0, 1.0] for _ in texts]


def case_engine(graph: CaseGraph) -> RetrievalEngine:
    return RetrievalEngine(driver=graph, embedder=CaseEmbedder())


def hierarchy_graph() -> CaseGraph:
    """The goods hierarchy, whose nearest term to any question is a leaf of it.

    The leaf is the closest hit of any index, above the entity hit the context must keep, so no
    vector-score cutoff drops it: only whether the question names a concept of it can.
    """
    nearest = next(t for t in GOODS_TERMS if t.parent)
    return CaseGraph({TERM_INDEX: [(term_node(nearest), CLOSEST_SCORE)],
                      ENTITY_INDEX: [(entity_node("transfer"), ENTITY_SCORE)]})


def ranking_graph(question: str) -> CaseGraph:
    """Two columns a question term can bind to, scored by how close each definition is."""
    hits = [(property_node(c.table, c.column), score)
            for c, score in PROPERTY_SCORES[question].items()]
    entities = [(entity_node(table), ENTITY_SCORE) for table in ENTITY_HITS.get(question, ())]
    return CaseGraph({PROPERTY_INDEX: sorted(hits, key=lambda hit: -hit[1]),
                      ENTITY_INDEX: entities})


def lines_naming(text: str, token: str) -> list[str]:
    """Every line of ``text`` that names ``token`` as a whole word."""
    pattern = re.compile(rf"(?<![\w-]){re.escape(token)}(?![\w-])")
    return [line for line in text.splitlines() if pattern.search(line)]


def hierarchy_edge_lines(text: str, terms: tuple[CaseTerm, ...]) -> list[str]:
    """Lines that name a term together with its parent, by code or by label."""
    by_code = {t.code: t for t in terms}

    def names(term: CaseTerm, line: str) -> bool:
        return bool(lines_naming(line, term.code) or lines_naming(line, term.label))

    return [
        line for line in text.splitlines()
        if any(t.parent and names(t, line) and names(by_code[t.parent], line) for t in terms)
    ]


def ranks_before_alternative(line: str, first: str, alternative: str) -> bool:
    """``first`` is named before the word "alternative", and ``alternative`` after it."""
    marker = line.lower().find("alternative")
    if marker < 0:
        return False
    return bool(lines_naming(line[:marker], first)) and bool(
        lines_naming(line[marker:], alternative))

"""Done-when 5 (N7): identity and labels in the core query context.

Two levels. The context object with the new fields (contract F7) is rendered by
`render_sections`, and each line is asserted word for word. Then the core query path
(retrieval, `prune_to_sco`, `render_sections`) runs over a stand-in graph whose physical
mapping rows carry the F6 read forms: the `ENTITY_ON_TABLE` edge's `owner`, `key_columns`,
`verification`, `identity` (the `EntityIdentity` JSON) and `null_key_rows`, and the table's
`table_key_columns` and `table_key_verification`. The mapping read returns every edge, an
`owner = false` edge included, so the reader rule is the query path's own.
"""

from __future__ import annotations

import importlib
import re
from collections.abc import Iterator
from typing import Any

import pytest

from sema.consumers.nl2sql.prompting import build_sql_prompt, render_sections
from sema.graph.queries import CypherQueries
from sema.models.context import (
    JoinPath,
    JoinPredicate,
    PhysicalAsset,
    Provenance,
    ResolvedEntity,
    ResolvedProperty,
    SemanticContextObject,
)
from sema.models.target.identity import EntityIdentity, ResolvedAttribute, ResolutionRule
from sema.pipeline.context import prune_to_sco
from sema.pipeline.retrieval import RetrievalEngine
from showcase.ontology_query_benchmark.core_s_provider import without_denied
from tests.support.connector_fixture_context import ENTITY_INDEX, CaseEmbedder, entity_node
from tests.support.connector_fixture_identity import IDENTITY_CATALOG, IDENTITY_SCHEMA, TABLES

pytestmark = pytest.mark.unit

DIALECT = "duckdb"
SECTION = "KEYS AND ENTITIES:"
MARKERS = ("one row per", "per distinct", "picks one row", "label:")
PROVENANCE = Provenance(source="retrieval", confidence=0.8)


def fqn(table: str) -> str:
    return f"{IDENTITY_CATALOG}.{IDENTITY_SCHEMA}.{table}"


def _context_models() -> Any:
    return importlib.import_module("sema.models.context")


def _latest(column: str, by: str) -> ResolvedAttribute:
    return ResolvedAttribute(column=column, rule=ResolutionRule.MAX_BY, by=by)


HULL_IDENTITY = EntityIdentity(key_columns=("hull_no",),
                               attributes=(_latest("paint_band", "rev_no"),))
CRANE_IDENTITY = EntityIdentity(key_columns=("crane_no",),
                                attributes=(_latest("lift_tons", "rev_no"),))
BUOY_IDENTITY = EntityIdentity(key_columns=("buoy_no",))
RIG_JOIN = JoinPredicate(left_table="rig_kind", left_column="rig_code", right_table="hull_rev",
                         right_column="rig_type")


def _asset(table: str, **fields: Any) -> PhysicalAsset:
    return PhysicalAsset(catalog=IDENTITY_CATALOG, schema=IDENTITY_SCHEMA, table=table,
                         columns=list(TABLES[table].column_names), **fields)


def _property(table: str, column: str, **fields: Any) -> ResolvedProperty:
    return ResolvedProperty(name=column.replace("_", " "), semantic_type="free_text",
                            physical_column=column, physical_table=fqn(table),
                            provenance=PROVENANCE, **fields)


def _entity(name: str, table: str, properties: list[ResolvedProperty] | None = None,
            **fields: Any) -> ResolvedEntity:
    return ResolvedEntity(name=name, physical_table=fqn(table), properties=properties or [],
                          provenance=PROVENANCE, **fields)


def identity_sco() -> SemanticContextObject:
    """A key, three entities (an ordering rule, NULL key rows, no rule), a shared entity, a
    label in another relation and a label beside its code."""
    models = _context_models()
    rig_label = models.LabelSource(table="rig_kind", column="rig_title", via=[RIG_JOIN])
    sibling = models.LabelSource(table="dock_kind", column="dock_title", via=[])
    shared = models.SharedEntity(entity="Hull", owner_table=fqn("hull_rev"),
                                 key_columns=["hull_no"])
    return SemanticContextObject(
        physical_assets=[
            _asset("hull_rev", key_columns=["hull_no", "rev_no"]), _asset("rig_kind"),
            _asset("hull_survey", key_columns=["survey_no"], shared_entities=[shared]),
            _asset("crane_rev"), _asset("buoy_check"), _asset("dock_kind"),
        ],
        entities=[
            _entity("Hull", "hull_rev", [_property("hull_rev", "rig_type", labels=[rig_label])],
                    key_columns=["hull_no"], identity=HULL_IDENTITY),
            _entity("Crane", "crane_rev", key_columns=["crane_no"], identity=CRANE_IDENTITY,
                    null_key_rows=2),
            _entity("Buoy", "buoy_check", key_columns=["buoy_no"], identity=BUOY_IDENTITY),
            _entity("dock kind", "dock_kind",
                    [_property("dock_kind", "dock_code", labels=[sibling])]),
        ],
        join_paths=[JoinPath(from_table="hull_rev", to_table="rig_kind",
                             join_predicates=[RIG_JOIN], confidence=0.9)],
    )


EXPECTED_SECTION = [
    f"{fqn('hull_rev')}: one row per hull_no, rev_no",
    f"{fqn('hull_survey')}: one row per survey_no",
    f"{fqn('hull_survey')}: rows belong to Hull of {fqn('hull_rev')} through hull_no",
    f"{fqn('hull_rev')}: several rows can hold one Hull; one Hull per distinct hull_no; "
    "each Hull has exactly one row per rev_no, so the greatest rev_no picks one row of each",
    f"{fqn('crane_rev')}: several rows can hold one Crane; one Crane per distinct crane_no; "
    "2 rows have a NULL in crane_no; each Crane has exactly one row per rev_no, so the "
    "greatest rev_no picks one row of each",
    f"{fqn('buoy_check')}: several rows can hold one Buoy; one Buoy per distinct buoy_no",
]


def section_lines(lines: list[str]) -> list[str]:
    """The stripped lines of the `KEYS AND ENTITIES` section, empty when it is absent."""
    text = "\n".join(lines).splitlines()
    if SECTION not in [line.strip() for line in text]:
        return []
    start = [line.strip() for line in text].index(SECTION) + 1
    body = []
    for line in text[start:]:
        if not line.strip() or line.strip().endswith(":") and line.strip().isupper():
            break
        body.append(line.strip())
    return body


def headers(lines: list[str]) -> list[str]:
    return [line.strip() for line in "\n".join(lines).splitlines()
            if line.strip().endswith(":") and line.strip().split(" (")[0].isupper()]


def table_line(lines: list[str], table: str) -> str:
    (found,) = [line for line in "\n".join(lines).splitlines()
                if line.strip().startswith(f"{fqn(table)}: ") and "one row per" not in line
                and "several rows" not in line and "rows belong" not in line]
    return found


def test_the_section_states_each_key_entity_and_shared_entity_word_for_word() -> None:
    lines = render_sections(identity_sco(), DIALECT)

    assert sorted(section_lines(lines)) == sorted(EXPECTED_SECTION)


def test_the_section_follows_join_paths() -> None:
    order = headers(render_sections(identity_sco(), DIALECT))

    assert order[order.index("JOIN PATHS:") + 1] == SECTION


def test_every_marker_is_in_the_context() -> None:
    text = "\n".join(render_sections(identity_sco(), DIALECT))

    assert SECTION in text
    assert all(marker in text for marker in MARKERS), [m for m in MARKERS if m not in text]


def test_a_label_in_another_relation_is_on_its_column_line_with_the_join() -> None:
    line = table_line(render_sections(identity_sco(), DIALECT), "hull_rev")

    assert "label: rig_kind.rig_title on rig_kind.rig_code = hull_rev.rig_type" in line


def test_a_label_beside_its_code_names_its_column_and_no_join() -> None:
    line = table_line(render_sections(identity_sco(), DIALECT), "dock_kind")

    assert "label: dock_kind.dock_title" in line
    assert "label: dock_kind.dock_title on" not in line


def test_every_table_and_column_a_label_line_names_is_spelled_in_it() -> None:
    line = table_line(render_sections(identity_sco(), DIALECT), "hull_rev")
    named = re.findall(r"(\w+)\.(\w+)", line.split("label:", 1)[1])

    for table, column in named:
        assert column in TABLES[table].column_names, (table, column)
    assert ("rig_kind", "rig_code") in named and ("hull_rev", "rig_type") in named


def test_no_sentence_states_a_null_policy_a_function_or_a_current_row() -> None:
    body = " ".join(section_lines(render_sections(identity_sco(), DIALECT))).lower()

    assert body
    assert "current" not in body and "latest" not in body
    assert not re.search(r"\w\(", body), body


def test_no_cut_removes_the_section() -> None:
    prompt = build_sql_prompt(identity_sco(), "Which hull?", dialect=DIALECT, max_chars=1)

    assert SECTION in prompt


def new_text(lines: list[str]) -> str:
    """Every line the unit adds: the section, and each label fragment of a column line."""
    labels = [line.split("label:", 1)[1] for line in "\n".join(lines).splitlines()
              if "label:" in line]
    return "\n".join([*section_lines(lines), *labels])


@pytest.mark.parametrize("denied", ["hull_no", "rev_no", "rig_title", "rig_code", "crane_no"])
def test_a_denied_column_appears_in_no_new_line(denied: str) -> None:
    sco = identity_sco()
    assert denied in new_text(render_sections(sco, DIALECT))

    kept = without_denied(sco, lambda column: column == denied)

    assert not re.search(rf"\b{denied}\b", new_text(render_sections(kept, DIALECT)))


BASE_SCO_RENDER = [
    "\nAVAILABLE TABLES AND COLUMNS:",
    "  fixident.yard.hull_rev: hull_no, rev_no, paint_band, keel_class, rig_type",
    "  fixident.yard.rig_kind: rig_code, kind_group, rig_title, rig_note",
    "\nJOIN PATHS:",
    "  hull_rev -> rig_kind ON rig_kind.rig_code = hull_rev.rig_type (unknown)",
    "\nDIALECT NOTES (DuckDB SQL):",
    "- Use the table names as listed: catalog.schema.table",
    "- Use double quotes for an identifier that is a reserved word or holds a character other "
    "than letters, digits or underscores",
]


def test_a_context_with_none_of_the_fields_renders_as_the_base_tip() -> None:
    sco = SemanticContextObject(
        physical_assets=[_asset("hull_rev"), _asset("rig_kind")],
        entities=[_entity("Hull", "hull_rev")],
        join_paths=[JoinPath(from_table="hull_rev", to_table="rig_kind",
                             join_predicates=[RIG_JOIN], confidence=0.9)],
    )

    assert render_sections(sco, DIALECT) == BASE_SCO_RENDER


class _Record:
    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def data(self) -> dict[str, Any]:
        return self._data


class MappingGraph:
    """Answers the entity vector search with `hits` and each entity's physical mapping with
    `mapping`'s rows; every other read is empty."""

    def __init__(self, mapping: dict[str, list[dict[str, Any]]], hits: list[str]) -> None:
        self._mapping = mapping
        self._vector = {CypherQueries.vector_search(ENTITY_INDEX, k):
                        [(entity_node(name), 0.8) for name in hits] for k in range(1, 51)}

    def session(self, **_: Any) -> MappingGraph:
        return self

    def __enter__(self) -> MappingGraph:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def run(self, query: str, **params: Any) -> Iterator[_Record]:
        if query in self._vector:
            return iter([_Record({"node": n, "score": s}) for n, s in self._vector[query]])
        if query == CypherQueries.resolve_physical_mapping():
            return iter([_Record(row) for row in self._mapping.get(params["entity_name"], [])])
        return iter([])


def mapping_row(table: str, **read_forms: Any) -> dict[str, Any]:
    columns = [{"property": c.replace("_", " "), "column": c, "data_type": "string",
                "semantic_type": "free_text"} for c in TABLES[table].column_names]
    return {"table_name": table, "schema_name": IDENTITY_SCHEMA, "catalog": IDENTITY_CATALOG,
            "columns": columns, **read_forms}


def owner_edge(identity: EntityIdentity | None, verification: str = "proven",
               **more: Any) -> dict[str, Any]:
    return {"owner": True, "key_columns": list(identity.key_columns) if identity else None,
            "verification": verification,
            "identity": identity.model_dump_json() if identity else None, **more}


def context(graph: MappingGraph, question: str) -> list[str]:
    retrieved = RetrievalEngine(driver=graph, embedder=CaseEmbedder()).retrieve(question)
    return render_sections(prune_to_sco(retrieved), DIALECT)


def assets(lines: list[str]) -> set[str]:
    listed = "\n".join(lines).split("AVAILABLE TABLES AND COLUMNS:", 1)[1].split("\n\n", 1)[0]
    return {line.strip().split(":", 1)[0].rsplit(".", 1)[-1]
            for line in listed.splitlines() if line.strip()}


QUESTION = "Which crane lifts the most beside each hull?"
HULL_KEY = {"table_key_columns": ["hull_no", "rev_no"], "table_key_verification": "proven"}


def hull_rows(**hull_rev: Any) -> list[dict[str, Any]]:
    """The owner edge, an edge L2 wrote before the step, and a step edge with owner false."""
    return [
        mapping_row("hull_rev", **{**owner_edge(HULL_IDENTITY), **HULL_KEY, **hull_rev}),
        mapping_row("hull_cargo", key_columns=["hull_no"], verification="proven"),
        mapping_row("hull_survey", owner=False, key_columns=["hull_no"], verification="proven"),
    ]


def test_an_owner_false_edge_brings_no_table_and_an_l2_edge_still_maps() -> None:
    lines = context(MappingGraph({"Hull": hull_rows()}, ["Hull"]), QUESTION)

    assert assets(lines) == {"hull_rev", "hull_cargo"}


def test_the_graph_read_forms_reach_the_section() -> None:
    crane = mapping_row("crane_rev", **owner_edge(CRANE_IDENTITY, "supported", null_key_rows=2))
    graph = MappingGraph({"Hull": hull_rows(), "Crane": [crane]}, ["Hull", "Crane"])

    body = section_lines(context(graph, QUESTION))

    assert f"{fqn('hull_rev')}: one row per hull_no, rev_no" in body
    assert EXPECTED_SECTION[3] in body and EXPECTED_SECTION[4] in body


@pytest.mark.parametrize("grade", ["open", "rejected"])
def test_an_open_or_rejected_fact_renders_nothing(grade: str) -> None:
    hull = hull_rows(table_key_verification=grade)[0] | owner_edge(HULL_IDENTITY, grade)
    crane = mapping_row("crane_rev", **owner_edge(CRANE_IDENTITY, "supported", null_key_rows=2))
    graph = MappingGraph({"Hull": [hull], "Crane": [crane]}, ["Hull", "Crane"])

    body = section_lines(context(graph, QUESTION))

    assert not [line for line in body if line.startswith(f"{fqn('hull_rev')}:")]
    assert EXPECTED_SECTION[4] in body, "a supported fact beside it still renders"


BASE_GRAPH_RENDER = [
    "\nAVAILABLE TABLES AND COLUMNS:",
    "  fixident.yard.hull_rev: hull_no, rev_no, paint_band, keel_class, rig_type",
    "  fixident.yard.hull_cargo: cargo_no, hull_no, cargo_tons, hatch_kind",
    "  fixident.yard.crane_rev: crane_rev_id, crane_no, rev_no, lift_tons",
    "\nDIALECT NOTES (DuckDB SQL):",
    "- Use the table names as listed: catalog.schema.table",
    "- Use double quotes for an identifier that is a reserved word or holds a character other "
    "than letters, digits or underscores",
]


def test_a_graph_with_none_of_the_properties_renders_as_the_base_tip() -> None:
    mapping = {"Hull": [mapping_row("hull_rev"), mapping_row("hull_cargo")],
               "Crane": [mapping_row("crane_rev")]}

    assert context(MappingGraph(mapping, ["Hull", "Crane"]), QUESTION) == BASE_GRAPH_RENDER


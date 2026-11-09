"""Arm S context: retrieval output rendered with bindings, values, grain, rules."""

from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path
from typing import Any

import pytest

from sema.models.context import SemanticCandidateSet
from showcase.ontology_query_benchmark.sema_bindings import (
    load_products,
    load_sidecar,
    resolve_bindings,
)
from showcase.ontology_query_benchmark import sema_live
from showcase.ontology_query_benchmark.sema_context import SContextProvider
from showcase.ontology_query_benchmark.profile import Profile
from showcase.ontology_query_benchmark.routing import RoutingError
from showcase.ontology_query_benchmark.sema_trace import trace
from tests.unit.ontology_query_benchmark.test_sema_bindings import (
    CONVENTIONS,
    PEOPLE,
    RID,
    EVENTS,
    catalog_fixture,
    facts_fixture,
    write_fixture,
)

pytestmark = pytest.mark.unit

QUESTION = "How many living people are female (administrative gender)?"


def _col(name: str, semantic_type: str = "free_text") -> dict[str, str]:
    return {"property": name, "column": name, "data_type": "string",
            "semantic_type": semantic_type}


def _entity(table: str, columns: list[dict[str, str]]) -> dict[str, Any]:
    return {"type": "entity", "name": table, "table": table, "schema": "sch",
            "catalog": "cat", "confidence": 0.8, "status": "auto",
            "confidence_policy": "semantic", "source": "retrieval", "columns": columns}


def _value(table: str, column: str, code: str, label: str) -> dict[str, Any]:
    return {"type": "value", "property_name": column, "column": column, "table": table,
            "code": code, "label": label, "status": "auto", "confidence": 0.8,
            "confidence_policy": "semantic"}


PEOPLE_ENTITY = _entity(PEOPLE, [
    _col(RID), _col("gender_G1", "categorical"), _col("gender_G1_label"),
    _col("vital_V1", "categorical"), _col("city_C1", "categorical"),
])
EVENTS_ENTITY = _entity(EVENTS, [_col("row_id"), _col(RID), _col("site_S1", "categorical")])
PEOPLE_VALUES = [
    _value(PEOPLE, "gender_G1", "L8532", "Female"),
    _value(PEOPLE, "gender_G1", "L8507", "Male"),
    _value(PEOPLE, "vital_V1", "L1", "Alive"),
    _value(PEOPLE, "vital_V1", "L2", "Deceased"),
]


class _Retriever:
    def __init__(self, candidates: list[dict[str, Any]]) -> None:
        self.candidates = candidates
        self.questions: list[str] = []

    def retrieve(self, question: str, top_k: int = 10) -> SemanticCandidateSet:
        self.questions.append(question)
        return SemanticCandidateSet(query=question, candidates=list(self.candidates))


def _provider(
    tmp_path: Path,
    candidates: list[dict[str, Any]],
    *,
    conventions: tuple[str, ...] = CONVENTIONS,
    budget: int | None = None,
    list_limit: int | None = None,
    graph_identity: str = "graph-1",
) -> SContextProvider:
    sidecar = load_sidecar(write_fixture(tmp_path), tmp_path)
    if budget is not None:
        sidecar = dataclasses.replace(sidecar, context_char_budget=budget)
    if list_limit is not None:
        sidecar = dataclasses.replace(sidecar, value_list_limit=list_limit)
    resolution = resolve_bindings(sidecar, load_products(sidecar), catalog_fixture())
    return SContextProvider(
        _Retriever(candidates), sidecar, load_products(sidecar), resolution,
        facts_fixture(tmp_path), graph_identity=graph_identity,
        conventions=conventions,
    )


def test_context_renders_assets_bindings_and_complete_values(tmp_path: Path) -> None:
    ctx = _provider(tmp_path, [PEOPLE_ENTITY, *PEOPLE_VALUES])(QUESTION)
    assert f"cat.sch.{PEOPLE}" in ctx.text
    assert "gender_G1" in ctx.text and "Administrative gender" in ctx.text
    assert "L8532 = Female" in ctx.text and "L8507 = Male" in ctx.text
    assert "L1 = Alive" in ctx.text
    assert "Spec test authority" in ctx.text
    assert "V_label holds its label" in ctx.text


def test_incompletely_retrieved_value_set_is_not_listed(tmp_path: Path) -> None:
    ctx = _provider(tmp_path, [PEOPLE_ENTITY, *PEOPLE_VALUES[1:]])(QUESTION)
    assert "L8507 = Male" not in ctx.text
    assert "membership query G1" in ctx.text
    assert any("G1" in f and "incomplete" in f for f in ctx.findings)


def test_rejected_values_are_never_rendered(tmp_path: Path) -> None:
    rejected = [{**v, "status": "rejected"} for v in PEOPLE_VALUES]
    ctx = _provider(tmp_path, [PEOPLE_ENTITY, *rejected])(QUESTION)
    assert "L8532 = Female" not in ctx.text
    assert "L1 = Alive" not in ctx.text


def test_large_value_set_points_at_membership_relation(tmp_path: Path) -> None:
    ctx = _provider(tmp_path, [EVENTS_ENTITY], list_limit=2)("How many events by site?")
    assert "lib.sl.data_element" in ctx.text
    assert "membership query S1" in ctx.text and "3 members" in ctx.text
    assert not any("S1" in f for f in ctx.findings)


def test_value_list_limit_switches_to_membership_reference(tmp_path: Path) -> None:
    ctx = _provider(tmp_path, [PEOPLE_ENTITY, *PEOPLE_VALUES], list_limit=1)(QUESTION)
    assert "L8532 = Female" not in ctx.text
    assert "membership query G1" in ctx.text
    assert not any("G1" in f for f in ctx.findings)


def test_catalog_companion_is_rendered_as_unsupplied(tmp_path: Path) -> None:
    event = _entity(EVENTS, [*EVENTS_ENTITY["columns"], _col("site_S1_concept_code")])
    ctx = _provider(tmp_path, [event])("How many events by site code?")
    assert "site_S1_concept_code" in ctx.text
    line = next(x for x in ctx.text.splitlines() if "site_S1 <-" in x)
    assert "not in SPEC" in line and "site_S1_concept_code" in line


def test_pattern_companions_are_not_listed_twice(tmp_path: Path) -> None:
    ctx = _provider(tmp_path, [PEOPLE_ENTITY, *PEOPLE_VALUES])(QUESTION)
    assert "gender_G1_label" not in ctx.text
    table_line = next(x for x in ctx.text.splitlines() if x.startswith(f"  cat.sch.{PEOPLE}: "))
    assert "gender_G1" in table_line and RID in table_line


def test_membership_query_template_is_stated_once(tmp_path: Path) -> None:
    ctx = _provider(tmp_path, [PEOPLE_ENTITY, *PEOPLE_VALUES], list_limit=1)(QUESTION)
    assert ctx.text.count("SELECT member_id") == 1
    assert "membership query G1" in ctx.text and "membership query V1" in ctx.text


def test_absent_library_elements_are_findings(tmp_path: Path) -> None:
    ctx = _provider(tmp_path, [PEOPLE_ENTITY, *PEOPLE_VALUES])(QUESTION)
    assert any(f.startswith("library element absent: C1") for f in ctx.findings)
    assert any(f.startswith("library element absent: N1") for f in ctx.findings)


def test_overflow_is_reported_without_truncating(tmp_path: Path) -> None:
    full = _provider(tmp_path / "a", [PEOPLE_ENTITY, *PEOPLE_VALUES])(QUESTION)
    tight = _provider(tmp_path / "b", [PEOPLE_ENTITY, *PEOPLE_VALUES], budget=200)(QUESTION)
    assert tight.text == full.text
    assert any("overflow" in f for f in tight.findings)
    assert not any("overflow" in f for f in full.findings)


def test_conventions_are_referenced_not_paraphrased(tmp_path: Path) -> None:
    ctx = _provider(tmp_path, [EVENTS_ENTITY])("How many distinct events?")
    assert not any(rule in ctx.text for rule in CONVENTIONS)
    assert "conventions" in ctx.text.lower()
    mismatch = _provider(tmp_path, [EVENTS_ENTITY], conventions=("other",))("q")
    assert any("conventions" in f for f in mismatch.findings)


def test_grain_and_hierarchy_reference_are_rendered(tmp_path: Path) -> None:
    ctx = _provider(tmp_path, [EVENTS_ENTITY])("How many distinct events?")
    assert f"{RID}, site_S1" in ctx.text
    assert "row_id" in ctx.text and "surrogate" in ctx.text
    assert "ancestor_concept_id is broader" in ctx.text
    assert "TOY HIERARCHY REFERENCE (supplied):" in ctx.text
    assert "executable" not in ctx.text


def test_identity_covers_text_and_graph(tmp_path: Path) -> None:
    one = _provider(tmp_path / "a", [PEOPLE_ENTITY, *PEOPLE_VALUES])(QUESTION)
    two = _provider(tmp_path / "b", [PEOPLE_ENTITY, *PEOPLE_VALUES],
                    graph_identity="graph-2")(QUESTION)
    assert one.identity != two.identity
    assert one.identity == hashlib.sha256(f"{one.text}\ngraph-1".encode()).hexdigest()


def test_no_retrieved_asset_is_a_finding(tmp_path: Path) -> None:
    ctx = _provider(tmp_path, [])(QUESTION)
    assert any("no physical asset" in f for f in ctx.findings)


def test_trace_links_library_binding_asset_and_context(tmp_path: Path) -> None:
    provider = _provider(tmp_path, [PEOPLE_ENTITY, *PEOPLE_VALUES])
    rows = {row["physical"]["column"]: row for row in trace(provider, QUESTION)}
    gender = rows["gender_G1"]
    assert gender["library"] == {"de_id": "G1", "library_id": "gender-v1",
                                 "label": "Administrative gender"}
    assert gender["binding"]["status"] == "approved_supplied"
    assert gender["binding"]["spec_sha256"]
    assert gender["physical"]["relation"] == f"cat.sch.{PEOPLE}"
    assert gender["context"] == {"retrieved": True, "rendered": True,
                                 "values_listed": 2}
    assert "derived_x" not in rows


def test_live_host_guard_fails_closed(profile: Profile) -> None:
    class _Workspace:
        class config:
            host = "https://personal.example.test"

    with pytest.raises(RoutingError, match="host"):
        sema_live.embedder(_Workspace, profile.routing)


def test_embedder_resolved_host_is_checked(
    profile: Profile, monkeypatch: pytest.MonkeyPatch
) -> None:
    import databricks_langchain
    from mlflow.utils import databricks_utils

    class _Embeddings:
        def __init__(self, endpoint: str, target_uri: str) -> None:
            self.target_uri = target_uri

    class _Creds:
        host = "https://personal.example.test"

    class _Workspace:
        class config:
            host = profile.routing.host

    monkeypatch.setattr(databricks_langchain, "DatabricksEmbeddings", _Embeddings)
    monkeypatch.setattr(databricks_utils, "get_databricks_host_creds", lambda uri: _Creds)
    with pytest.raises(RoutingError, match="host"):
        sema_live.embedder(_Workspace, profile.routing)

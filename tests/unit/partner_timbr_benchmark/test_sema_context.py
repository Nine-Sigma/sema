"""Arm S context: retrieval output rendered with bindings, values, grain, rules."""

from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path
from typing import Any

import pytest

from sema.models.context import SemanticCandidateSet
from showcase.partner_timbr_benchmark.sema_bindings import (
    load_products,
    load_sidecar,
    resolve_bindings,
)
from showcase.partner_timbr_benchmark import sema_live
from showcase.partner_timbr_benchmark.sema_context import SContextProvider
from showcase.partner_timbr_benchmark.sema_live import WrongWorkspaceError, assert_test_host
from showcase.partner_timbr_benchmark.sema_trace import trace
from tests.unit.partner_timbr_benchmark.test_sema_bindings import (
    CONVENTIONS,
    DEMO,
    MRN,
    TUMOR,
    catalog_fixture,
    facts_fixture,
    write_fixture,
)

pytestmark = pytest.mark.unit

Q01 = "How many living patients are female (administrative gender)?"


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


DEMO_ENTITY = _entity(DEMO, [
    _col(MRN), _col("gender_G1", "categorical"), _col("gender_G1_label"),
    _col("vital_V1", "categorical"), _col("city_C1", "categorical"),
])
TUMOR_ENTITY = _entity(TUMOR, [_col("row_id"), _col(MRN), _col("site_S1", "categorical")])
DEMO_VALUES = [
    _value(DEMO, "gender_G1", "VF", "Female"),
    _value(DEMO, "gender_G1", "VM", "Male"),
    _value(DEMO, "vital_V1", "VA", "Alive"),
    _value(DEMO, "vital_V1", "VD", "Deceased"),
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
    sidecar = load_sidecar(write_fixture(tmp_path))
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
    ctx = _provider(tmp_path, [DEMO_ENTITY, *DEMO_VALUES])(Q01)
    assert f"cat.sch.{DEMO}" in ctx.text
    assert "gender_G1" in ctx.text and "Administrative gender" in ctx.text
    assert "VF = Female" in ctx.text and "VM = Male" in ctx.text
    assert "VA = Alive" in ctx.text
    assert "CPD test authority" in ctx.text
    assert "V_label holds its label" in ctx.text


def test_incompletely_retrieved_value_set_is_not_listed(tmp_path: Path) -> None:
    ctx = _provider(tmp_path, [DEMO_ENTITY, *DEMO_VALUES[1:]])(Q01)
    assert "VM = Male" not in ctx.text
    assert "membership query G1" in ctx.text
    assert any("G1" in f and "incomplete" in f for f in ctx.findings)


def test_rejected_values_are_never_rendered(tmp_path: Path) -> None:
    rejected = [{**v, "status": "rejected"} for v in DEMO_VALUES]
    ctx = _provider(tmp_path, [DEMO_ENTITY, *rejected])(Q01)
    assert "VF = Female" not in ctx.text
    assert "VA = Alive" not in ctx.text


def test_large_value_set_points_at_membership_relation(tmp_path: Path) -> None:
    ctx = _provider(tmp_path, [TUMOR_ENTITY], list_limit=2)("How many tumors by site?")
    assert "lib.sl.data_element" in ctx.text
    assert "membership query S1" in ctx.text and "3 members" in ctx.text
    assert not any("S1" in f for f in ctx.findings)


def test_value_list_limit_switches_to_membership_reference(tmp_path: Path) -> None:
    ctx = _provider(tmp_path, [DEMO_ENTITY, *DEMO_VALUES], list_limit=1)(Q01)
    assert "VF = Female" not in ctx.text
    assert "membership query G1" in ctx.text
    assert not any("G1" in f for f in ctx.findings)


def test_catalog_companion_is_rendered_as_unsupplied(tmp_path: Path) -> None:
    tumor = _entity(TUMOR, [*TUMOR_ENTITY["columns"], _col("site_S1_concept_code")])
    ctx = _provider(tmp_path, [tumor])("How many tumors by site code?")
    assert "site_S1_concept_code" in ctx.text
    line = next(x for x in ctx.text.splitlines() if "site_S1 <-" in x)
    assert "not in CPD" in line and "site_S1_concept_code" in line


def test_pattern_companions_are_not_listed_twice(tmp_path: Path) -> None:
    ctx = _provider(tmp_path, [DEMO_ENTITY, *DEMO_VALUES])(Q01)
    assert "gender_G1_label" not in ctx.text
    table_line = next(x for x in ctx.text.splitlines() if x.startswith(f"  cat.sch.{DEMO}: "))
    assert "gender_G1" in table_line and MRN in table_line


def test_membership_query_template_is_stated_once(tmp_path: Path) -> None:
    ctx = _provider(tmp_path, [DEMO_ENTITY, *DEMO_VALUES], list_limit=1)(Q01)
    assert ctx.text.count("SELECT permissible_value_prt_id") == 1
    assert "membership query G1" in ctx.text and "membership query V1" in ctx.text


def test_committed_budget_fits_the_first_prompt_limit() -> None:
    from showcase.partner_timbr_benchmark.sema_bindings import load_sidecar as load
    from showcase.partner_timbr_benchmark.sema_context import (
        CHARS_PER_TOKEN,
        FIRST_PROMPT_TOKEN_LIMIT,
        HARNESS_OVERHEAD_CHARS,
    )
    budget = load().context_char_budget
    assert budget + HARNESS_OVERHEAD_CHARS <= FIRST_PROMPT_TOKEN_LIMIT * CHARS_PER_TOKEN


def test_absent_library_elements_are_findings(tmp_path: Path) -> None:
    ctx = _provider(tmp_path, [DEMO_ENTITY, *DEMO_VALUES])(Q01)
    assert any(f.startswith("library element absent: C1") for f in ctx.findings)
    assert any(f.startswith("library element absent: N1") for f in ctx.findings)


def test_overflow_is_reported_without_truncating(tmp_path: Path) -> None:
    full = _provider(tmp_path / "a", [DEMO_ENTITY, *DEMO_VALUES])(Q01)
    tight = _provider(tmp_path / "b", [DEMO_ENTITY, *DEMO_VALUES], budget=200)(Q01)
    assert tight.text == full.text
    assert any("overflow" in f for f in tight.findings)
    assert not any("overflow" in f for f in full.findings)


def test_conventions_are_referenced_not_paraphrased(tmp_path: Path) -> None:
    ctx = _provider(tmp_path, [TUMOR_ENTITY])("How many distinct tumors?")
    assert not any(rule in ctx.text for rule in CONVENTIONS)
    assert "conventions" in ctx.text.lower()
    mismatch = _provider(tmp_path, [TUMOR_ENTITY], conventions=("other",))("q")
    assert any("conventions" in f for f in mismatch.findings)


def test_tumor_grain_and_hierarchy_reference_are_rendered(tmp_path: Path) -> None:
    ctx = _provider(tmp_path, [TUMOR_ENTITY])("How many distinct tumors?")
    assert f"{MRN}, site_S1" in ctx.text
    assert "row_id" in ctx.text and "surrogate" in ctx.text
    assert "ancestor_concept_id is broader" in ctx.text
    assert "HIERARCHY REFERENCE (supplied):" in ctx.text
    assert "executable" not in ctx.text


def test_committed_hierarchy_reference_holds_relation_facts_only() -> None:
    from showcase.partner_timbr_benchmark.sema_bindings import load_sidecar as load
    text = "\n".join(load().hierarchy_reference)
    assert "Delta version" in text
    assert "SELECT" not in text.upper() and "src_partner_wh" not in text
    assert "Maps to" in text


def test_identity_covers_text_and_graph(tmp_path: Path) -> None:
    one = _provider(tmp_path / "a", [DEMO_ENTITY, *DEMO_VALUES])(Q01)
    two = _provider(tmp_path / "b", [DEMO_ENTITY, *DEMO_VALUES],
                    graph_identity="graph-2")(Q01)
    assert one.identity != two.identity
    assert one.identity == hashlib.sha256(f"{one.text}\ngraph-1".encode()).hexdigest()


def test_no_retrieved_asset_is_a_finding(tmp_path: Path) -> None:
    ctx = _provider(tmp_path, [])(Q01)
    assert any("no physical asset" in f for f in ctx.findings)


def test_q01_trace_links_library_binding_asset_and_context(tmp_path: Path) -> None:
    provider = _provider(tmp_path, [DEMO_ENTITY, *DEMO_VALUES])
    rows = {row["physical"]["column"]: row for row in trace(provider, Q01)}
    gender = rows["gender_G1"]
    assert gender["library"] == {"de_id": "G1", "library_id": "gender-v1",
                                 "label": "Administrative gender"}
    assert gender["binding"]["status"] == "approved_supplied"
    assert gender["binding"]["spec_sha256"]
    assert gender["physical"]["relation"] == f"cat.sch.{DEMO}"
    assert gender["context"] == {"retrieved": True, "rendered": True,
                                 "values_listed": 2}
    assert "derived_x" not in rows


def test_live_host_guard_fails_closed() -> None:
    assert_test_host("https://partner-workspace.cloud.databricks.com/")
    with pytest.raises(WrongWorkspaceError):
        assert_test_host("https://dbc-6395977d.cloud.databricks.com")


def test_embedder_resolved_host_is_checked(monkeypatch: pytest.MonkeyPatch) -> None:
    import databricks_langchain
    from mlflow.utils import databricks_utils

    class _Embeddings:
        def __init__(self, endpoint: str, target_uri: str) -> None:
            self.target_uri = target_uri

    class _Creds:
        host = "https://dbc-6395977d.cloud.databricks.com"

    class _Workspace:
        class config:
            host = sema_live.TEST_HOST

    monkeypatch.setattr(databricks_langchain, "DatabricksEmbeddings", _Embeddings)
    monkeypatch.setattr(databricks_utils, "get_databricks_host_creds", lambda uri: _Creds)
    with pytest.raises(WrongWorkspaceError):
        sema_live.embedder(_Workspace)

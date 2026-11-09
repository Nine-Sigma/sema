"""Arm S context: retrieval output rendered with bindings, values, grain, rules."""

from __future__ import annotations

import dataclasses
import hashlib
import math
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

from sema.models.context import SemanticCandidateSet
from showcase.ontology_query_benchmark.sema_bindings import (
    load_products,
    load_sidecar,
    resolve_bindings,
)
from showcase.ontology_query_benchmark import runner_utils, sema_live, sema_trace
from showcase.ontology_query_benchmark.product_spec import SpecExposure
from showcase.ontology_query_benchmark.profile import Attribution, Limits, Profile
from showcase.ontology_query_benchmark.routing import RoutingError
from showcase.ontology_query_benchmark.sema_build import BuildManifest, write_manifest
from showcase.ontology_query_benchmark.sema_build_utils import import_library_facts
from showcase.ontology_query_benchmark.sema_context import (
    ManifestMismatchError,
    SContextProvider,
)
from showcase.ontology_query_benchmark.sema_trace import open_live_provider, trace
from tests.unit.ontology_query_benchmark.conftest import PROFILE_PATH
from tests.unit.ontology_query_benchmark.test_sema_bindings import (
    CONCEPT_SPEC,
    BINDING_NOTE,
    CONVENTIONS,
    EVENTS,
    ID_SUFFIX,
    PEOPLE,
    PROMPT_FRAME_CHARS,
    RID,
    catalog_fixture,
    facts_fixture,
    write_fixture,
    write_projection,
)
from tests.unit.ontology_query_benchmark.test_sema_build import (
    ClientBuilt,
    clean_process,
    no_client,
)

if TYPE_CHECKING:
    from showcase.ontology_query_benchmark.sema_build_utils import Finding

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
    _col(RID), _col("gender_G1", "categorical"), _col("gender_G1_text"),
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
    limits: Limits | None = None,
    list_limit: int | None = None,
    graph_identity: str = "graph-1",
    denied: tuple[str, ...] | None = None,
) -> SContextProvider:
    sidecar = load_sidecar(write_fixture(tmp_path), tmp_path)
    if list_limit is not None:
        sidecar = dataclasses.replace(sidecar, value_list_limit=list_limit)
    products = load_products(sidecar)
    if denied is not None:
        exposure = SpecExposure(denied_columns=denied)
        products = tuple(
            dataclasses.replace(p, doc=p.doc.model_copy(update={"exposure": exposure}))
            if p.pin.table == PEOPLE else p
            for p in products
        )
    resolution = resolve_bindings(sidecar, products, catalog_fixture())
    return SContextProvider(
        _Retriever(candidates), sidecar, products, resolution,
        facts_fixture(tmp_path), graph_identity=graph_identity,
        conventions=conventions, limits=limits or LIMITS,
    )


LIMITS = Limits(
    input_ceiling=16_000, repair_reserve=2_066, output_ceiling=8_192, guard_chars_per_token=2.334,
    conservative_chars_per_token=2.08, row_limit=1_000, max_repairs=2,
)


def _first_prompt(tokens: int) -> Limits:
    return LIMITS.model_copy(update={"input_ceiling": tokens + LIMITS.repair_reserve})


def _overflows(findings: tuple[Finding, ...]) -> bool:
    return any(f.message.startswith("overflow:") for f in findings)


def test_context_renders_assets_bindings_and_complete_values(tmp_path: Path) -> None:
    ctx = _provider(tmp_path, [PEOPLE_ENTITY, *PEOPLE_VALUES])(QUESTION)
    assert f"cat.sch.{PEOPLE}" in ctx.text
    assert "gender_G1" in ctx.text and "Administrative gender" in ctx.text
    assert "L8532 = Female" in ctx.text and "L8507 = Male" in ctx.text
    assert "L1 = Alive" in ctx.text
    assert "Spec test authority" in ctx.text


def test_the_binding_note_is_the_sidecar_text(tmp_path: Path) -> None:
    ctx = _provider(tmp_path, [PEOPLE_ENTITY, *PEOPLE_VALUES])(QUESTION)
    header = next(x for x in ctx.text.splitlines() if x.startswith("APPROVED BINDINGS"))
    assert header.endswith(f"). {BINDING_NOTE}")


def test_incompletely_retrieved_value_set_is_not_listed(tmp_path: Path) -> None:
    ctx = _provider(tmp_path, [PEOPLE_ENTITY, *PEOPLE_VALUES[1:]])(QUESTION)
    assert "L8507 = Male" not in ctx.text
    assert "membership query G1" in ctx.text
    assert [(f.relation, f.column) for f in ctx.findings if "incomplete" in f.message] == [
        (f"cat.sch.{PEOPLE}", "gender_G1")
    ]


def test_rejected_values_are_never_rendered(tmp_path: Path) -> None:
    rejected = [{**v, "status": "rejected"} for v in PEOPLE_VALUES]
    ctx = _provider(tmp_path, [PEOPLE_ENTITY, *rejected])(QUESTION)
    assert "L8532 = Female" not in ctx.text
    assert "L1 = Alive" not in ctx.text


def test_large_value_set_points_at_membership_relation(tmp_path: Path) -> None:
    ctx = _provider(tmp_path, [EVENTS_ENTITY], list_limit=2)("How many events by site?")
    assert "lib.sl.data_element" in ctx.text
    assert "membership query S1" in ctx.text and "3 members" in ctx.text
    assert not any("S1" in f.message for f in ctx.findings)


def test_value_list_limit_switches_to_membership_reference(tmp_path: Path) -> None:
    ctx = _provider(tmp_path, [PEOPLE_ENTITY, *PEOPLE_VALUES], list_limit=1)(QUESTION)
    assert "L8532 = Female" not in ctx.text
    assert "membership query G1" in ctx.text
    assert not any("G1" in f.message for f in ctx.findings)


def test_catalog_companion_is_rendered_as_unsupplied(tmp_path: Path) -> None:
    event = _entity(EVENTS, [*EVENTS_ENTITY["columns"], _col("site_S1_concept_code")])
    ctx = _provider(tmp_path, [event])("How many events by site code?")
    assert "site_S1_concept_code" in ctx.text
    line = next(x for x in ctx.text.splitlines() if "site_S1 <-" in x)
    assert "not in SPEC" in line and "site_S1_concept_code" in line


def _table_line(text: str, table: str) -> str:
    return next(x for x in text.splitlines() if x.startswith(f"  cat.sch.{table}: "))


def _unbound_line(text: str, table: str) -> str:
    return next(x for x in text.splitlines() if x.startswith(f"  cat.sch.{table} no governed"))


def test_declared_companion_columns_are_listed(tmp_path: Path) -> None:
    ctx = _provider(tmp_path, [PEOPLE_ENTITY, *PEOPLE_VALUES])(QUESTION)
    assert "gender_G1_text" in _table_line(ctx.text, PEOPLE)
    assert "no governed binding" not in ctx.text


def test_an_unbound_catalog_column_is_listed_and_marked(tmp_path: Path) -> None:
    event = _entity(EVENTS, [*EVENTS_ENTITY["columns"], _col("ended_on", "temporal")])
    ctx = _provider(tmp_path, [event])("How long after the event did it end?")
    assert "ended_on (temporal)" in _table_line(ctx.text, EVENTS)
    assert _unbound_line(ctx.text, EVENTS).endswith(": ended_on")


def test_unbound_columns_leave_the_binding_lines_unchanged(tmp_path: Path) -> None:
    unbound = _entity(EVENTS, [*EVENTS_ENTITY["columns"], _col("ended_on")])
    with_unbound = _provider(tmp_path, [unbound])("Events by site?").text
    without = _provider(tmp_path, [EVENTS_ENTITY])("Events by site?").text
    def bindings(text: str) -> str:
        return text[text.index("\nAPPROVED BINDINGS"):]
    assert bindings(with_unbound) == bindings(without)


def test_denied_columns_are_never_rendered(tmp_path: Path) -> None:
    people = _entity(PEOPLE, [*PEOPLE_ENTITY["columns"], _col("first_name")])
    with_denied = _provider(tmp_path, [people, EVENTS_ENTITY, *PEOPLE_VALUES])(QUESTION)
    without = _provider(tmp_path, [PEOPLE_ENTITY, EVENTS_ENTITY, *PEOPLE_VALUES])(QUESTION)
    assert "first_name" not in with_denied.text
    assert with_denied.text == without.text


def test_a_denied_bound_column_has_no_binding_or_values(tmp_path: Path) -> None:
    provider = _provider(tmp_path, [PEOPLE_ENTITY, *PEOPLE_VALUES], denied=("gender_G1",))
    text = provider(QUESTION).text
    assert "gender_G1 " not in text and "gender_G1 <-" not in text
    assert "L8532 = Female" not in text
    assert "gender_G1_text" in _table_line(text, PEOPLE)
    assert "L1 = Alive" in text


def test_membership_query_template_is_stated_once(tmp_path: Path) -> None:
    ctx = _provider(tmp_path, [PEOPLE_ENTITY, *PEOPLE_VALUES], list_limit=1)(QUESTION)
    assert ctx.text.count("SELECT member_id") == 1
    assert "membership query G1" in ctx.text and "membership query V1" in ctx.text


def test_absent_library_elements_are_findings(tmp_path: Path) -> None:
    ctx = _provider(tmp_path, [PEOPLE_ENTITY, *PEOPLE_VALUES])(QUESTION)
    absent = {f.message.split()[3]: f for f in ctx.findings if "element absent" in f.message}
    assert absent.keys() == {"C1", "N1"}
    assert (absent["C1"].relation, absent["C1"].column) == (f"cat.sch.{PEOPLE}", "city_C1")


def test_overflow_is_reported_without_truncating(tmp_path: Path) -> None:
    full = _provider(tmp_path / "a", [PEOPLE_ENTITY, *PEOPLE_VALUES])(QUESTION)
    tight = _provider(tmp_path / "b", [PEOPLE_ENTITY, *PEOPLE_VALUES],
                      limits=_first_prompt(100))(QUESTION)
    assert tight.text == full.text
    assert _overflows(tight.findings)
    assert not _overflows(full.findings)


def test_the_context_budget_follows_the_first_prompt_limit(tmp_path: Path) -> None:
    """The budget fills the first-prompt limit, less the prompt frame and the question."""
    candidates = [PEOPLE_ENTITY, *PEOPLE_VALUES]
    size = len(_provider(tmp_path / "a", candidates)(QUESTION).text)
    needed = (size + PROMPT_FRAME_CHARS + len(QUESTION)) / LIMITS.conservative_chars_per_token
    roomy = _provider(tmp_path / "b", candidates, limits=_first_prompt(math.ceil(needed) + 1))
    tight = _provider(tmp_path / "c", candidates, limits=_first_prompt(math.floor(needed) - 1))
    assert not _overflows(roomy(QUESTION).findings)
    assert _overflows(tight(QUESTION).findings)


def test_an_over_budget_context_reaches_the_harness_as_overflow(
    tmp_path: Path, harness: SimpleNamespace
) -> None:
    provider = _provider(tmp_path, [PEOPLE_ENTITY, *PEOPLE_VALUES], limits=_first_prompt(100))
    assert runner_utils.SArmContext(harness, provider, 1_000)(QUESTION).overflow


def test_conventions_are_referenced_not_paraphrased(tmp_path: Path) -> None:
    ctx = _provider(tmp_path, [EVENTS_ENTITY])("How many distinct events?")
    assert not any(rule in ctx.text for rule in CONVENTIONS)
    assert "conventions" in ctx.text.lower()
    mismatch = _provider(tmp_path, [EVENTS_ENTITY], conventions=("other",))("q")
    assert [f.column for f in mismatch.findings if "conventions" in f.message] == [None]


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
    assert [f.column for f in ctx.findings if "no physical asset" in f.message] == [None]


def test_trace_links_library_binding_asset_and_context(tmp_path: Path) -> None:
    provider = _provider(tmp_path, [PEOPLE_ENTITY, *PEOPLE_VALUES])
    rows = {row["physical"]["column"]: row for row in trace(provider, QUESTION)}
    gender = rows["gender_G1"]
    assert gender["library"] == {"de_id": "G1", "library_id": f"gender{ID_SUFFIX}",
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


def test_trace_entry_refuses_a_foreign_route_before_reading_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clean_process(monkeypatch)
    no_client(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "set")
    with pytest.raises(RoutingError, match="OPENAI_API_KEY"):
        sema_trace.main([
            "--profile", str(PROFILE_PATH), "--neo4j-uri", "bolt://unused",
            "--manifest", str(tmp_path / "manifest.json"),
            "--projection", str(tmp_path / "projection.duckdb"),
            "--projection-receipt", str(tmp_path / "receipt.json"), "--question", QUESTION,
        ])


def _live_inputs(tmp_path: Path, profile: Profile) -> tuple[Profile, BuildManifest, Path, Path]:
    """A profile on the test sidecar, its projection, and the manifest those inputs build."""
    sema = profile.sema.model_copy(update={"sidecar": write_fixture(tmp_path)})
    harness = profile.harness.model_copy(update={"root": tmp_path})
    local = profile.model_copy(update={"sema": sema, "harness": harness})
    projection, receipt = write_projection(tmp_path / "library")
    sidecar = load_sidecar(sema.sidecar, tmp_path)
    facts = import_library_facts(
        CONCEPT_SPEC, projection, receipt, spec_attribution=Attribution.ASSISTED
    )
    manifest = BuildManifest(
        graph_identity="graph-1", model_role=sidecar.model_role,
        sidecar_version=sidecar.version, sidecar_sha256=sidecar.sha256, spec_sha256=(),
        facts_sha256=facts.sha256, facts_release=facts.release, embedder_model="toy-embed",
        dimensions=8,
    )
    return local, manifest, projection, receipt


@pytest.mark.parametrize("pinned", ["sidecar_sha256", "facts_sha256"])
def test_a_graph_built_from_other_inputs_is_refused_before_any_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, profile: Profile, pinned: str
) -> None:
    clean_process(monkeypatch)
    no_client(monkeypatch)
    local, manifest, projection, receipt = _live_inputs(tmp_path, profile)
    path = tmp_path / "manifest.json"
    other: dict[str, Any] = {pinned: "0" * 64}
    write_manifest(dataclasses.replace(manifest, **other), path)
    with pytest.raises(ManifestMismatchError):
        with open_live_provider(local, "bolt://unused", path, projection, receipt):
            pass


def test_a_graph_built_from_these_inputs_goes_on_to_the_clients(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, profile: Profile
) -> None:
    clean_process(monkeypatch)
    no_client(monkeypatch)
    local, manifest, projection, receipt = _live_inputs(tmp_path, profile)
    write_manifest(manifest, tmp_path / "manifest.json")
    with pytest.raises(ClientBuilt):
        with open_live_provider(local, "bolt://unused", tmp_path / "manifest.json",
                                projection, receipt):
            pass

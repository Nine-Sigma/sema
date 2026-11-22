"""Arm S context: retrieval output rendered with bindings, values, grain, rules."""

from __future__ import annotations

import dataclasses
import hashlib
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from showcase.ontology_query_benchmark import runner_utils
from showcase.ontology_query_benchmark.product_spec import ProductSpecDoc
from showcase.ontology_query_benchmark.sema_bindings import (
    load_products,
    load_sidecar,
    resolve_bindings,
)
from showcase.ontology_query_benchmark.sema_context import SContextProvider
from showcase.ontology_query_benchmark.sema_trace import trace
from tests.unit.ontology_query_benchmark.sema_bindings_env import (
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
)
from tests.unit.ontology_query_benchmark.sema_context_env import (
    EVENTS_ENTITY,
    LIMITS,
    PEOPLE_ENTITY,
    PEOPLE_VALUES,
    QUESTION,
    _Retriever,
    _col,
    _entity,
    _first_prompt,
    _overflows,
    _provider,
)

pytestmark = pytest.mark.unit


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
    provider = _provider(tmp_path, [PEOPLE_ENTITY, *PEOPLE_VALUES], denied={PEOPLE: ("gender_G1",)})
    text = provider(QUESTION).text
    assert "gender_G1 " not in text and "gender_G1 <-" not in text
    assert "L8532 = Female" not in text
    assert "gender_G1_text" in _table_line(text, PEOPLE)
    assert "L1 = Alive" in text


def test_denied_companion_columns_never_appear(tmp_path: Path) -> None:
    event = _entity(EVENTS, [*EVENTS_ENTITY["columns"], _col("site_S1_concept_code")])
    denied = {PEOPLE: ("gender_G1_text",), EVENTS: ("site_S1_concept_code",)}
    provider = _provider(tmp_path, [PEOPLE_ENTITY, event, *PEOPLE_VALUES], denied=denied)
    text = provider(QUESTION).text
    assert "gender_G1_text" not in text and "site_S1_concept_code" not in text
    assert "gender_G1 <-" in text and "site_S1 <-" in text
    assert "L8532 = Female" in text


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
    deny_nothing = harness.models.ColumnPolicy().is_denied
    assert runner_utils.SArmContext(harness, provider, 1_000, deny_nothing)(QUESTION).overflow


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


ACRIS_MASTER = "acris_real_property_master"
ACRIS_MASTER_DOC = ProductSpecDoc.model_validate({
    "product_id": "acris_master",
    "fields": [
        {"column_id": "document-id", "column_name": "document_id",
         "physical_type": "STRING", "strategy": "scalar"},
        {"column_id": "doc-type", "column_name": "doc_type", "physical_type": "STRING",
         "strategy": "pass_through", "label_column": "doc_type_description"},
    ],
})  # fmt: skip


def _display_context(
    tmp_path: Path,
    candidates: list[dict[str, Any]],
    *,
    without_display: str | None = None,
    acris: bool = False,
) -> str:
    """The rendered context after dropping one field's display companion or adding ACRIS."""
    sidecar = load_sidecar(write_fixture(tmp_path), tmp_path)
    products = load_products(sidecar)
    if without_display is not None:
        products = tuple(
            dataclasses.replace(p, doc=p.doc.model_copy(update={"fields": tuple(
                f.model_copy(update={"label_column": None}) if f.column_name == without_display
                else f for f in p.doc.fields
            )}))
            for p in products
        )
    catalog = catalog_fixture()
    if acris:
        pin = dataclasses.replace(products[0].pin, product="acris_master", table=ACRIS_MASTER)
        products = (*products, dataclasses.replace(products[0], pin=pin, doc=ACRIS_MASTER_DOC))
        catalog[ACRIS_MASTER] = dict.fromkeys(
            ("document_id", "doc_type", "doc_type_description"), "string"
        )
    resolution = resolve_bindings(sidecar, products, catalog)
    provider = SContextProvider(
        _Retriever(candidates), sidecar, products, resolution, facts_fixture(tmp_path),
        graph_identity="graph-1", conventions=CONVENTIONS, limits=LIMITS,
    )
    return provider(QUESTION).text


def test_a_coded_column_states_the_column_that_displays_it(tmp_path: Path) -> None:
    text = _display_context(tmp_path, [PEOPLE_ENTITY, *PEOPLE_VALUES])
    assert text.count("gender_G1 displays as gender_G1_text") == 1
    assert text.count("vital_V1 displays as vital_V1_label") == 1


def test_a_coded_column_without_a_display_companion_states_none(tmp_path: Path) -> None:
    """The display column comes from the declared role, never from a column-name pattern."""
    text = _display_context(tmp_path, [PEOPLE_ENTITY, *PEOPLE_VALUES], without_display="vital_V1")
    assert "gender_G1 displays as gender_G1_text" in text
    assert "vital_V1 displays as" not in text


def test_an_acris_document_type_displays_as_its_description(tmp_path: Path) -> None:
    master = _entity(ACRIS_MASTER, [
        _col("document_id"), _col("doc_type", "categorical"), _col("doc_type_description"),
    ])  # fmt: skip
    text = _display_context(tmp_path, [master], acris=True)
    assert f"cat.sch.{ACRIS_MASTER}" in text and "doc_type <-" in text
    assert text.count("doc_type displays as doc_type_description") == 1
    assert "document_id displays as" not in text

"""Shared setup for test_sema_context.py and the tests that reuse it."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sema.models.context import SemanticCandidateSet
from showcase.ontology_query_benchmark.product_spec import SpecExposure
from showcase.ontology_query_benchmark.profile import Limits
from showcase.ontology_query_benchmark.sema_bindings import (
    load_products,
    load_sidecar,
    resolve_bindings,
)
from showcase.ontology_query_benchmark.sema_context import SContextProvider
from tests.unit.ontology_query_benchmark.sema_bindings_env import (
    CONVENTIONS,
    EVENTS,
    PEOPLE,
    RID,
    catalog_fixture,
    facts_fixture,
    write_fixture,
)

if TYPE_CHECKING:
    from showcase.ontology_query_benchmark.sema_build_utils import Finding

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
    denied: Mapping[str, tuple[str, ...]] | None = None,
    grain_note: str | None = None,
    **fixture: Any,
) -> SContextProvider:
    sidecar = load_sidecar(write_fixture(tmp_path, **fixture), tmp_path)
    if grain_note is not None:
        sidecar = dataclasses.replace(sidecar, grain_note=grain_note)
    if list_limit is not None:
        sidecar = dataclasses.replace(sidecar, value_list_limit=list_limit)
    products = load_products(sidecar)
    if denied is not None:
        products = tuple(
            dataclasses.replace(p, doc=p.doc.model_copy(update={
                "exposure": SpecExposure(denied_columns=denied.get(p.pin.table, ()))
            }))
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

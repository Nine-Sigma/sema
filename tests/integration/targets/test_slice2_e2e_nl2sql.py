"""S2-10 end-to-end acceptance (the demo): a natural-language question yields
valid SQL over the materialized target, via the UNCHANGED retrieval engine +
NL2SQLConsumer and a deterministic fake LLM.

Q1 (target term "glioblastoma") exercises the target layer + value index.
Q2 (the SAME question phrased with a source OncoTree code) forces the
warehouse→ontology join through MAPS_TO_CONCEPT and must resolve to the
IDENTICAL concept. The fake LLM emits SQL as a function of the retrieved
context, so asserting on the SQL is an assertion about retrieval.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import sqlglot

from sema.consumers.base import ConsumerDeps, ConsumerRequest
from sema.consumers.nl2sql.consumer import NL2SQLConsumer
from sema.graph.concept_source import ConceptAncestor, ConceptDetail
from sema.graph.concept_value_set import (
    ValueSetColumn,
    materialize_concept_value_set,
)
from sema.graph.cross_layer_bridge import write_value_bridge
from sema.graph.loader import GraphLoader
from sema.graph.physical_catalog import (
    PhysicalColumn,
    StaticPhysicalCatalogSource,
)
from sema.graph.target_join_paths import materialize_target_join_paths
from sema.graph.target_retrieval_loader import (
    EntityTableBinding,
    PhysicalSchemaBinding,
    project_target_to_retrieval_graph,
)
from sema.pipeline.context import prune_to_sco
from sema.pipeline.retrieval import RetrievalEngine, RetrievalScope
from sema.targets.adapters.manifest import ManifestTargetAdapter
from sema.targets.normalizer import TargetModelNormalizer

pytestmark = pytest.mark.integration

_MANIFEST = (
    Path(__file__).resolve().parents[3]
    / "showcase" / "cbioportal_to_omop" / "manifests"
    / "omop_condition_slice0.yaml"
)
_CATALOG = "workspace"
_SCHEMA = "omop_stage_a"
_GBM = "4001458"
_DISTRACTOR = "4001459"
_ONCOTREE_CODE = "GBM"


class _FakeConcepts:
    _NAMES = {_GBM: "Glioblastoma multiforme", _DISTRACTOR: "Astrocytoma"}

    def name_synonyms(self, code: str) -> ConceptDetail | None:
        name = self._NAMES.get(code)
        return ConceptDetail(code, name) if name else None

    def ancestors(self, code: str) -> list[ConceptAncestor]:
        return []


def _int(name: str) -> PhysicalColumn:
    return PhysicalColumn(name, "BIGINT", False)


def _load(driver) -> None:
    normalized = TargetModelNormalizer.normalize(ManifestTargetAdapter(_MANIFEST))
    binding = PhysicalSchemaBinding(
        catalog=_CATALOG, schema=_SCHEMA,
        entity_tables=(
            EntityTableBinding("omop.person", "person"),
            EntityTableBinding("omop.condition_occurrence", "condition_occurrence"),
        ),
    )
    catalog = StaticPhysicalCatalogSource({
        (_CATALOG, _SCHEMA, "person"): [_int("person_id")],
        (_CATALOG, _SCHEMA, "condition_occurrence"): [
            _int("condition_occurrence_id"), _int("person_id"),
            _int("condition_concept_id"),
            PhysicalColumn("condition_start_date", "DATE", True),
        ],
    })
    loader = GraphLoader(driver)
    project_target_to_retrieval_graph(loader, normalized, binding, catalog)
    materialize_target_join_paths(loader, normalized, binding)
    materialize_concept_value_set(
        loader,
        column=ValueSetColumn(
            _CATALOG, _SCHEMA, "condition_occurrence", "condition_concept_id",
        ),
        concept_vocabulary="OMOP",
        concept_codes=[_GBM, _DISTRACTOR],
        concepts=_FakeConcepts(), source_schema=_SCHEMA,
    )
    # Source OncoTree term (label = its code, as a user would ask it) + bridge.
    loader.upsert_term(
        _ONCOTREE_CODE, _ONCOTREE_CODE, source="build", confidence=1.0,
        source_schema="cbioportal_gbm", vocabulary_name="OncoTree",
    )
    write_value_bridge(
        loader, source_vocabulary="OncoTree", source_code=_ONCOTREE_CODE,
        concept_vocabulary="OMOP", concept_code=_GBM,
        source_schema="cbioportal_gbm",
    )


def _resolve_concept(question: str, candidates: list[dict]) -> str:
    """The fake LLM's concept resolution, purely from retrieved context.

    A source-code question resolves through the bridge candidate; a target-term
    question resolves by the concept whose name the question names.
    """
    for c in candidates:
        if c.get("source") == "retrieval_concept_bridge" and c.get("code"):
            return str(c["code"])
    q = question.lower()
    for c in candidates:
        label = str(c.get("label", "")).lower()
        if c.get("code") and label and label.split()[0] in q:
            return str(c["code"])
    return ""


def _fake_llm(question, candidates, sco):
    asset = next(
        a for a in sco.physical_assets if a.table == "condition_occurrence"
    )
    fqn = f"{asset.catalog}.{asset.schema}.{asset.table}"
    code = _resolve_concept(question, candidates)
    sql = (
        f"SELECT COUNT(DISTINCT person_id) FROM {fqn} "
        f"WHERE condition_concept_id = {code}"
    )
    llm = MagicMock()
    llm.invoke = MagicMock(return_value=sql)
    return llm, code


def _plan(engine, question):
    candidate_set = engine.retrieve(question)
    sco = prune_to_sco(candidate_set)
    llm, code = _fake_llm(question, candidate_set.candidates, sco)
    plan = NL2SQLConsumer().plan(
        ConsumerRequest(question=question, operation="plan"),
        sco, ConsumerDeps(llm=llm),
    )
    return plan, code, candidate_set.candidates


def _engine(driver):
    return RetrievalEngine(driver, scope=RetrievalScope(model_role="TARGET"))


def _assert_targets_condition(sql: str) -> None:
    parsed = sqlglot.parse_one(sql, dialect="databricks")
    tables = list(parsed.find_all(sqlglot.exp.Table))
    assert any(t.name == "condition_occurrence" for t in tables)
    assert any(t.db == _SCHEMA for t in tables)


def test_q1_target_term_resolves_correct_concept(clean_neo4j) -> None:
    _load(clean_neo4j)
    plan, code, _ = _plan(
        _engine(clean_neo4j),
        "how many patients have a glioblastoma condition?",
    )
    assert plan.valid, plan.errors
    assert code == _GBM
    _assert_targets_condition(plan.sql)
    assert f"condition_concept_id = {_GBM}" in plan.sql


def test_q2_source_code_bridges_to_same_concept(clean_neo4j) -> None:
    _load(clean_neo4j)
    engine = _engine(clean_neo4j)
    plan, code, candidates = _plan(
        engine, f"how many patients have a {_ONCOTREE_CODE} condition?"
    )
    assert plan.valid, plan.errors
    # Reached the concept by traversing MAPS_TO_CONCEPT, not a lexical match on
    # the concept name (the question never names "glioblastoma").
    assert any(
        c.get("source") == "retrieval_concept_bridge" and c.get("code") == _GBM
        for c in candidates
    )
    assert code == _GBM
    _assert_targets_condition(plan.sql)
    assert f"condition_concept_id = {_GBM}" in plan.sql


def test_q1_and_q2_resolve_to_identical_concept(clean_neo4j) -> None:
    _load(clean_neo4j)
    engine = _engine(clean_neo4j)
    _, q1_code, _ = _plan(
        engine, "how many patients have a glioblastoma condition?"
    )
    _, q2_code, _ = _plan(
        engine, f"how many patients have a {_ONCOTREE_CODE} condition?"
    )
    assert q1_code == q2_code == _GBM

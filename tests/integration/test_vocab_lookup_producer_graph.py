"""US-009 integration: produce the VOCAB_LOOKUP graph mapping in Neo4j.

Skip-guarded — runs only when a Neo4j is reachable. Loads the US-007 OMOP
condition_occurrence TARGET binding, creates a SOURCE ONCOTREE_CODE
:Property, seeds one resolved value-mapping decision, runs the producer, and
asserts the AC bridge query returns one row (ONCOTREE_CODE | VOCAB_LOOKUP |
condition_concept_id) with ``MAPS_TO`` originating at the :FieldMap.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pytest

from sema.graph.target_loader_migrations import cypher_down, cypher_up
from sema.models.planner.lifecycle import Status
from sema.models.planner.provenance import Provenance, RunProvenance, SourceScope
from sema.resolve.engine_utils import ResolveContext
from sema.resolve.policies import resolve_policy
from sema.resolve.producer import MappingNodes, VocabLookupProducer
from sema.resolve.value_mapping_store import ValueMappingStore
from sema.resolve.value_mapping_store_utils import ResolutionStatus, ValueMapping
from sema.targets.adapters.manifest import ManifestTargetAdapter
from sema.targets.loader import load_target
from sema.targets.neo4j_writer import Neo4jGraphWriter
from sema.targets.normalizer import TargetModelNormalizer

pytestmark = pytest.mark.integration

_MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "sema"
    / "targets"
    / "manifests"
    / "omop_condition_slice0.yaml"
)
_TARGET_PROPERTY = "target.condition_occurrence.condition_concept_id"
_VOCAB_RELEASE = "OMOP_2024"


@pytest.fixture
def migrated_neo4j(clean_neo4j):
    with clean_neo4j.session() as session:
        for stmt in cypher_up():
            session.run(stmt)
    yield clean_neo4j
    with clean_neo4j.session() as session:
        for stmt in cypher_down():
            session.run(stmt)


def _policy():
    normalized = TargetModelNormalizer.normalize(ManifestTargetAdapter(_MANIFEST))
    decl = next(
        b
        for b in normalized.vocabulary_bindings
        if b.property_name == "condition_concept_id"
    )
    return resolve_policy(decl)


def _context(policy) -> ResolveContext:
    prov = Provenance(
        run=RunProvenance(
            run_id="run-1",
            target_model_version="v1",
            target_schema_snapshot_hash="h1",
            vocab_release=_VOCAB_RELEASE,
            context_card_version="cc1",
            prompt_template_version="pt1",
            few_shot_set_version="fs1",
            constraint_version="cv1",
            llm_model="none",
        ),
        source=SourceScope(
            source_id="cbioportal", source_schema_hash="sh1", source_profile_hash="sp1"
        ),
        timestamp=datetime(2026, 6, 30, tzinfo=timezone.utc),
    )
    return ResolveContext(
        source_field_ref="source.sample.cancer_type_code",
        source_value_ref="source.sample.cancer_type_code",
        target_property_ref=_TARGET_PROPERTY,
        target_field="condition_concept_id",
        domain_constraint_ref="target.condition_occurrence.domain=Condition",
        vocabulary_ref="target.vocabulary.SNOMED",
        vocab_binding="omop.condition_occurrence.condition_concept_id",
        vocab_release=_VOCAB_RELEASE,
        resolver_policy_ref=policy.resolver_policy_ref,
        run_id="run-1",
        provenance=prov,
    )


def _seed_store(policy) -> ValueMappingStore:
    store = ValueMappingStore(duckdb.connect(":memory:"))
    store.upsert(
        [
            ValueMapping(
                source_vocabulary=policy.source_vocabulary,
                normalized_source_value="LUAD",
                target_property_ref=_TARGET_PROPERTY,
                target_field="condition_concept_id",
                vocab_binding="omop.condition_occurrence.condition_concept_id",
                concept_id=45768916,
                vocab_release=_VOCAB_RELEASE,
                valid_start=None,
                valid_end=None,
                resolution_status=ResolutionStatus.RESOLVED,
                no_map_reason=None,
                confidence=1.0,
                status=Status.auto_accepted,
                resolver_policy_ref=policy.resolver_policy_ref,
                run_id="run-1",
            )
        ]
    )
    return store


def test_producer_writes_field_map_bridge(migrated_neo4j) -> None:
    policy = _policy()
    load_target(
        ManifestTargetAdapter(_MANIFEST), writer=Neo4jGraphWriter(migrated_neo4j)
    )
    with migrated_neo4j.session() as s:
        s.run(
            "CREATE (p:Property {id: 'src-onco', name: 'ONCOTREE_CODE', "
            "model_role: 'SOURCE', source_id: 'cbioportal'})"
        )
        target_id = s.run(
            "MATCH (p:Property {name: 'condition_concept_id', model_role: 'TARGET'}) "
            "RETURN p.id AS id"
        ).single()["id"]

    store = _seed_store(policy)
    with migrated_neo4j.session() as s:
        VocabLookupProducer(s).produce(
            store.read_all(),
            policy,
            _context(policy),
            MappingNodes(source_property_id="src-onco", target_property_id=target_id),
        )
    store.close()

    with migrated_neo4j.session() as s:
        rows = list(
            s.run(
                "MATCH (src:Property {model_role: 'SOURCE'})<-[:DERIVED_FROM]-"
                "(fm:FieldMap)-[:MAPS_TO]->(tgt:Property {model_role: 'TARGET'}) "
                "RETURN src.name AS src, fm.pattern AS pattern, tgt.name AS tgt"
            )
        )
    assert len(rows) == 1
    assert rows[0]["src"] == "ONCOTREE_CODE"
    assert rows[0]["pattern"] == "VOCAB_LOOKUP"
    assert rows[0]["tgt"] == "condition_concept_id"

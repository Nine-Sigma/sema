"""S1-10 — deterministic identity-collapse unit tests (hermetic, temp DuckDB).

Stage B collapses two source entities onto ONE canonical ``entity_id`` only when
they share an exact source key within the SAME identity namespace (e.g. the same
institutional patient id recurring across two studies of one institution). The
tests freeze the guardrails that keep this from over-collapsing (D3): exact-key
only, never across identity namespaces, deterministic survivor, idempotent replay.

The collapse is DOMAIN-GENERIC (R29): it names no ``person``/OMOP literal and the
"which source namespaces share an identity namespace" grouping arrives as data.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sema.resolve.identity_collapse import CollapseResult, collapse_identities
from sema.resolve.identity_collapse_utils import (
    CollapsePlan,
    compute_collapse_plan,
    identity_namespace_of,
)
from sema.resolve.identity_registry import (
    IdentityRegistry,
    open_duckdb_identity_registry,
)
from sema.resolve.identity_registry_utils import IdentityAssignment

pytestmark = pytest.mark.unit

_CHORD = "study_msk_chord"
_IMPACT = "study_msk_impact"
_GBM = "study_gbm_tcga"

# Both MSK studies live in one institutional id namespace; GBM is its own.
_MSK_GROUPING = {_CHORD: "inst_msk", _IMPACT: "inst_msk"}


def _assignment(namespace: str, key: str, entity_id: int) -> IdentityAssignment:
    from sema.resolve.identity_registry_utils import source_entity_uid

    return IdentityAssignment(
        source_namespace=namespace,
        source_entity_key=key,
        source_entity_uid=source_entity_uid(namespace, key),
        entity_id=entity_id,
        run_id="run-a",
    )


@pytest.fixture()
def registry(tmp_path: Path) -> IdentityRegistry:
    return open_duckdb_identity_registry(str(tmp_path / "identity.duckdb"))


class TestIdentityNamespaceOf:
    def test_grouped_namespace_maps_to_identity_namespace(self) -> None:
        assert identity_namespace_of(_CHORD, _MSK_GROUPING) == "inst_msk"
        assert identity_namespace_of(_IMPACT, _MSK_GROUPING) == "inst_msk"

    def test_ungrouped_namespace_is_its_own_identity_namespace(self) -> None:
        # Safe default: an un-configured study can never collapse with another.
        assert identity_namespace_of(_GBM, _MSK_GROUPING) == _GBM


class TestComputeCollapsePlan:
    def test_shared_key_same_identity_namespace_collapses(self) -> None:
        rows = [
            _assignment(_CHORD, "P-1", 5),
            _assignment(_IMPACT, "P-1", 30_000),
        ]
        plan = compute_collapse_plan(rows, namespace_grouping=_MSK_GROUPING)
        assert plan.remaps == {(_IMPACT, "P-1"): 5}
        assert plan.retired_entity_ids == (30_000,)
        assert plan.collapsed_person_count == 1
        assert plan.remapped_row_count == 1

    def test_survivor_is_min_entity_id(self) -> None:
        rows = [
            _assignment(_IMPACT, "P-9", 12),
            _assignment(_CHORD, "P-9", 7),
        ]
        plan = compute_collapse_plan(rows, namespace_grouping=_MSK_GROUPING)
        assert plan.remaps == {(_IMPACT, "P-9"): 7}

    def test_distinct_keys_never_collapse(self) -> None:
        rows = [
            _assignment(_CHORD, "P-1", 1),
            _assignment(_IMPACT, "P-2", 2),
        ]
        plan = compute_collapse_plan(rows, namespace_grouping=_MSK_GROUPING)
        assert plan.remaps == {}
        assert plan.groups == ()

    def test_same_key_across_identity_namespaces_never_collapses(self) -> None:
        # The over-collapse trap: a "P-1" in GBM's namespace is a DIFFERENT person
        # from "P-1" in MSK's — no grouping links them, so they must not merge.
        rows = [
            _assignment(_CHORD, "P-1", 1),
            _assignment(_GBM, "P-1", 2),
        ]
        plan = compute_collapse_plan(rows, namespace_grouping=_MSK_GROUPING)
        assert plan.remaps == {}

    def test_no_grouping_collapses_nothing(self) -> None:
        rows = [
            _assignment(_CHORD, "P-1", 1),
            _assignment(_IMPACT, "P-1", 2),
        ]
        plan = compute_collapse_plan(rows, namespace_grouping={})
        assert plan.remaps == {}

    def test_three_studies_one_survivor(self) -> None:
        grouping = {_CHORD: "inst_msk", _IMPACT: "inst_msk", "study_msk_access": "inst_msk"}
        rows = [
            _assignment(_CHORD, "P-1", 5),
            _assignment(_IMPACT, "P-1", 30_000),
            _assignment("study_msk_access", "P-1", 40_000),
        ]
        plan = compute_collapse_plan(rows, namespace_grouping=grouping)
        assert plan.remaps == {(_IMPACT, "P-1"): 5, ("study_msk_access", "P-1"): 5}
        assert set(plan.retired_entity_ids) == {30_000, 40_000}
        assert plan.collapsed_person_count == 2

    def test_already_collapsed_plan_is_empty(self) -> None:
        rows = [
            _assignment(_CHORD, "P-1", 5),
            _assignment(_IMPACT, "P-1", 5),
        ]
        plan = compute_collapse_plan(rows, namespace_grouping=_MSK_GROUPING)
        assert plan.remaps == {}
        assert plan.collapsed_person_count == 0


class TestRegistryRemap:
    def test_remap_updates_entity_id_only(self, registry: IdentityRegistry) -> None:
        registry.get_or_create([(_CHORD, "P-1"), (_IMPACT, "P-1")], run_id="run-a")
        before = {(a.source_namespace, a.source_entity_key): a for a in registry.read_all()}
        survivor = before[(_CHORD, "P-1")].entity_id

        updated = registry.remap_entity_ids({(_IMPACT, "P-1"): survivor})

        assert updated == 1
        after = registry.get(_IMPACT, "P-1")
        assert after is not None
        assert after.entity_id == survivor
        # uid is per source entity and must be untouched by an entity_id remap.
        assert after.source_entity_uid == before[(_IMPACT, "P-1")].source_entity_uid

    def test_remap_noop_when_already_target(self, registry: IdentityRegistry) -> None:
        registry.get_or_create([(_CHORD, "P-1")], run_id="run-a")
        eid = registry.get(_CHORD, "P-1").entity_id  # type: ignore[union-attr]
        assert registry.remap_entity_ids({(_CHORD, "P-1"): eid}) == 0


class TestCollapseIdentities:
    def test_collapse_applies_plan_to_registry(self, registry: IdentityRegistry) -> None:
        registry.get_or_create([(_CHORD, "P-1"), (_IMPACT, "P-1")], run_id="run-a")

        result = collapse_identities(registry, namespace_grouping=_MSK_GROUPING)

        assert isinstance(result, CollapseResult)
        assert result.collapsed_person_count == 1
        assert result.remapped_row_count == 1
        ids = {a.entity_id for a in registry.read_all()}
        assert len(ids) == 1  # both source patients now share one canonical id

    def test_collapse_is_idempotent(self, registry: IdentityRegistry) -> None:
        registry.get_or_create([(_CHORD, "P-1"), (_IMPACT, "P-1")], run_id="run-a")
        collapse_identities(registry, namespace_grouping=_MSK_GROUPING)

        second = collapse_identities(registry, namespace_grouping=_MSK_GROUPING)

        assert second.collapsed_person_count == 0
        assert second.remapped_row_count == 0

    def test_collapse_across_namespaces_is_a_noop(self, registry: IdentityRegistry) -> None:
        registry.get_or_create([(_CHORD, "P-1"), (_GBM, "P-1")], run_id="run-a")
        result = collapse_identities(registry, namespace_grouping=_MSK_GROUPING)
        assert result.collapsed_person_count == 0
        assert len({a.entity_id for a in registry.read_all()}) == 2


def _spec_plan_is_empty(plan: CollapsePlan) -> bool:
    return plan.remaps == {} and plan.groups == ()

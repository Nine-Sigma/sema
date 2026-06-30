"""Cross-capability integration tests using InMemoryGraphWriter.

Mirrors the unit-testable subset of Section 8: provenance integration,
pin invalidation, lazy subset, multi-adapter, supplied vs deferred,
operator skip-facets.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from sema.models.planner.lifecycle import HumanPin, PinState
from sema.models.planner.lifecycle_utils import detect_pin_stale
from sema.models.planner.provenance import RunProvenance, SourceScope
from sema.models.target.enrichment import EnrichmentStatus, Facet
from sema.targets.adapters.manifest import ManifestTargetAdapter
from sema.targets.loader import load_target
from sema.targets.materializer import InMemoryGraphWriter

pytestmark = pytest.mark.unit

_FIXTURES = Path(__file__).parent / "fixtures"
GOLDEN = _FIXTURES / "golden_manifest.yaml"


def _build_run(snapshot_hash: str, card_version: str = "1.0.0") -> RunProvenance:
    return RunProvenance(
        run_id="run-1",
        target_model_version="1.0.0",
        target_schema_snapshot_hash=snapshot_hash,
        context_card_version=card_version,
        prompt_template_version="prompt-v1",
        few_shot_set_version="fs-v1",
        constraint_version="cv-v1",
        llm_model="model-v1",
    )


def _build_source_scope() -> SourceScope:
    return SourceScope(
        source_id="src-1",
        source_schema_hash="srchash" * 9 + "x",
        source_profile_hash="profh" * 12 + "x",
    )


def test_loaded_target_hash_feeds_run_provenance() -> None:
    adapter = ManifestTargetAdapter(GOLDEN)
    loaded = load_target(adapter, writer=InMemoryGraphWriter())
    run = _build_run(loaded.target_schema_snapshot_hash)
    assert run.target_schema_snapshot_hash == loaded.target_schema_snapshot_hash


def test_pin_goes_stale_on_schema_hash_drift(tmp_path: Path) -> None:
    base = load_target(ManifestTargetAdapter(GOLDEN), writer=InMemoryGraphWriter())
    raw = yaml.safe_load(GOLDEN.read_text())
    raw["entities"][0]["properties"][0]["type"] = "string"
    drifted_path = tmp_path / "drifted.yaml"
    drifted_path.write_text(yaml.safe_dump(raw))
    drifted = load_target(
        ManifestTargetAdapter(drifted_path), writer=InMemoryGraphWriter()
    )
    pin = HumanPin(
        pin_id="pin-1",
        assertion_id="a-1",
        pinned_at=datetime.now(tz=timezone.utc),
        pinned_by="reviewer",
        confirmed_under_run=_build_run(base.target_schema_snapshot_hash),
        confirmed_under_source=_build_source_scope(),
    )
    new_run = _build_run(drifted.target_schema_snapshot_hash)
    updated = detect_pin_stale(pin, new_run, _build_source_scope())
    assert updated.pin_state is PinState.stale


def test_pin_goes_stale_on_card_version_drift_only(tmp_path: Path) -> None:
    base = load_target(ManifestTargetAdapter(GOLDEN), writer=InMemoryGraphWriter())
    raw = yaml.safe_load(GOLDEN.read_text())
    raw["entities"][0]["context_card"]["card_version"] = "2.0.0"
    raw["entities"][0]["context_card"]["description"] = "Bumped wording"
    bumped_path = tmp_path / "bumped.yaml"
    bumped_path.write_text(yaml.safe_dump(raw))
    bumped = load_target(
        ManifestTargetAdapter(bumped_path), writer=InMemoryGraphWriter()
    )
    assert base.target_schema_snapshot_hash == bumped.target_schema_snapshot_hash
    assert (
        base.aggregate_context_card_version != bumped.aggregate_context_card_version
    )
    pin = HumanPin(
        pin_id="pin-1",
        assertion_id="a-1",
        pinned_at=datetime.now(tz=timezone.utc),
        pinned_by="reviewer",
        confirmed_under_run=_build_run(
            base.target_schema_snapshot_hash, base.aggregate_context_card_version
        ),
        confirmed_under_source=_build_source_scope(),
    )
    new_run = _build_run(
        bumped.target_schema_snapshot_hash, bumped.aggregate_context_card_version
    )
    updated = detect_pin_stale(pin, new_run, _build_source_scope())
    assert updated.pin_state is PinState.stale


def test_lazy_subset_materializes_only_selected_entities() -> None:
    adapter = ManifestTargetAdapter(GOLDEN)
    refs = {r.qualified_name: r for r in adapter.discover_entities()}
    selected = [refs["omop.person"], refs["omop.observation"]]
    writer = InMemoryGraphWriter()
    loaded = load_target(adapter, writer=writer, selected_refs=selected)
    materialized = {
        r.qualified_name for r in loaded.entity_refs
    }
    assert materialized == {"omop.person", "omop.observation"}


def test_lazy_subset_hash_is_stable_across_reruns() -> None:
    adapter = ManifestTargetAdapter(GOLDEN)
    refs = {r.qualified_name: r for r in adapter.discover_entities()}
    selected = [refs["omop.person"]]
    a = load_target(adapter, writer=InMemoryGraphWriter(), selected_refs=selected)
    b = load_target(adapter, writer=InMemoryGraphWriter(), selected_refs=selected)
    assert a.target_schema_snapshot_hash == b.target_schema_snapshot_hash


def test_multi_adapter_no_collisions(tmp_path: Path) -> None:
    raw = yaml.safe_load(GOLDEN.read_text())
    raw["descriptor"]["target_model_id"] = "second-target"
    second_path = tmp_path / "second.yaml"
    second_path.write_text(yaml.safe_dump(raw))
    a = load_target(ManifestTargetAdapter(GOLDEN), writer=InMemoryGraphWriter())
    b = load_target(ManifestTargetAdapter(second_path), writer=InMemoryGraphWriter())
    assert a.descriptor.target_model_id != b.descriptor.target_model_id
    assert a.target_schema_snapshot_hash != b.target_schema_snapshot_hash


def test_supplied_versus_deferred_enrichment() -> None:
    adapter = ManifestTargetAdapter(GOLDEN)
    loaded = load_target(adapter, writer=InMemoryGraphWriter())
    by_qname = {d.entity_ref.qualified_name: d for d in loaded.enrichment_decisions}
    person = by_qname["omop.person"]
    measurement = by_qname["omop.measurement"]
    assert (
        person.decisions[Facet.semantic_aliases].status
        is EnrichmentStatus.supplied_by_adapter
    )
    assert (
        measurement.decisions[Facet.semantic_aliases].status
        is EnrichmentStatus.required_deferred
    )


def test_operator_skip_facets_overrides_other_decisions() -> None:
    adapter = ManifestTargetAdapter(GOLDEN)
    loaded = load_target(
        adapter,
        writer=InMemoryGraphWriter(),
        skip_facets=["semantic_aliases"],
    )
    for record in loaded.enrichment_decisions:
        status = record.decisions[Facet.semantic_aliases].status
        assert status in {
            EnrichmentStatus.required_skipped,
            EnrichmentStatus.supplied_by_adapter,
            EnrichmentStatus.not_required,
        }

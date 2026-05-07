"""Tests for the provenance-and-caching capability."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

pytestmark = pytest.mark.unit


def _run_prov(**overrides: object) -> object:
    from sema.models.planner.provenance import RunProvenance

    base = dict(
        run_id="run-1",
        target_model_version="omop-cdm-5.4",
        target_schema_snapshot_hash="t-abc",
        vocab_release="omop-2026-q1",
        context_card_version="cards-v3",
        prompt_template_version="tpl-7",
        few_shot_set_version="fs-12",
        constraint_version="rules-v2",
        llm_model="claude-opus-4.7",
        embedding_model="bge-large",
    )
    base.update(overrides)
    return RunProvenance(**base)


def _source_scope(**overrides: object) -> object:
    from sema.models.planner.provenance import SourceScope

    base = dict(
        source_id="cbioportal_gbm",
        source_schema_hash="s-abc",
        source_profile_hash="p-abc",
    )
    base.update(overrides)
    return SourceScope(**base)


def test_run_provenance_round_trip() -> None:
    from sema.models.planner.provenance import RunProvenance

    rp = _run_prov()
    payload = rp.model_dump(mode="json")
    rt = RunProvenance.model_validate(payload)
    assert rt.run_id == "run-1"
    assert rt.llm_model == "claude-opus-4.7"


def test_run_provenance_required_fields() -> None:
    from sema.models.planner.provenance import RunProvenance

    with pytest.raises(ValidationError):
        RunProvenance(  # type: ignore[call-arg]
            run_id="r",
            target_model_version="v",
            target_schema_snapshot_hash="h",
            context_card_version="c",
            prompt_template_version="t",
            few_shot_set_version="f",
            llm_model="m",
        )


def test_source_scope_round_trip() -> None:
    from sema.models.planner.provenance import SourceScope

    s = _source_scope()
    rt = SourceScope.model_validate(s.model_dump(mode="json"))
    assert rt.source_id == "cbioportal_gbm"


def test_provenance_composes_run_and_source() -> None:
    from sema.models.planner.provenance import Provenance

    p = Provenance(
        run=_run_prov(),
        source=_source_scope(),
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    assert p.run.run_id == "run-1"
    assert p.source.source_id == "cbioportal_gbm"


def test_run_version_lock_detects_drift() -> None:
    from sema.models.planner.provenance import RunVersionLock

    lock = RunVersionLock()
    rp1 = _run_prov()
    rp2 = _run_prov(prompt_template_version="tpl-8")
    lock.bind(rp1)
    with pytest.raises(ValueError):
        lock.bind(rp2)


def test_run_version_lock_allows_same() -> None:
    from sema.models.planner.provenance import RunVersionLock

    lock = RunVersionLock()
    lock.bind(_run_prov())
    lock.bind(_run_prov())


def test_source_scope_lock_per_run_id() -> None:
    from sema.models.planner.provenance import SourceScopeLock

    lock = SourceScopeLock(run_id="run-1")
    lock.bind(_source_scope())
    lock.bind(_source_scope(source_id="msk_chord", source_schema_hash="h2", source_profile_hash="p2"))
    with pytest.raises(ValueError):
        lock.bind(_source_scope(source_profile_hash="p-DRIFT"))


def test_prompt_artifact_prefix_hash_deterministic() -> None:
    from sema.models.planner.provenance import PromptArtifact

    a = PromptArtifact.build(
        prefix_text="system\ntarget cards\nfew-shot A\n",
        suffix_text="source: cbio.patient.gender\n",
        versions={"target_model_version": "omop-cdm-5.4"},
    )
    b = PromptArtifact.build(
        prefix_text="system\ntarget cards\nfew-shot A\n",
        suffix_text="source: cbio.patient.race\n",
        versions={"target_model_version": "omop-cdm-5.4"},
    )
    assert a.prefix_hash == b.prefix_hash
    assert a.suffix_text != b.suffix_text


def test_prompt_artifact_explicit_hash_must_match() -> None:
    from sema.models.planner.provenance import PromptArtifact

    with pytest.raises(ValidationError):
        PromptArtifact(
            prefix_text="x",
            prefix_hash="not-a-real-digest",
            suffix_text="y",
            versions={},
        )


def test_prompt_artifact_source_isolation_passes_clean_prefix() -> None:
    from sema.models.planner.provenance import PromptArtifact

    art = PromptArtifact.build(
        prefix_text="system\nOMOP cards\nfew-shot for omop.person\n",
        suffix_text="source: cbio.patient.gender\n",
        versions={},
    )
    art.assert_source_isolated("cbio.patient.gender")


def test_prompt_artifact_source_isolation_rejects_leak() -> None:
    from sema.models.planner.provenance import PromptArtifact

    art = PromptArtifact.build(
        prefix_text="system\nfew-shot referencing cbio.patient.gender\n",
        suffix_text="trailer\n",
        versions={},
    )
    with pytest.raises(ValueError, match="cbio.patient.gender"):
        art.assert_source_isolated("cbio.patient.gender")


def test_prompt_artifact_source_isolation_empty_ref_rejected() -> None:
    from sema.models.planner.provenance import PromptArtifact

    art = PromptArtifact.build(prefix_text="p", suffix_text="s", versions={})
    with pytest.raises(ValueError, match="non-empty"):
        art.assert_source_isolated("")


def test_cache_key_changes_with_tracked_dimension() -> None:
    from sema.models.planner.provenance import (
        PromptArtifact,
        derive_cache_key,
    )

    art = PromptArtifact.build(
        prefix_text="prefix",
        suffix_text="s1",
        versions={"context_card_version": "v1"},
    )
    rp1 = _run_prov(context_card_version="cards-v3")
    rp2 = _run_prov(context_card_version="cards-v4")
    assert derive_cache_key(art, rp1) != derive_cache_key(art, rp2)


def test_cache_key_ignores_source_scope() -> None:
    from sema.models.planner.provenance import (
        PromptArtifact,
        derive_cache_key,
    )

    art = PromptArtifact.build(prefix_text="prefix", suffix_text="s", versions={})
    rp = _run_prov()
    src1 = _source_scope()
    src2 = _source_scope(source_id="msk_chord", source_schema_hash="x", source_profile_hash="y")
    assert derive_cache_key(art, rp, src1) == derive_cache_key(art, rp, src2)


def test_source_profile_hash_stable() -> None:
    from sema.models.planner.provenance import compute_source_profile_hash

    sig = {
        "columns": [{"name": "gender", "samples": ["M", "F"], "distinct": 2, "null_rate": 0.0}],
    }
    a = compute_source_profile_hash(sig)
    b = compute_source_profile_hash(sig)
    sig_drift = {
        "columns": [{"name": "gender", "samples": ["M", "F"], "distinct": 2, "null_rate": 0.05}],
    }
    c = compute_source_profile_hash(sig_drift)
    assert a == b
    assert a != c


def test_llm_runtime_protocol_anthropic_caches_prefix() -> None:
    from sema.models.planner.provenance import (
        AnthropicCachingAdapter,
        PromptArtifact,
    )

    adapter = AnthropicCachingAdapter(name="claude-opus-4.7")
    art = PromptArtifact.build(prefix_text="prefix", suffix_text="s", versions={})
    headers = adapter.cache_directives(art)
    assert headers.get("cache_control") == "ephemeral"
    assert adapter.dialect == "anthropic"


def test_llm_runtime_protocol_mosaic_no_cache() -> None:
    from sema.models.planner.provenance import (
        MosaicAIAdapter,
        PromptArtifact,
    )

    adapter = MosaicAIAdapter(name="dbrx-instruct")
    art = PromptArtifact.build(prefix_text="prefix", suffix_text="s", versions={})
    headers = adapter.cache_directives(art)
    assert "cache_control" not in headers
    assert adapter.dialect == "mosaic"

"""Arm S live entry points: routing guards and the build-manifest check."""

from __future__ import annotations

import contextlib
import dataclasses
from pathlib import Path
from typing import Any

import pytest

from sema.pipeline import retrieval
from showcase.ontology_query_benchmark import sema_live, sema_trace
from showcase.ontology_query_benchmark.profile import Attribution, Profile
from showcase.ontology_query_benchmark.routing import RoutingError
from showcase.ontology_query_benchmark.sema_bindings import load_sidecar
from showcase.ontology_query_benchmark.sema_build import BuildManifest, write_manifest
from showcase.ontology_query_benchmark.sema_build_utils import import_library_facts
from showcase.ontology_query_benchmark.sema_context import ManifestMismatchError
from showcase.ontology_query_benchmark.sema_trace import open_live_provider
from tests.unit.ontology_query_benchmark.conftest import PROFILE_PATH
from tests.unit.ontology_query_benchmark.test_sema_bindings import (
    CONCEPT_SPEC,
    catalog_fixture,
    write_fixture,
    write_projection,
)
from tests.unit.ontology_query_benchmark.test_sema_build import (
    ClientBuilt,
    clean_process,
    no_client,
)
from tests.unit.ontology_query_benchmark.test_sema_context import QUESTION

pytestmark = pytest.mark.unit


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


def test_a_hierarchy_manifest_without_term_embeddings_is_refused_as_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, profile: Profile
) -> None:
    clean_process(monkeypatch)
    no_client(monkeypatch)
    local, manifest, projection, receipt = _live_inputs(tmp_path, profile)
    path = tmp_path / "manifest.json"
    write_manifest(dataclasses.replace(manifest, counts={"hierarchy_depth": 2}), path)
    with pytest.raises(ManifestMismatchError, match="embedded_terms"):
        with open_live_provider(local, "bolt://unused", path, projection, receipt):
            pass


def test_the_live_engine_walks_as_deep_as_the_manifest_slice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, profile: Profile
) -> None:
    clean_process(monkeypatch)
    engines: list[dict[str, Any]] = []

    class _Engine:
        def __init__(self, *_args: Any, **kwargs: Any) -> None:
            engines.append(kwargs)

    monkeypatch.setattr(sema_live, "pinned_workspace", lambda _routing: object())
    monkeypatch.setattr(sema_live, "UnityCatalogSource", lambda _workspace: object())
    monkeypatch.setattr(sema_live, "embedder", lambda *_args: object())
    monkeypatch.setattr(sema_live, "neo4j_driver", lambda _uri: contextlib.nullcontext())
    monkeypatch.setattr(sema_trace, "read_catalog", lambda *_args: catalog_fixture())
    monkeypatch.setattr(sema_trace, "load_conventions", lambda _sidecar: ())
    monkeypatch.setattr(retrieval, "RetrievalEngine", _Engine)
    local, manifest, projection, receipt = _live_inputs(tmp_path, profile)
    counts = {"hierarchy_depth": 4, "embedded_terms": 7}
    write_manifest(dataclasses.replace(manifest, counts=counts), tmp_path / "manifest.json")
    with open_live_provider(local, "bolt://unused", tmp_path / "manifest.json",
                            projection, receipt):  # fmt: skip
        pass

    [kwargs] = engines
    assert kwargs["hierarchy_anchor"].max_depth == 4

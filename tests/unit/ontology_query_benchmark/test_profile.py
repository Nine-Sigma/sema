"""A profile parses once at the boundary into a frozen model; relative paths are local."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from showcase.ontology_query_benchmark.profile import Knobs, load_profile
from showcase.ontology_query_benchmark.sema_bindings import load_products, load_sidecar
from showcase.ontology_query_benchmark.sema_context import load_conventions

pytestmark = pytest.mark.unit

FIXTURES = Path(__file__).parents[2] / "fixtures" / "ontology_query_benchmark"
PROFILE = FIXTURES / "profile.yaml"


def _variant(tmp_path: Path, edit: dict[str, object]) -> Path:
    raw = yaml.safe_load(PROFILE.read_text())
    for dotted, value in edit.items():
        *parents, leaf = dotted.split(".")
        node = raw
        for key in parents:
            node = node[key]
        node[leaf] = value
    path = tmp_path / "profile.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def test_relative_paths_resolve_against_the_profile_directory() -> None:
    profile = load_profile(PROFILE)
    assert profile.harness.root == FIXTURES / "harness"
    assert profile.sema.sidecar == FIXTURES / "sidecar.json"
    assert profile.sema.concept_spec == FIXTURES / "concept_source.yaml"


def test_a_loaded_profile_is_frozen() -> None:
    profile = load_profile(PROFILE)
    with pytest.raises(ValidationError):
        profile.routing.host = "https://elsewhere.example.test"  # type: ignore[misc]


def test_an_unknown_key_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="warehouse_alias"):
        load_profile(_variant(tmp_path, {"routing.warehouse_alias": "x"}))


@pytest.mark.parametrize("host", ["https://Toy.example.test", "https://toy.example.test/",
                                  "toy.example.test"])  # fmt: skip
def test_a_pinned_host_must_be_normalized(tmp_path: Path, host: str) -> None:
    with pytest.raises(ValidationError, match="host"):
        load_profile(_variant(tmp_path, {"routing.host": host}))


def test_sweep_grid_varies_the_input_ceiling_fastest() -> None:
    grid = load_profile(PROFILE).sweep.grid()
    knobs = Knobs(budget_tokens=3000, values_top_k=25, description_cap_chars=160)
    assert len(grid) == 8 and grid[:2] == [(knobs, 16000), (knobs, 36000)]


def test_timbr_mcp_url_joins_url_and_path() -> None:
    assert load_profile(PROFILE).timbr.mcp_url == "https://ontology.example.test/api/mcp"


def test_s_inputs_named_by_the_profile_load_and_pin() -> None:
    profile = load_profile(PROFILE)
    sidecar = load_sidecar(profile.sema.sidecar, profile.harness.root)
    assert [p.pin.product for p in load_products(sidecar)] == ["people", "events"]
    assert load_conventions(sidecar) == ("People means distinct rid values.",)


def test_sources_pin_each_relation_to_a_version(tmp_path: Path) -> None:
    pins = [{"relation": "cat.s.person", "version": 7}, {"relation": "cat.s.visit", "version": 0}]
    profile = load_profile(_variant(tmp_path, {"sources.pins": pins}))
    assert profile.source_versions == {"cat.s.person": 7, "cat.s.visit": 0}


@pytest.mark.parametrize(
    ("pins", "refusal"),
    [
        ([{"relation": "cat.s.person"}], r"sources\.pins\.0\.version"),
        ([{"relation": "cat.s.person", "version": None}], r"sources\.pins\.0\.version"),
        ([{"relation": "person", "version": 7}], r"sources\.pins\.0\.relation"),
        ([], "at least 1"),
    ],
)
def test_an_unpinned_or_unqualified_source_is_refused(
    tmp_path: Path, pins: list[dict[str, object]], refusal: str
) -> None:
    with pytest.raises(ValidationError, match=refusal):
        load_profile(_variant(tmp_path, {"sources.pins": pins}))


@pytest.mark.parametrize(
    "version_sql",
    ["SELECT 1 FROM history", "SELECT v FROM {relation} WHERE {other}", "SELECT {} FROM {relation}"],
)
def test_a_version_query_must_template_exactly_the_relation(tmp_path: Path, version_sql: str) -> None:
    with pytest.raises(ValidationError, match=r"sources\.version_sql"):
        load_profile(_variant(tmp_path, {"sources.version_sql": version_sql}))


def test_a_sealed_set_profile_expands_home_and_resolves_relative_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from showcase.ontology_query_benchmark.profile import load_sealed_profile

    sealed_profile = Path(__file__).parents[2] / "fixtures" / "sealed_set" / "profile.yaml"
    raw = yaml.safe_load(sealed_profile.read_text())
    raw["duckdb"] = "~/data/source.duckdb"
    path = tmp_path / "profile.yaml"
    path.write_text(yaml.safe_dump(raw))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    profile = load_sealed_profile(path)

    assert profile.duckdb == tmp_path / "home" / "data" / "source.duckdb"
    assert profile.set_dir == tmp_path / "neutral"
    assert profile.harness.frozen == tmp_path / "frozen"
    assert (profile.model.provider, profile.knobs.budget_tokens) == ("openrouter", 12000)

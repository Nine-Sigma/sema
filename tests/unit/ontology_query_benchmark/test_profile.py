"""A profile parses once at the boundary into a frozen model; relative paths are local."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from showcase.ontology_query_benchmark.profile import Knobs, load_profile

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

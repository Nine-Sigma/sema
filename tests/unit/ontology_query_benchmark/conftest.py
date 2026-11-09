from pathlib import Path
from types import SimpleNamespace

import pytest

from showcase.ontology_query_benchmark import freeze, runner_utils
from showcase.ontology_query_benchmark.profile import Profile, load_profile

FIXTURES = Path(__file__).parents[2] / "fixtures" / "ontology_query_benchmark"
PROFILE_PATH = FIXTURES / "profile.yaml"


@pytest.fixture(scope="session")
def profile() -> Profile:
    return load_profile(PROFILE_PATH)


@pytest.fixture(scope="session")
def frozen(profile: Profile, tmp_path_factory: pytest.TempPathFactory) -> Path:
    return freeze.freeze(profile.harness.root, tmp_path_factory.mktemp("frozen"),
                         profile.harness.closure)  # fmt: skip


@pytest.fixture(scope="session")
def harness(profile: Profile, frozen: Path) -> SimpleNamespace:
    return runner_utils.load_harness(frozen, profile.harness)

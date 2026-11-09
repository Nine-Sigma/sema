from pathlib import Path
from types import SimpleNamespace

import pytest

from showcase.partner_timbr_benchmark import freeze, runner_utils


@pytest.fixture(scope="session")
def frozen(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if not freeze.HARNESS_ROOT.is_dir():
        pytest.skip("Timbr harness tree is not on this machine")
    return freeze.freeze(freeze.HARNESS_ROOT, tmp_path_factory.mktemp("frozen"))


@pytest.fixture(scope="session")
def harness(frozen: Path) -> SimpleNamespace:
    return runner_utils.load_harness(frozen)

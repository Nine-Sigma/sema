"""Session fixtures for the verifier tests: A2, A3, A7 CSV, and A10 landed once, with their
structure."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from sema.binding.spec import load_concept_source_spec
from sema.ingest.duckdb_staging import Staging
from sema.intake.land.loaders.semsql import land_semsql
from sema.intake.land.loaders.tabular import land_tabular
from tests.unit.intake.structure.shapes.shapes_env import OMOP_CSV, Source
from tests.unit.intake.structure_env import (
    A2_WORKBOOK,
    A3_DB,
    ACRIS,
    ACRIS_SPEC,
    fresh_tools,
    land,
    run,
)


@dataclass
class Landed:
    staging: Staging
    source: Source
    root: Path

    def structure(self, name: str, hint: Any = None) -> Any:
        return run(self.source, fresh_tools(self.source, self.root / "ledger" / name), hint)


def _landed(factory: pytest.TempPathFactory, name: str, lander: Any, path: Path) -> Landed:
    root = factory.mktemp(f"verify_{name}")
    staging, source = land(root, name, lander, path)
    return Landed(staging, source, root)


@pytest.fixture(scope="session")
def a2(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Landed]:
    landed = _landed(tmp_path_factory, "a2", land_tabular, A2_WORKBOOK)
    yield landed
    landed.staging.close()


@pytest.fixture(scope="session")
def a3(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Landed]:
    landed = _landed(tmp_path_factory, "a3", land_semsql, A3_DB)
    yield landed
    landed.staging.close()


@pytest.fixture(scope="session")
def acris(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Landed]:
    landed = _landed(tmp_path_factory, "acris", land_tabular, ACRIS)
    yield landed
    landed.staging.close()


@pytest.fixture(scope="session")
def omop(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Landed]:
    landed = _landed(tmp_path_factory, "omop", land_tabular, OMOP_CSV)
    yield landed
    landed.staging.close()


@pytest.fixture(scope="session")
def omop_structure(omop: Landed) -> Any:
    return omop.structure("omop_structure")


@pytest.fixture(scope="session")
def a2_structure(a2: Landed) -> Any:
    return a2.structure("a2_structure")


@pytest.fixture(scope="session")
def acris_structure(acris: Landed) -> Any:
    return acris.structure("acris_structure")


@pytest.fixture(scope="session")
def acris_hinted(acris: Landed) -> Any:
    return acris.structure("acris_hinted", load_concept_source_spec(ACRIS_SPEC))

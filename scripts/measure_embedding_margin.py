"""Measure ``EMBEDDING_MARGIN_FLOOR`` once over the measurement fixtures (task 8.3).

Each fixture is landed, structured, and run through Roles with no embedder, then every
relation-level role with two or more eligible columns is ranked by the configured
embedder and compared to the hand-written spec. Held-out sources are refused.

    EMBEDDING_PROVIDER=openrouter EMBEDDING_MODEL=<model> \\
        uv run --env-file .env python -m scripts.measure_embedding_margin
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from scripts.measure_embedding_margin_utils import (
    MarginCase,
    expected_columns,
    margin_cases,
    refuse_heldout,
    select_floor,
)
from sema import cli_factories
from sema.binding.spec import load_concept_source_spec
from sema.ingest.duckdb_staging import Staging
from sema.intake.eligibility import run_roles
from sema.intake.eligibility_utils import Embedder
from sema.intake.landing import LandResult
from sema.intake.ledger import Ledger
from sema.intake.loaders.semsql import land_semsql
from sema.intake.loaders.tabular import land_tabular
from sema.intake.profile import EvidencePolicy, ProfileTools
from sema.intake.relations import RelationProvider
from sema.intake.shapes.base import profile_relations
from sema.intake.structure import StructureResult, run_structure
from sema.models.config import EmbeddingConfig

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "intake"
SEMSQL_SUFFIX = ".db"
ROW = "{:<8} {:<70} {:<14} {:<26} {:<26} {:>8}  {}"

Lander = Callable[..., LandResult]


@dataclass(frozen=True)
class MeasurementFixture:
    name: str
    path: Path
    spec: Path

    @property
    def schema(self) -> str:
        return f"land_{self.name}"

    @property
    def lander(self) -> Lander:
        return land_semsql if Path(self.path).suffix == SEMSQL_SUFFIX else land_tabular


MEASUREMENT_FIXTURES = (
    MeasurementFixture(
        "a2", FIXTURES / "tabular" / "codes_a2.xlsx", FIXTURES / "tabular" / "codes_a2.spec.yaml"
    ),
    MeasurementFixture(
        "a3", FIXTURES / "semsql" / "ontology.db", FIXTURES / "semsql" / "ontology.spec.yaml"
    ),
    MeasurementFixture(
        "a8",
        FIXTURES / "tabular" / "ambiguity_a8.csv",
        FIXTURES / "tabular" / "ambiguity_a8.spec.yaml",
    ),
    MeasurementFixture("acris", FIXTURES / "acris", FIXTURES / "acris" / "acris.spec.yaml"),
)


def _tools(
    staging: Staging,
    ledger: Ledger,
    stage: str,
    land: LandResult,
    structure: StructureResult | None = None,
) -> ProfileTools:
    return ProfileTools(
        RelationProvider(warehouse=None, staging=staging),
        EvidencePolicy(tier="metadata_only"),
        ledger.run(stage),
        stage=stage,
        structure=structure,
        land=land,
    )


def measure_fixture(
    fixture: MeasurementFixture, embedder: Embedder, workdir: Path
) -> list[MarginCase]:
    """Structure and Roles run as on any unseen source: no hint, no model, no floor."""
    path, spec_path = refuse_heldout(fixture.path), refuse_heldout(fixture.spec)
    truth = expected_columns(load_concept_source_spec(spec_path))
    root = workdir / fixture.name
    root.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(root / "ledger")
    with Staging(str(root / "staging.duckdb"), schemas=()) as staging:
        land = fixture.lander(path, staging, schema=fixture.schema)
        structure = run_structure(land, _tools(staging, ledger, "structure", land))
        tools = _tools(staging, ledger, "roles", land, structure)
        profiles = profile_relations(land, tools, "staging")
        roles = run_roles(structure, profiles, tools, margin_floor=None)
        return margin_cases(fixture.name, roles.eligibilities, profiles, truth, embedder)


def _print_cases(cases: Sequence[MarginCase]) -> None:
    print(ROW.format("fixture", "relation", "role", "top", "expected", "margin", "verdict"))
    for case in cases:
        expected = case.expected or "-"
        print(ROW.format(
            case.fixture, case.relation, case.role, case.top, expected,
            f"{case.margin:.4f}", case.verdict,
        ))


def _measure(chosen: Sequence[MeasurementFixture], workdir: Path) -> list[MarginCase]:
    embedder = cli_factories._get_embedder(EmbeddingConfig())
    return [case for fixture in chosen for case in measure_fixture(fixture, embedder, workdir)]


def main(argv: Sequence[str] | None = None) -> int:
    names = [fixture.name for fixture in MEASUREMENT_FIXTURES]
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--fixture", action="append", choices=names)
    parser.add_argument("--workdir", type=Path)
    args = parser.parse_args(argv)
    chosen = [f for f in MEASUREMENT_FIXTURES if f.name in (args.fixture or names)]
    if args.workdir is not None:
        cases = _measure(chosen, args.workdir)
    else:
        with tempfile.TemporaryDirectory() as scratch:
            cases = _measure(chosen, Path(scratch))
    _print_cases(cases)
    counts = {v: sum(c.verdict == v for c in cases) for v in ("wrong", "correct", "skipped")}
    print(" ".join(f"{verdict}={count}" for verdict, count in counts.items()))
    print(f"EMBEDDING_MARGIN_FLOOR = {select_floor(cases):.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""The generality gate (task 13.2, generality-gate spec, design Decision 19).

The held-out fixture runs through landing, structure with no hint, roles with a run
descriptor written here, the verifiers, and the dry run under ``metadata_only``, with no
change to ``src/``. The gate asserts only the partition, the Land and Structure verdicts,
and the policy audit. The ambiguity lists are written to the ledger and their counts
printed; a human reads them and the design document records that reading. Nothing here is
an input to a floor, token, pattern, or predicate.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

from sema.intake.eligibility_utils import RunDescriptor
from sema.intake.metrics import RunMetrics, run_metrics
from tests.unit.intake.pipeline_env import (
    FINAL_STAGE,
    Landed,
    StageRun,
    describe,
    landed_run,
    partition_failures,
    read_ambiguity_list,
    run_stages,
)
from tests.unit.intake.structure_env import HELDOUT

pytestmark = pytest.mark.unit

AUDITED_STAGES = frozenset({"land", "structure", "roles", FINAL_STAGE})
METRICS_ARTIFACT = "run_metrics"
HELDOUT2 = HELDOUT.parent / "heldout2"


@dataclass(frozen=True)
class HeldOut:
    directory: Path
    schema: str
    descriptor: RunDescriptor


SOURCES = {
    "heldout": HeldOut(
        HELDOUT,
        "land_heldout",
        RunDescriptor(id="heldout", identity_system="heldout", release="2026-10-20"),
    ),
    "heldout2": HeldOut(
        HELDOUT2,
        "land_heldout2",
        RunDescriptor(id="heldout2", identity_system="heldout2", release="2026-10-21"),
    ),
}


@pytest.fixture(scope="module", params=sorted(SOURCES))
def gate(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[StageRun]:
    source = SOURCES[request.param]
    root = tmp_path_factory.mktemp(f"{request.param}_gate")
    with landed_run(source.directory, root, source.schema) as landed:
        run = run_stages(landed, source.descriptor)
        _print_counts(run)
        _write_metrics(run)
        yield run


def _write_metrics(run: StageRun) -> None:
    landed = run.landed
    metrics = run_metrics(landed.land, run.structure, run.roles, landed.record())
    landed.ledger.artifact(METRICS_ARTIFACT, metrics)
    print(f"{landed.land.relations[0].schema} run_metrics: {metrics.model_dump()}")


def _read_metrics(run: StageRun) -> RunMetrics:
    path = run.landed.ledger.directory / "artifacts" / f"{METRICS_ARTIFACT}.json"
    return RunMetrics.model_validate_json(path.read_text())


def _print_counts(run: StageRun) -> None:
    ledger = run.landed.ledger
    structure = read_ambiguity_list(ledger, "structure")
    roles = read_ambiguity_list(ledger, "roles")
    print(f"\nheld-out structure ambiguities: {structure.count}")
    for entry in structure.ambiguities:
        print(f"  structure {entry.subject}: {len(entry.members)} members {entry.members[:3]}")
    print(f"held-out role ambiguities: {roles.count}")
    for entry in roles.ambiguities:
        print(f"  roles {entry.subject}: {entry.members} hints {entry.hints}")
    print(f"held-out roles verdict: {run.roles_verdict.status} {describe(run.roles_verdict)}")
    print(f"held-out spec assembled: {run.spec is not None}")


def test_every_relation_is_accounted_for_exactly_once(gate: StageRun) -> None:
    assert partition_failures(gate.landed, gate.structure) == []


@pytest.mark.parametrize("gate", ["heldout"], indirect=True)
def test_land_verdict_passes(gate: StageRun) -> None:
    assert gate.land_verdict.status == "pass", describe(gate.land_verdict)


@pytest.mark.parametrize("gate", ["heldout2"], indirect=True)
def test_s2_baseline_land_verdict_of_the_second_held_out_source(gate: StageRun) -> None:
    """Recorded S2 baseline: the ragged CSV lands as one column named by its first line."""
    (relation,) = gate.landed.land.relations
    data_columns = [name for name, _ in relation.columns if not name.startswith("_sema_")]

    assert gate.land_verdict.status == "fail"
    assert [(f.check, f.relation) for f in gate.land_verdict.failures] == [
        ("header_row_valid", gate.landed.qualified(relation.name))
    ]
    assert data_columns == [gate.land_verdict.failures[0].column]
    assert gate.landed.land.headerless == ()


def test_structure_verdict_passes(gate: StageRun) -> None:
    assert gate.structure_verdict.status == "pass", describe(gate.structure_verdict)


def test_every_value_bearing_statement_is_over_a_schema_relation(gate: StageRun) -> None:
    landed: Landed = gate.landed
    lines = landed.value_bearing(AUDITED_STAGES)

    assert all(line["schema_relation"] for line in lines), [
        (line["stage"], line["statement"]) for line in lines if not line["schema_relation"]
    ]


def test_ambiguity_lists_are_written_to_the_ledger(gate: StageRun) -> None:
    ledger = gate.landed.ledger
    structure = read_ambiguity_list(ledger, "structure")
    roles = read_ambiguity_list(ledger, "roles")

    assert structure.ambiguities == gate.structure.ambiguities
    assert roles.ambiguities == gate.roles.ambiguities
    assert gate.landed.model_calls() == []


def test_run_metrics_are_written_to_the_ledger(gate: StageRun) -> None:
    written = _read_metrics(gate)
    landed = gate.landed

    assert written == run_metrics(landed.land, gate.structure, gate.roles, landed.record())


def test_run_metrics_calls_tokens_and_cost_equal_the_registry_row(gate: StageRun) -> None:
    written = _read_metrics(gate)
    record = gate.landed.record()

    assert (written.model_calls, written.tokens, written.cost) == (
        record.model_calls,
        record.tokens,
        record.cost,
    )


@pytest.mark.parametrize("gate", ["heldout"], indirect=True)
def test_s2_baseline_metrics(gate: StageRun) -> None:
    """Recorded S2 baseline with no model; S3 compares its slice values against these."""
    written = _read_metrics(gate)

    assert written.relations == 8
    assert written.bound == 1
    assert written.bound_fraction == pytest.approx(1 / 8)
    assert written.ignored == {}
    assert (written.open_structure, written.open_roles) == (7, 1)
    assert written.open_structure + written.open_roles == 8
    assert (written.proposals_made, written.proposals_bound) == (0, 0)
    assert (written.model_calls, written.tokens, written.cost) == (0, 0, 0.0)
    assert written.value_asks == 0


@pytest.mark.parametrize("gate", ["heldout2"], indirect=True)
def test_s2_baseline_metrics_of_the_second_held_out_source(gate: StageRun) -> None:
    """Recorded S2 baseline with no model, measured once when the source was sealed."""
    written = _read_metrics(gate)

    assert written.relations == 1
    assert written.bound == 1
    assert written.bound_fraction == pytest.approx(1.0)
    assert written.ignored == {}
    assert (written.open_structure, written.open_roles) == (0, 1)
    assert (written.proposals_made, written.proposals_bound) == (0, 0)
    assert (written.model_calls, written.tokens, written.cost) == (0, 0, 0.0)
    assert written.value_asks == 0

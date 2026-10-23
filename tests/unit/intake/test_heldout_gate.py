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
import yaml

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
HELDOUT3 = HELDOUT.parent / "heldout3"
HELDOUT4 = HELDOUT.parent / "heldout4"
HELDOUT5 = HELDOUT.parent / "heldout5"


@dataclass(frozen=True)
class LandBaseline:
    status: str
    failed_checks: tuple[str, ...]
    data_columns: int
    headerless: bool


# Recorded when the second held-out source was sealed (5469dcb), before task 1.8 fixed
# the CSV delimiter choice; kept as data so the post-fix run is read beside it.
HELDOUT2_S2_BASELINE_METRICS = RunMetrics(
    relations=1,
    ignored={},
    bound=1,
    bound_fraction=1.0,
    open_structure=0,
    open_roles=1,
    proposals_made=0,
    proposals_bound=0,
    model_calls=0,
    tokens=0,
    cost=0.0,
    value_asks=0,
)
HELDOUT2_S2_BASELINE_LAND = LandBaseline(
    status="fail", failed_checks=("header_row_valid",), data_columns=1, headerless=False
)
HELDOUT2_POST_FIX_METRICS = RunMetrics(
    relations=1,
    ignored={"no_header": 1},
    bound=0,
    bound_fraction=0.0,
    open_structure=0,
    open_roles=0,
    proposals_made=0,
    proposals_bound=0,
    model_calls=0,
    tokens=0,
    cost=0.0,
    value_asks=0,
)


@dataclass(frozen=True)
class GateThreshold:
    min_bound_fraction: float
    max_open_choices: int


@dataclass(frozen=True)
class GateMiss:
    metric: str
    value: float
    threshold: float

    def __str__(self) -> str:
        return f"{self.metric} {self.value} misses threshold {self.threshold}"


# Decided by Dean 2026-10-21 and committed before any model-backed run over either source
# (task 14.1). Open choices are counted at Review, before any decisions file.
GATE_THRESHOLDS = {
    "heldout": GateThreshold(min_bound_fraction=0.75, max_open_choices=2),
    "heldout2": GateThreshold(min_bound_fraction=1.0, max_open_choices=3),
    # Decided by Dean 2026-10-22 (S4 task 3.1), committed before any model-backed run over
    # either source.
    "heldout3": GateThreshold(min_bound_fraction=1.0, max_open_choices=2),
    "heldout4": GateThreshold(min_bound_fraction=1.0, max_open_choices=3),
    # Set by the supervisor 2026-10-23 (S5 decision 24), committed before any model-backed run.
    "heldout5": GateThreshold(min_bound_fraction=1.0, max_open_choices=4),
}


def gate_misses(metrics: RunMetrics, threshold: GateThreshold) -> tuple[GateMiss, ...]:
    misses: list[GateMiss] = []
    if metrics.bound_fraction < threshold.min_bound_fraction:
        misses.append(
            GateMiss("bound_fraction", metrics.bound_fraction, threshold.min_bound_fraction)
        )
    open_choices = metrics.open_structure + metrics.open_roles
    if open_choices > threshold.max_open_choices:
        misses.append(GateMiss("open_choices", open_choices, threshold.max_open_choices))
    return tuple(misses)


@dataclass(frozen=True)
class HeldOut:
    directory: Path
    schema: str
    descriptor: RunDescriptor


def _run_descriptor(directory: Path) -> RunDescriptor:
    path = directory / f"{directory.name}.run.yaml"
    return RunDescriptor.model_validate(yaml.safe_load(path.read_text()))


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
    "heldout3": HeldOut(HELDOUT3, "land_heldout3", _run_descriptor(HELDOUT3)),
    "heldout4": HeldOut(HELDOUT4, "land_heldout4", _run_descriptor(HELDOUT4)),
    "heldout5": HeldOut(HELDOUT5, "land_heldout5", _run_descriptor(HELDOUT5)),
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


def test_land_verdict_passes(gate: StageRun) -> None:
    assert gate.land_verdict.status == "pass", describe(gate.land_verdict)


@pytest.mark.parametrize("gate", ["heldout2"], indirect=True)
def test_the_ragged_csv_of_the_second_held_out_source_lands_headerless(
    gate: StageRun,
) -> None:
    (relation,) = gate.landed.land.relations
    data_columns = [name for name, _ in relation.columns if not name.startswith("_sema_")]

    assert gate.landed.land.headerless == (relation.name,)
    assert relation.header_row is None
    assert len(data_columns) > HELDOUT2_S2_BASELINE_LAND.data_columns


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
    """Recorded S2 baseline with no model; S3 compares its slice values against these.
    Re-pinned at task 2.2 (S5 decisions 3, 18, 18a): with no model the role map leaves every
    unsettled role of the bound relation open, so ``open_roles`` is 18, not 1."""
    written = _read_metrics(gate)

    assert written.relations == 8
    assert written.bound == 1
    assert written.bound_fraction == pytest.approx(1 / 8)
    assert written.ignored == {}
    assert (written.open_structure, written.open_roles) == (7, 18)
    assert written.open_structure == written.relations - written.bound
    assert (written.proposals_made, written.proposals_bound) == (0, 0)
    assert (written.model_calls, written.tokens, written.cost) == (0, 0, 0.0)
    assert written.value_asks == 0


@pytest.mark.parametrize("gate", ["heldout2"], indirect=True)
def test_post_fix_metrics_of_the_second_held_out_source(gate: StageRun) -> None:
    """Measured once after task 1.8, beside the recorded S2 baseline (SOURCE.md)."""
    assert _read_metrics(gate) == HELDOUT2_POST_FIX_METRICS


def _synthetic(bound_fraction: float, open_structure: int, open_roles: int) -> RunMetrics:
    return HELDOUT2_POST_FIX_METRICS.model_copy(
        update={
            "bound_fraction": bound_fraction,
            "open_structure": open_structure,
            "open_roles": open_roles,
        }
    )


def test_the_thresholds_are_committed_for_every_held_out_source() -> None:
    assert GATE_THRESHOLDS.keys() == SOURCES.keys()
    assert GATE_THRESHOLDS["heldout"] == GateThreshold(
        min_bound_fraction=0.75, max_open_choices=2
    )
    assert GATE_THRESHOLDS["heldout2"] == GateThreshold(
        min_bound_fraction=1.0, max_open_choices=3
    )
    assert GATE_THRESHOLDS["heldout3"] == GateThreshold(
        min_bound_fraction=1.0, max_open_choices=2
    )
    assert GATE_THRESHOLDS["heldout4"] == GateThreshold(
        min_bound_fraction=1.0, max_open_choices=3
    )
    assert GATE_THRESHOLDS["heldout5"] == GateThreshold(
        min_bound_fraction=1.0, max_open_choices=4
    )


def test_a_bound_fraction_below_the_minimum_names_the_metric_value_and_threshold() -> None:
    threshold = GateThreshold(min_bound_fraction=0.75, max_open_choices=2)

    misses = gate_misses(_synthetic(0.5, 0, 1), threshold)

    assert misses == (GateMiss(metric="bound_fraction", value=0.5, threshold=0.75),)
    assert "bound_fraction" in str(misses[0])
    assert "0.5" in str(misses[0]) and "0.75" in str(misses[0])


def test_open_choices_over_the_maximum_name_the_metric_value_and_threshold() -> None:
    threshold = GateThreshold(min_bound_fraction=0.75, max_open_choices=2)

    misses = gate_misses(_synthetic(1.0, 2, 1), threshold)

    assert misses == (GateMiss(metric="open_choices", value=3, threshold=2),)
    assert "open_choices" in str(misses[0])
    assert "3" in str(misses[0]) and "2" in str(misses[0])


def test_a_run_missing_both_reports_both_misses() -> None:
    threshold = GateThreshold(min_bound_fraction=1.0, max_open_choices=3)

    misses = gate_misses(_synthetic(0.0, 4, 0), threshold)

    assert [miss.metric for miss in misses] == ["bound_fraction", "open_choices"]


def test_a_run_meeting_both_thresholds_has_no_miss() -> None:
    threshold = GateThreshold(min_bound_fraction=0.75, max_open_choices=2)

    assert gate_misses(_synthetic(0.75, 1, 1), threshold) == ()

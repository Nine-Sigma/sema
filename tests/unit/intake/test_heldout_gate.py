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

import pytest

from sema.intake.eligibility_utils import RunDescriptor
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

SCHEMA = "land_heldout"
AUDITED_STAGES = frozenset({"land", "structure", "roles", FINAL_STAGE})
DESCRIPTOR = RunDescriptor(id="heldout", identity_system="heldout", release="2026-10-20")


@pytest.fixture(scope="module")
def gate(tmp_path_factory: pytest.TempPathFactory) -> Iterator[StageRun]:
    with landed_run(HELDOUT, tmp_path_factory.mktemp("heldout_gate"), SCHEMA) as landed:
        run = run_stages(landed, DESCRIPTOR)
        _print_counts(run)
        yield run


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

"""Run metrics (design Decision 13): computed from the stage results and the registry row."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from sema.intake.eligibility_spec_utils import RolesResult
from sema.intake.landing import LandedRelation, LandResult, Provenance
from sema.intake.ledger import Ledger
from sema.intake.metrics import RunMetrics, run_metrics
from sema.intake.registry import RunRecord
from sema.intake.shapes import concept_list, entity_model, ignore
from sema.intake.shapes.base import SlotBinding
from sema.intake.structure import BoundEntry, StructureResult
from sema.intake.verify.base import Ambiguity

pytestmark = pytest.mark.unit

SCHEMA = "land_any"
METRIC_NAMES = {
    "relations",
    "ignored",
    "bound",
    "bound_fraction",
    "open_structure",
    "open_roles",
    "proposals_made",
    "proposals_bound",
    "model_calls",
    "tokens",
    "cost",
    "value_asks",
}


def _land(*names: str) -> LandResult:
    provenance = Provenance(
        source_uri="file:///any",
        loader="tabular",
        loader_version="1",
        landed_at="2026-10-21T00:00:00+00:00",
        policy_class="public",
        locator_kind="row",
    )
    relations = tuple(
        LandedRelation(SCHEMA, name, (("a", "VARCHAR"),), 1, provenance) for name in names
    )
    return LandResult("tabular", (("source", "any"),), relations, "public")


def _bound(template: str, *relations: str) -> BoundEntry:
    binding = SlotBinding(template=template, slots={}, relations=relations)
    return BoundEntry(binding=binding, source="code")


def _ignored(relation: str, reason: str) -> SlotBinding:
    slots = {ignore.REASON: reason, ignore.WOULD_MATCH: ""}
    return SlotBinding(template=ignore.ID, slots=slots, relations=(relation,))


def _open(stage: str, subject: str) -> Ambiguity:
    return Ambiguity(stage=stage, subject=subject, members=("first", "second"))


def _chosen(stage: str, subject: str) -> Ambiguity:
    return Ambiguity(
        stage=stage,
        subject=subject,
        members=("first", "second"),
        choice="first",
        confidence=0.9,
        source="model",
    )


def _structure(
    bound: tuple[BoundEntry, ...] = (),
    ignored: tuple[SlotBinding, ...] = (),
    ambiguities: tuple[Ambiguity, ...] = (),
) -> StructureResult:
    return StructureResult(
        order=(),
        groups=(),
        signatures={},
        bound=bound,
        ignored=ignored,
        ambiguities=ambiguities,
        edges=(),
        floors={},
    )


def _roles(structure: StructureResult, *ambiguities: Ambiguity) -> RolesResult:
    return RolesResult(
        eligibilities=(),
        ambiguities=ambiguities,
        joins=(),
        headers={},
        structure=structure,
        refs={},
    )


def _record(model_calls: int = 0, tokens: int = 0, cost: float = 0.0) -> RunRecord:
    now = datetime(2026, 10, 21, tzinfo=UTC)
    return RunRecord(
        run_id="run-1",
        source="any",
        stage="roles",
        status="running",
        approver=None,
        started_at=now,
        updated_at=now,
        model_calls=model_calls,
        tokens=tokens,
        cost=cost,
    )


def _partitioned() -> tuple[LandResult, StructureResult]:
    land = _land("r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8")
    structure = _structure(
        bound=(_bound(entity_model.ID, "r1", "r2"), _bound(concept_list.ID, "r3")),
        ignored=(
            _ignored("r4", ignore.NO_ROWS),
            _ignored("r5", ignore.NO_ROWS),
            _ignored("r6", ignore.BOILERPLATE),
        ),
        ambiguities=(_open("structure", "r7"), _open("structure", "r8")),
    )
    return land, structure


def test_metric_names_are_exactly_the_reported_set() -> None:
    land, structure = _partitioned()

    metrics = run_metrics(land, structure, _roles(structure), _record())

    assert set(metrics.model_dump()) == METRIC_NAMES


def test_relations_count_every_landed_relation() -> None:
    land, structure = _partitioned()

    metrics = run_metrics(land, structure, _roles(structure), _record())

    assert metrics.relations == 8


def test_bound_counts_relations_not_bindings() -> None:
    land, structure = _partitioned()

    metrics = run_metrics(land, structure, _roles(structure), _record())

    assert metrics.bound == 3
    assert metrics.bound_fraction == pytest.approx(3 / 8)


def test_ignored_counts_relations_by_reason() -> None:
    land, structure = _partitioned()

    metrics = run_metrics(land, structure, _roles(structure), _record())

    assert metrics.ignored == {ignore.NO_ROWS: 2, ignore.BOILERPLATE: 1}


def test_nothing_ignored_is_an_empty_mapping() -> None:
    structure = _structure(bound=(_bound(concept_list.ID, "r1"),))

    metrics = run_metrics(_land("r1"), structure, _roles(structure), _record())

    assert metrics.ignored == {}


def test_open_choices_count_only_entries_without_a_choice() -> None:
    land = _land("r1", "r2", "r3")
    structure = _structure(
        ambiguities=(_open("structure", "r1"), _open("structure", "r2"), _chosen("structure", "r3"))
    )
    roles = _roles(structure, _open("roles", "r1.section"), _chosen("roles", "r2.code"))

    metrics = run_metrics(land, structure, roles, _record())

    assert (metrics.open_structure, metrics.open_roles) == (2, 1)


def test_a_run_that_has_not_reached_roles_has_no_open_role_choice() -> None:
    land, structure = _partitioned()

    metrics = run_metrics(land, structure, None, _record())

    assert (metrics.open_structure, metrics.open_roles) == (2, 0)


def test_call_token_and_cost_values_equal_the_registry_row() -> None:
    land, structure = _partitioned()
    record = _record(model_calls=3, tokens=1840, cost=0.0425)

    metrics = run_metrics(land, structure, _roles(structure), record)

    assert (metrics.model_calls, metrics.tokens, metrics.cost) == (3, 1840, 0.0425)


def test_proposals_and_value_asks_are_reported_as_given() -> None:
    land, structure = _partitioned()

    metrics = run_metrics(
        land,
        structure,
        _roles(structure),
        _record(),
        proposals_made=3,
        proposals_bound=2,
        value_asks=1,
    )

    assert (metrics.proposals_made, metrics.proposals_bound, metrics.value_asks) == (3, 2, 1)


def test_proposals_and_value_asks_default_to_zero() -> None:
    land, structure = _partitioned()

    metrics = run_metrics(land, structure, _roles(structure), _record())

    assert (metrics.proposals_made, metrics.proposals_bound, metrics.value_asks) == (0, 0, 0)


def test_bound_fraction_of_an_empty_landing_is_zero() -> None:
    structure = _structure()

    metrics = run_metrics(_land(), structure, _roles(structure), _record())

    assert (metrics.relations, metrics.bound, metrics.bound_fraction) == (0, 0, 0.0)


def test_metrics_are_frozen() -> None:
    land, structure = _partitioned()
    metrics = run_metrics(land, structure, _roles(structure), _record())

    with pytest.raises(ValidationError):
        metrics.bound = 0  # type: ignore[misc]


def test_metrics_round_trip_through_the_ledger_artifact(tmp_path: Path) -> None:
    land, structure = _partitioned()
    metrics = run_metrics(land, structure, _roles(structure), _record(3, 1840, 0.0425))
    ledger = Ledger(tmp_path).run("run-1")

    path = ledger.artifact("run_metrics", metrics)

    assert path.name == "run_metrics.json"
    assert RunMetrics.model_validate_json(path.read_text()) == metrics

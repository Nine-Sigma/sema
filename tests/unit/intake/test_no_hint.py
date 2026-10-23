"""Every committed source with no hand-written spec (task 14.5, generality-gate spec).

Each source replays its committed recording with no hint and zero provider calls (see
``no_hint_env``). Known sources must reach Review with an assembled spec whose import
equals the import of the source's hand-written spec on the manifest and on term staging
(at the spec's cap and at a cap of one, which stages every term), with the spec diff
printed. A1 settles after its committed ignore-only decisions file and is compared on
``concepts`` and ``value_sets`` only. A7's Athena bundle has no hand-written spec, so its
oracle is the loader golden: the landed tables and the staged terms. A10 is never an
expected failure; any other known-source or held-out miss is an ``xfail(strict=True)``
carrying the spec diff, the missed metric, or the rejection, and together they are the S3
baseline (Dean, 2026-10-22). Every leg with a spec oracle is also scored per import cell kind
(``sema.intake.scorecard``) and held at its floor in ``scorecard_env.BASELINE``, with the run's
relation and ignored counts printed beside the table; a fresh held-out is scored against its
reviewed spec with no spec diff printed. Held-out ledger text is never printed, only counts
and metrics.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import pytest
import yaml

from sema.binding.spec import ConceptSourceSpec, load_concept_source_spec
from sema.intake.scorecard import render
from sema.log import logger
from tests.support.athena_bundle import export_schema
from tests.unit.intake.no_hint_env import (
    DECISIONS_FILE,
    RECORDED_CALL_CAP,
    RECORDED_COST_CAP,
    Replayed,
    assert_zero_provider_calls,
    capped,
    column,
    csv_blocks,
    decide,
    imported,
    marked_param,
    metrics_of,
    recording,
    replay,
    scorecard_of,
    source_of,
    spec_diff,
    spec_of,
    staging_copy,
    text_diff,
    uncovered_reads,
    utils,
)
from tests.unit.intake.scorecard_env import BASELINE, ORACLES, UNREVIEWED, floor_misses
from tests.unit.intake.test_athena_golden import GOLDEN
from tests.unit.intake.test_heldout_gate import GATE_THRESHOLDS, gate_misses

pytestmark = pytest.mark.unit

T = TypeVar("T")

FIXTURES: Path = utils().FIXTURES
WRITTEN_SPECS = {name: ORACLES[name] for name in ("a2", "a3", "a7_csv", "a10")}
NO_HINT_SOURCES = ("a2", "a3", "a7_csv", "a7_athena", "a10")
HELD_OUT = ("heldout", "heldout2", "heldout3", "heldout4", "heldout5")
DECIDED_HELD_OUT = ("heldout", "heldout3", "heldout4")
UNDECIDED_HELD_OUT = ("heldout2", "heldout5")
A1, ATHENA = "a1", "a7_athena"
A1_SPEC = ORACLES[A1]
A1_DECISIONS = FIXTURES / "partner_sl" / DECISIONS_FILE
SCORED_SPECS = ORACLES
DECISIONS = {
    A1: A1_DECISIONS,
    **{name: source_of(name).path / DECISIONS_FILE for name in DECIDED_HELD_OUT},
}
SPEC_HEAD = ("spec_version", "id", "identity_system", "release", "default_provider", "scheme")
A1_SECTIONS = ("concepts", "value_sets")
CAP_VALUES: dict[str, int | None] = {"spec_cap": None, "stage_all": 1}
CAPS = pytest.mark.parametrize("cap", list(CAP_VALUES.values()), ids=list(CAP_VALUES))
MANIFEST_DIFF_LINES = 200
UNREVIEWED_REASON = "unreviewed: the oracle is not reviewed_correct, so no floor row"
A7_CSV_MISS = (
    "S3 baseline: the A7 CSV manifest misses the roles type, required, primary_key,"
    " fk_entity, fk_property, domain, description"
)
IMPORT_MISSES = {
    ("a2", "stage_all"): "S3 baseline: A2 at cap 1 misses replaced_by in term staging",
    ("a7_csv", "spec_cap"): A7_CSV_MISS,
    ("a7_csv", "stage_all"): A7_CSV_MISS,
}
HELD_OUT_UNASSEMBLED = (
    "with its two ignores accepted the decided College Scorecard run settles every entry but"
    " assembles no spec: its per-file dictionaries do not share one role map and a concept"
    " section has no settled name role (S4 closure C1, 2026-10-23) (bug-846)"
)
DECIDED_MISSES = {"heldout": HELD_OUT_UNASSEMBLED}
GATE_MISSES = {
    "heldout": "College Scorecard misses its gate at Review: bound_fraction 0.625 < 0.75"
    " and open_choices 6 > 2 (Dean, 2026-10-22: recorded, carried to S4)",
    "heldout2": "PUMS misses its gate at Review: bound_fraction 0.0 < 1.0, its only"
    " relation ignored no_header and its layout proposal unbound (Dean, 2026-10-22)",
    "heldout3": "heldout3 misses its gate at Review: bound_fraction 0.5 < 1.0"
    " (S4 task 3.4, 2026-10-22: recorded, carried to S4 review)",
    "heldout4": "heldout4 misses its gate at Review: bound_fraction 0.0 < 1.0, both relations"
    " open on a section conflict (S4 decision 18, 2026-10-22: recorded, carried to S4 review)",
    "heldout5": "heldout5 is sealed with no recording yet (S5 task 1.2, 2026-10-23); replaced at"
    " task 1.6",
}


class Runs:
    """One replay per source per module, and one decisions pass over it.

    A held-out run logs nothing and, when it raises, fails with the exception type only: a
    traceback or captured log would print held-out relation names and decision subjects.
    """

    def __init__(self, factory: pytest.TempPathFactory) -> None:
        self._factory = factory
        self._replayed: dict[str, Replayed] = {}
        self._decided: dict[str, Any] = {}

    def replayed(self, name: str) -> Replayed:
        if name not in self._replayed:
            root = self._factory.mktemp(name)
            self._replayed[name] = _sealed(name, lambda: replay(name, root))
        return self._replayed[name]

    def decided(self, name: str, decisions: Path) -> Any:
        if name not in self._decided:
            run = self.replayed(name)
            self._decided[name] = _sealed(name, lambda: decide(run, decisions))
        return self._decided[name]


def _sealed(name: str, step: Callable[[], T]) -> T:
    if name not in HELD_OUT:
        return step()
    logger.disable("sema")
    try:
        return step()
    except Exception as error:  # noqa: BLE001 - only the type may leave a held-out run
        failure = type(error).__name__
    finally:
        logger.enable("sema")
    pytest.fail(f"{name}: the run raised {failure}; held-out details withheld", pytrace=False)


@pytest.fixture(scope="module")
def runs(tmp_path_factory: pytest.TempPathFactory) -> Runs:
    return Runs(tmp_path_factory)


def _assert_imports_equal(
    run: Replayed, spec: ConceptSourceSpec, written: ConceptSourceSpec, cap: int | None,
    directory: Path,
) -> None:
    diff = spec_diff(spec, written)
    print(f"\n{run.name} spec diff (written -> assembled):\n{diff}")
    with staging_copy(run, directory) as staging:
        _, manifest, staged = imported(capped(spec, cap), staging)
        _, expected_manifest, expected_staged = imported(capped(written, cap), staging)

    manifest_diff = "\n".join(
        text_diff(expected_manifest, manifest).splitlines()[:MANIFEST_DIFF_LINES]
    )
    assert manifest == expected_manifest, f"{diff}\n\nmanifest diff:\n{manifest_diff}"
    assert staged == expected_staged, f"{diff}\n\nterm staging diff:\n" + text_diff(
        expected_staged, staged
    )


@pytest.mark.parametrize("name", NO_HINT_SOURCES)
def test_known_source_reaches_review_with_a_spec_and_no_provider_call(
    runs: Runs, name: str
) -> None:
    run = runs.replayed(name)
    recorded = recording(name)

    assert run.outcome.node == "review"
    assert spec_of(run.outcome) is not None, metrics_of(run.outcome)
    assert_zero_provider_calls(run)
    assert recorded["calls"] <= RECORDED_CALL_CAP
    assert recorded["cost"] <= RECORDED_COST_CAP


def _import_leg(name: str, cap_id: str) -> Any:
    reason = IMPORT_MISSES.get((name, cap_id))
    marks = [] if reason is None else [
        pytest.mark.xfail(strict=True, raises=AssertionError, reason=reason)
    ]
    return pytest.param(name, CAP_VALUES[cap_id], id=f"{name}-{cap_id}", marks=marks)


@pytest.mark.parametrize(
    ("name", "cap"),
    [_import_leg(name, cap_id) for name in sorted(WRITTEN_SPECS) for cap_id in CAP_VALUES],
)
def test_known_source_import_equals_the_written_spec_import(
    runs: Runs, name: str, cap: int | None, tmp_path: Path
) -> None:
    run = runs.replayed(name)
    written = load_concept_source_spec(WRITTEN_SPECS[name])
    spec = spec_of(run.outcome)
    assert spec is not None, spec_diff(spec, written)

    _assert_imports_equal(run, spec, written, cap, tmp_path)


def test_athena_landed_tables_equal_the_loader_golden(runs: Runs, tmp_path: Path) -> None:
    with staging_copy(runs.replayed(ATHENA), tmp_path) as staging:
        landed = csv_blocks(export_schema(staging))
    golden = csv_blocks(GOLDEN.read_text())

    assert landed, "nothing landed in the vocabulary schema"
    assert landed == {table: golden.get(table) for table in landed}


def test_athena_staged_terms_equal_the_golden_concept_rows(
    runs: Runs, tmp_path: Path
) -> None:
    run = runs.replayed(ATHENA)
    spec = spec_of(run.outcome)
    assert spec is not None

    with staging_copy(run, tmp_path) as staging:
        _, _, staged = imported(capped(spec, 1), staging)
    concept = csv_blocks(GOLDEN.read_text())["concept"]
    terms = csv_blocks(staged)["concept"]

    expected = sorted(zip(column(concept, "concept_id"), column(concept, "concept_name")))
    assert sorted(zip(column(terms, "concept_code"), column(terms, "concept_name"))) == (
        expected
    ), staged


def test_a1_reaches_review_with_no_hint_and_no_provider_call(runs: Runs) -> None:
    run = runs.replayed(A1)

    assert run.outcome.node == "review"
    assert_zero_provider_calls(run)


def test_a1_decisions_settle_every_entry_with_zero_further_calls(runs: Runs) -> None:
    outcome = runs.decided(A1, A1_DECISIONS)
    metrics = metrics_of(outcome)

    assert outcome.node == "review"
    assert spec_of(outcome) is not None, metrics
    assert (metrics.open_structure, metrics.open_roles) == (0, 0)
    assert_zero_provider_calls(runs.replayed(A1))


def _a1_scoped(raw: dict[str, Any]) -> ConceptSourceSpec:
    return ConceptSourceSpec.model_validate(
        {key: raw[key] for key in (*SPEC_HEAD, *A1_SECTIONS) if key in raw}
    )


def _a1_pair(spec: ConceptSourceSpec) -> tuple[ConceptSourceSpec, ConceptSourceSpec]:
    """The A1 assembled and written specs, both scoped to ``concepts`` and ``value_sets``."""
    written = _a1_scoped(yaml.safe_load(A1_SPEC.read_text())["concept_source"])
    assembled = _a1_scoped(spec.model_dump(mode="json", by_alias=True, exclude_none=True))
    return assembled, written


@CAPS
def test_a1_import_equals_the_written_spec_on_concepts_and_value_sets(
    runs: Runs, cap: int | None, tmp_path: Path
) -> None:
    spec = spec_of(runs.decided(A1, A1_DECISIONS))
    assert spec is not None
    assembled, written = _a1_pair(spec)

    _assert_imports_equal(runs.replayed(A1), assembled, written, cap, tmp_path)


def _settled_outcome(runs: Runs, name: str) -> Any:
    decisions = DECISIONS.get(name)
    return runs.replayed(name).outcome if decisions is None else runs.decided(name, decisions)


def _scored_specs(outcome: Any, name: str) -> tuple[ConceptSourceSpec, ConceptSourceSpec]:
    """The assembled spec of a scorecard leg and its full oracle, neither scoped."""
    spec = spec_of(outcome)
    assert spec is not None, metrics_of(outcome)
    return spec, load_concept_source_spec(SCORED_SPECS[name])


def _scorecard_leg(name: str, cap_id: str) -> Any:
    marks = [] if name not in UNREVIEWED else [
        pytest.mark.xfail(strict=True, raises=AssertionError, reason=UNREVIEWED_REASON)
    ]
    return pytest.param(name, cap_id, id=f"{name}-{cap_id}", marks=marks)


@pytest.mark.parametrize(
    ("name", "cap_id"),
    [_scorecard_leg(name, cap_id) for name in sorted(SCORED_SPECS) for cap_id in CAP_VALUES],
)
def test_known_source_scorecard_meets_its_floor(
    runs: Runs, name: str, cap_id: str, tmp_path: Path
) -> None:
    run, outcome = runs.replayed(name), _settled_outcome(runs, name)
    metrics, cap = metrics_of(outcome), CAP_VALUES[cap_id]
    spec, oracle = _sealed(name, lambda: _scored_specs(outcome, name))
    card = _sealed(name, lambda: scorecard_of(run, spec, oracle, cap, tmp_path))
    table, ignored = render(card), sum(metrics.ignored.values())
    if name not in HELD_OUT:
        print(f"\n{name}-{cap_id} spec diff (oracle -> assembled):\n{spec_diff(spec, oracle)}")
    print(f"\n{table}\nrelations {metrics.relations} ignored {ignored}")

    floor = BASELINE.get((name, cap_id))
    assert floor is not None, f"no scorecard floor for {name}-{cap_id}\n{table}"
    assert floor_misses(card, ignored, floor) == (), table


@pytest.mark.parametrize("name", HELD_OUT)
def test_held_out_source_reaches_review_with_no_hint(runs: Runs, name: str) -> None:
    run = runs.replayed(name)
    node = run.outcome.node
    print(f"\n{name} run_metrics at Review: {metrics_of(run.outcome).model_dump()}")

    assert node == "review"
    assert_zero_provider_calls(run)


@pytest.mark.parametrize(
    "name", [marked_param(name, GATE_MISSES, raises=AssertionError) for name in HELD_OUT]
)
def test_held_out_source_meets_its_committed_thresholds(runs: Runs, name: str) -> None:
    misses = gate_misses(metrics_of(runs.replayed(name).outcome), GATE_THRESHOLDS[name])

    assert misses == (), [str(miss) for miss in misses]


def test_every_held_out_source_is_either_decided_or_undecided() -> None:
    assert sorted((*DECIDED_HELD_OUT, *UNDECIDED_HELD_OUT)) == sorted(HELD_OUT)


@pytest.mark.parametrize(
    "name", [marked_param(name, DECIDED_MISSES) for name in DECIDED_HELD_OUT]
)
def test_held_out_decisions_assemble_with_zero_further_calls(runs: Runs, name: str) -> None:
    decisions = DECISIONS[name]
    assert decisions.is_file(), f"the committed {name} decisions file is missing"

    outcome = runs.decided(name, decisions)
    node, assembled = outcome.node, spec_of(outcome) is not None
    metrics = metrics_of(outcome).model_dump()

    assert node == "review"
    assert assembled, metrics
    assert (metrics["open_structure"], metrics["open_roles"]) == (0, 0), metrics
    assert_zero_provider_calls(runs.replayed(name))


@pytest.mark.parametrize("name", UNDECIDED_HELD_OUT)
def test_undecided_held_out_source_has_no_open_entry_and_no_decisions_file(
    runs: Runs, name: str
) -> None:
    metrics = metrics_of(runs.replayed(name).outcome)

    assert not (source_of(name).path / DECISIONS_FILE).exists()
    assert (metrics.open_structure, metrics.open_roles) == (0, 0)


@pytest.mark.parametrize(
    "name", [marked_param(name, DECIDED_MISSES) for name in (A1, *NO_HINT_SOURCES, *HELD_OUT)]
)
def test_a_review_with_a_binding_and_no_open_entry_presents_a_spec(runs: Runs, name: str) -> None:
    outcome = _settled_outcome(runs, name)
    metrics = metrics_of(outcome)
    settled = outcome.node == "review" and metrics.bound > 0
    settled = settled and (metrics.open_structure, metrics.open_roles) == (0, 0)
    assembled = _sealed(name, lambda: spec_of(outcome) is not None)

    assert assembled or not settled, (
        "held-out details withheld" if name in HELD_OUT else metrics.model_dump()
    )


@pytest.mark.parametrize("name", (A1, *NO_HINT_SOURCES, *HELD_OUT))
def test_every_value_bearing_read_is_schema_text_or_value_approved(
    runs: Runs, name: str
) -> None:
    uncovered = uncovered_reads(runs.replayed(name))

    count = len(uncovered)
    assert count == 0, "held-out statements withheld" if name in HELD_OUT else uncovered

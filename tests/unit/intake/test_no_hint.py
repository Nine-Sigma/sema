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
relation and ignored counts printed beside the table. Held-out ledger text is never printed,
only counts and metrics.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import pytest
import yaml

from sema.binding.spec import ConceptSourceSpec, load_concept_source_spec
from sema.intake.metrics import RunMetrics
from sema.intake.model_calls import BUDGET_REASON
from sema.intake.scorecard import render, score
from sema.log import logger
from tests.support.athena_bundle import export_schema
from tests.unit.intake.no_hint_env import (
    DECISIONS_FILE,
    RECORDED_CALL_CAP,
    RECORDED_COST_CAP,
    Replayed,
    capped,
    csv_blocks,
    decide,
    imported,
    metrics_of,
    recording,
    replay,
    source_of,
    spec_diff,
    spec_of,
    staging_copy,
    text_diff,
    uncovered_reads,
    utils,
)
from tests.unit.intake.scorecard_env import BASELINE, floor_misses
from tests.unit.intake.test_athena_golden import GOLDEN
from tests.unit.intake.test_heldout_gate import GATE_THRESHOLDS, gate_misses

pytestmark = pytest.mark.unit

T = TypeVar("T")

FIXTURES: Path = utils().FIXTURES
WRITTEN_SPECS = {
    "a2": FIXTURES / "tabular" / "codes_a2.spec.yaml",
    "a3": FIXTURES / "semsql" / "ontology.spec.yaml",
    "a7_csv": FIXTURES / "omop" / "cdm_field_level.spec.yaml",
    "a10": FIXTURES / "acris" / "acris.spec.yaml",
}
NO_HINT_SOURCES = ("a2", "a3", "a7_csv", "a7_athena", "a10")
HELD_OUT = ("heldout", "heldout2", "heldout3", "heldout4")
A1, ATHENA = "a1", "a7_athena"
A1_SPEC = FIXTURES / "partner_sl" / "standard_library.spec.yaml"
A1_DECISIONS = FIXTURES / "partner_sl" / DECISIONS_FILE
SPEC_HEAD = ("spec_version", "id", "identity_system", "release", "default_provider", "scheme")
A1_SECTIONS = ("concepts", "value_sets")
CAP_VALUES: dict[str, int | None] = {"spec_cap": None, "stage_all": 1}
CAPS = pytest.mark.parametrize("cap", list(CAP_VALUES.values()), ids=list(CAP_VALUES))
MANIFEST_DIFF_LINES = 200
A7_CSV_MISS = (
    "S3 baseline: the A7 CSV manifest misses the roles type, required, primary_key,"
    " fk_entity, fk_property, domain, description"
)
IMPORT_MISSES = {
    ("a2", "stage_all"): "S3 baseline: A2 at cap 1 misses replaced_by in term staging",
    ("a7_csv", "spec_cap"): A7_CSV_MISS,
    ("a7_csv", "stage_all"): A7_CSV_MISS,
}
HELD_OUT_DECISIONS_REJECTED = (
    "the committed held-out decisions file holds two `no_template` choices rejected since"
    " 14.4b; they become `ignore` with the 14.4c re-record in S4"
)
GATE_MISSES = {
    "heldout": "College Scorecard misses its gate at Review: bound_fraction 0.625 < 0.75"
    " and open_choices 6 > 2 (Dean, 2026-10-22: recorded, carried to S4)",
    "heldout2": "PUMS misses its gate at Review: bound_fraction 0.0 < 1.0, its only"
    " relation ignored no_header and its layout proposal unbound (Dean, 2026-10-22)",
    "heldout3": "sealed, no recording yet (S4 task 3.2)",
    "heldout4": "sealed, no recording yet (S4 task 3.2)",
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


def _assert_zero_provider_calls(run: Replayed) -> None:
    row = run.row()
    calls, reason, invoked = row.model_calls, row.reason, run.client.invoke.call_count

    assert (calls, invoked) == (0, 0)
    assert reason != BUDGET_REASON


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
    _assert_zero_provider_calls(run)
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


def _column(block: list[list[str]], name: str) -> list[str]:
    header, *rows = block
    index = [cell.partition(":")[0] for cell in header].index(name)
    return [row[index] for row in rows]


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

    expected = sorted(zip(_column(concept, "concept_id"), _column(concept, "concept_name")))
    assert sorted(zip(_column(terms, "concept_code"), _column(terms, "concept_name"))) == (
        expected
    ), staged


def test_a1_reaches_review_with_no_hint_and_no_provider_call(runs: Runs) -> None:
    run = runs.replayed(A1)

    assert run.outcome.node == "review"
    _assert_zero_provider_calls(run)


def test_a1_decisions_settle_every_entry_with_zero_further_calls(runs: Runs) -> None:
    outcome = runs.decided(A1, A1_DECISIONS)
    metrics = metrics_of(outcome)

    assert outcome.node == "review"
    assert spec_of(outcome) is not None, metrics
    assert (metrics.open_structure, metrics.open_roles) == (0, 0)
    _assert_zero_provider_calls(runs.replayed(A1))


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


def _scored_specs(
    runs: Runs, name: str
) -> tuple[ConceptSourceSpec, ConceptSourceSpec, RunMetrics]:
    """The assembled and oracle specs of a scorecard leg and the metrics of the run that
    assembled it (A1's after its decisions)."""
    outcome = runs.decided(A1, A1_DECISIONS) if name == A1 else runs.replayed(name).outcome
    spec = spec_of(outcome)
    assert spec is not None, metrics_of(outcome)
    if name == A1:
        return (*_a1_pair(spec), metrics_of(outcome))
    return spec, load_concept_source_spec(WRITTEN_SPECS[name]), metrics_of(outcome)


@pytest.mark.parametrize(
    ("name", "cap_id"),
    [
        pytest.param(name, cap_id, id=f"{name}-{cap_id}")
        for name in (*sorted(WRITTEN_SPECS), A1)
        for cap_id in CAP_VALUES
    ],
)
def test_known_source_scorecard_meets_its_floor(
    runs: Runs, name: str, cap_id: str, tmp_path: Path
) -> None:
    spec, written, metrics = _scored_specs(runs, name)
    cap, ignored = CAP_VALUES[cap_id], sum(metrics.ignored.values())
    with staging_copy(runs.replayed(name), tmp_path) as staging:
        manifest, _, staged = imported(capped(spec, cap), staging)
        expected_manifest, _, expected_staged = imported(capped(written, cap), staging)
    card = score(expected_manifest, manifest, expected_staged, staged)
    table = render(card)
    print(f"\n{name}-{cap_id} spec diff (written -> assembled):\n{spec_diff(spec, written)}")
    print(f"{table}\nrelations {metrics.relations} ignored {ignored}")

    floor = BASELINE.get((name, cap_id))
    assert floor is not None, f"no scorecard floor for {name}-{cap_id}\n{table}"
    assert floor_misses(card, ignored, floor) == (), table


@pytest.mark.parametrize("name", HELD_OUT)
def test_held_out_source_reaches_review_with_no_hint(runs: Runs, name: str) -> None:
    run = runs.replayed(name)
    node = run.outcome.node
    print(f"\n{name} run_metrics at Review: {metrics_of(run.outcome).model_dump()}")

    assert node == "review"
    _assert_zero_provider_calls(run)


@pytest.mark.parametrize(
    "name",
    [
        pytest.param(
            name,
            marks=pytest.mark.xfail(strict=True, raises=AssertionError, reason=GATE_MISSES[name]),
        )
        for name in HELD_OUT
    ],
)
def test_held_out_source_meets_its_committed_thresholds(runs: Runs, name: str) -> None:
    misses = gate_misses(metrics_of(runs.replayed(name).outcome), GATE_THRESHOLDS[name])

    assert misses == (), [str(miss) for miss in misses]


@pytest.mark.xfail(strict=True, reason=HELD_OUT_DECISIONS_REJECTED)
def test_held_out_decisions_assemble_with_zero_further_calls(runs: Runs) -> None:
    decisions = source_of("heldout").path / DECISIONS_FILE
    assert decisions.is_file(), "the committed held-out decisions file is missing"

    outcome = runs.decided("heldout", decisions)
    node, assembled = outcome.node, spec_of(outcome) is not None
    metrics = metrics_of(outcome).model_dump()

    assert node == "review"
    assert assembled, metrics
    assert (metrics["open_structure"], metrics["open_roles"]) == (0, 0), metrics
    _assert_zero_provider_calls(runs.replayed("heldout"))


def test_second_held_out_source_has_no_open_entry_and_no_decisions_file(runs: Runs) -> None:
    metrics = metrics_of(runs.replayed("heldout2").outcome)

    assert not (source_of("heldout2").path / DECISIONS_FILE).exists()
    assert (metrics.open_structure, metrics.open_roles) == (0, 0)


@pytest.mark.parametrize("name", (A1, *NO_HINT_SOURCES, *HELD_OUT))
def test_every_value_bearing_read_is_schema_text_or_value_approved(
    runs: Runs, name: str
) -> None:
    uncovered = uncovered_reads(runs.replayed(name))

    count = len(uncovered)
    assert count == 0, "held-out statements withheld" if name in HELD_OUT else uncovered

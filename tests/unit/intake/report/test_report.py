"""Dry-run report: what an import would write, computed in a scratch term namespace.

``sema.intake.report.report`` is imported lazily, so a missing module fails each test on its own
instead of failing collection.

The contract the tests pin (dry-run-report spec, design Decision 13):

- ``report.dry_run(spec, provider, staging, ledger) -> DryRun``, whose ``report`` is the
  ``DryRunReport``: ``ledger`` is the run's ``RunLedger``; the run id is its directory
  name. Term staging goes to
  ``f"{spec.external_namespace}__dryrun_{run_id[:8]}"``, dropped in ``finally``. The report
  is written to the ledger as one artifact.
- ``report.REPORT_SEED``: the int seed of the rendered sample.
- ``report.DryRunReport``: pydantic, ``extra="forbid"``, carrying no path or timestamp, with
  at least ``vocabulary_count``, ``vocabularies`` (name -> per-vocabulary report with
  ``term_count``, ``seeded``, ``longest``, ``near_duplicates``, ``null_required``; rendered
  terms carry ``code`` and ``display``), ``root_count``, ``max_depth``, ``lifecycle``
  (``active``, ``retired``, ``replaced``: retired terms whose replacement is an active
  term), ``entities`` (qualified name -> ``properties``, ``foreign_keys``, ``domains``,
  ``inline_value_sets`` counts from the Table Schemas) and ``without_obligation``
  (qualified name -> reason).
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from pydantic import ValidationError

from sema.binding.spec import ConceptSourceSpec, load_concept_source_spec
from sema.ingest.duckdb_staging import Staging
from sema.intake.land.landing import LandResult
from sema.intake.calls.ledger import Ledger
from sema.intake.land.loaders.semsql import land_semsql
from sema.intake.land.loaders.tabular import land_tabular
from sema.intake.land.relations import RelationProvider
from tests.unit.intake.profile_env import A3_DEPTH
from tests.unit.intake.structure_env import A2_SPEC, A2_WORKBOOK, A3_DB, ACRIS, ACRIS_SPEC
from tests.unit.intake.verify.verify_env import edited_spec

pytestmark = pytest.mark.unit

A3_SPEC = A3_DB.parent / "ontology.spec.yaml"
A2_REPORT = A2_SPEC.parent / "codes_a2.report.json"
A3_REPORT = A3_DB.parent / "ontology.report.json"
ACRIS_TERMS = {
    "Country_Codes": 250,
    "Document_Control_Codes": 126,
    "Property_Types_Codes": 49,
    "State_Codes": 54,
    "UCC_Collateral_Codes": 8,
}
ACRIS_ENTITIES = 15
ACRIS_PROPERTIES = 125
ACRIS_FOREIGN_KEYS = 8
ACRIS_DOMAINS = 13
ACRIS_INLINE_VALUE_SETS = 6
A3_ACTIVE, A3_RETIRED = 48, 2
SEEDED, LONGEST = 10, 5
SCRATCH_MARK = "__dryrun_"
SENTINEL = "sentinel-row"
Lander = Callable[..., LandResult]


def report() -> ModuleType:
    return importlib.import_module("sema.intake.report.report")


@dataclass
class Landed:
    staging: Staging
    provider: RelationProvider
    land: LandResult
    root: Path

    def dry_run(self, spec: ConceptSourceSpec, run_id: str) -> Any:
        ledger = Ledger(self.root / "ledger").run(run_id)
        return report().dry_run(spec, self.provider, self.staging, ledger).report


def _land(root: Path, lander: Lander, path: Path, schema: str) -> Landed:
    staging = Staging(str(root / "staging.duckdb"), schemas=())
    landed = lander(path, staging, schema=schema)
    return Landed(staging, RelationProvider(warehouse=None, staging=staging), landed, root)


@pytest.fixture(scope="module")
def a2(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Landed]:
    landed = _land(tmp_path_factory.mktemp("report_a2"), land_tabular, A2_WORKBOOK, "land_a2")
    yield landed
    landed.staging.close()


@pytest.fixture(scope="module")
def a3(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Landed]:
    landed = _land(tmp_path_factory.mktemp("report_a3"), land_semsql, A3_DB, "land_a3")
    yield landed
    landed.staging.close()


@pytest.fixture(scope="module")
def acris(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Landed]:
    landed = _land(tmp_path_factory.mktemp("report_acris"), land_tabular, ACRIS, "land_acris")
    yield landed
    landed.staging.close()


class _Once:
    """Runs the ACRIS dry run on first use, so a missing module fails each test, not setup."""

    def __init__(self, landed: Landed) -> None:
        self.landed = landed
        self.found: Any = None

    def get(self) -> Any:
        if self.found is None:
            spec = load_concept_source_spec(ACRIS_SPEC)
            self.found = self.landed.dry_run(spec, "acris-report")
        return self.found


@pytest.fixture(scope="module")
def acris_once(acris: Landed) -> _Once:
    return _Once(acris)


@pytest.fixture
def fresh_a2(tmp_path: Path) -> Iterator[Landed]:
    landed = _land(tmp_path, land_tabular, A2_WORKBOOK, "land_a2")
    yield landed
    landed.staging.close()


def _staged_a2(directory: Path) -> ConceptSourceSpec:
    def edit(section: dict[str, Any]) -> None:
        section["term_storage"] = {"inline_cap": 0}

    return edited_spec(A2_SPEC, directory, edit)


def _seed_real_namespace(staging: Staging, namespace: str) -> list[tuple[Any, ...]]:
    staging.execute(f'CREATE SCHEMA "{namespace}"')
    staging.execute(f'CREATE TABLE "{namespace}".concept AS SELECT \'{SENTINEL}\' AS concept_id')
    return _namespace_rows(staging, namespace)


def _namespace_rows(staging: Staging, namespace: str) -> list[tuple[Any, ...]]:
    tables = staging.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = ?"
        " ORDER BY table_name",
        [namespace],
    ).fetchall()
    rows = [("table", name) for (name,) in tables]
    if ("table", "concept") in rows:
        rows += staging.execute(f'SELECT concept_id FROM "{namespace}".concept').fetchall()
    return rows


def _spy_term_staging(monkeypatch: pytest.MonkeyPatch, *, fail: bool) -> list[str]:
    from sema.binding import importer

    written: list[str] = []
    real = importer.write_term_staging

    def spy(staging: Staging, namespace: str, *tables: Any) -> None:
        written.append(namespace)
        real(staging, namespace, *tables)
        if fail:
            raise RuntimeError("forced failure after term staging")

    monkeypatch.setattr(importer, "write_term_staging", spy)
    return written


def _plain(model: Any) -> Any:
    return json.loads(model.model_dump_json())


def _term_counts(found: Any) -> dict[str, int]:
    return {name: vocabulary.term_count for name, vocabulary in found.vocabularies.items()}


def _reports_in(run_dir: Path) -> list[Any]:
    found = []
    for path in sorted((run_dir / "artifacts").glob("*.json")):
        try:
            found.append(report().DryRunReport.model_validate_json(path.read_text()))
        except ValidationError:
            continue
    return found


# Import into a scratch namespace


def test_staged_dry_run_leaves_the_real_namespace_and_drops_the_scratch_one(
    fresh_a2: Landed, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    spec = _staged_a2(tmp_path / "spec")
    before = _seed_real_namespace(fresh_a2.staging, spec.external_namespace)
    schemas = fresh_a2.staging.list_all_schemas()
    written = _spy_term_staging(monkeypatch, fail=False)

    found = fresh_a2.dry_run(spec, "staged-run")

    (scratch,) = written
    assert scratch.startswith(spec.external_namespace + SCRATCH_MARK)
    assert scratch not in fresh_a2.staging.list_all_schemas()
    assert fresh_a2.staging.list_all_schemas() == schemas
    assert _namespace_rows(fresh_a2.staging, spec.external_namespace) == before
    assert sum(_term_counts(found).values()) > 0


def test_scratch_namespace_is_dropped_on_a_forced_failure(
    fresh_a2: Landed, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    spec = _staged_a2(tmp_path / "spec")
    before = _seed_real_namespace(fresh_a2.staging, spec.external_namespace)
    schemas = fresh_a2.staging.list_all_schemas()
    written = _spy_term_staging(monkeypatch, fail=True)

    with pytest.raises(RuntimeError, match="forced failure"):
        fresh_a2.dry_run(spec, "failed-run")

    (scratch,) = written
    assert scratch.startswith(spec.external_namespace + SCRATCH_MARK)
    assert scratch not in fresh_a2.staging.list_all_schemas()
    assert fresh_a2.staging.list_all_schemas() == schemas
    assert _namespace_rows(fresh_a2.staging, spec.external_namespace) == before


def test_staged_and_inline_dry_runs_count_the_same_terms(
    fresh_a2: Landed, tmp_path: Path
) -> None:
    inline = fresh_a2.dry_run(load_concept_source_spec(A2_SPEC), "inline-run")
    staged = fresh_a2.dry_run(_staged_a2(tmp_path / "spec"), "staged-run")

    assert _term_counts(staged) == _term_counts(inline)


# Report counts


def test_a2_report_equals_golden(a2: Landed) -> None:
    found = a2.dry_run(load_concept_source_spec(A2_SPEC), "a2-golden")

    assert _plain(found) == json.loads(A2_REPORT.read_text())
    assert report().DryRunReport.model_validate_json(A2_REPORT.read_text()) == found


def test_a3_report_equals_golden_with_retired_terms_and_one_root(a3: Landed) -> None:
    found = a3.dry_run(load_concept_source_spec(A3_SPEC), "a3-golden")

    assert _plain(found) == json.loads(A3_REPORT.read_text())
    lifecycle = found.lifecycle
    assert (lifecycle.active, lifecycle.retired, lifecycle.replaced) == (
        A3_ACTIVE,
        A3_RETIRED,
        A3_RETIRED,
    )
    assert (found.root_count, found.max_depth) == (1, A3_DEPTH)


def test_acris_report_counts_entities_and_vocabularies(acris_once: _Once) -> None:
    acris_report = acris_once.get()
    assert acris_report.vocabulary_count == len(ACRIS_TERMS)
    assert _term_counts(acris_report) == ACRIS_TERMS
    entities = acris_report.entities
    assert len(entities) == ACRIS_ENTITIES
    assert sum(e.properties for e in entities.values()) == ACRIS_PROPERTIES


def test_acris_report_counts_table_schema_constraints(acris_once: _Once) -> None:
    entities = acris_once.get().entities.values()

    assert sum(e.foreign_keys for e in entities) == ACRIS_FOREIGN_KEYS
    assert sum(e.domains for e in entities) == ACRIS_DOMAINS
    assert sum(e.inline_value_sets for e in entities) == ACRIS_INLINE_VALUE_SETS


def test_acris_entities_have_no_obligation_for_want_of_a_primary_key(
    acris_once: _Once,
) -> None:
    acris_report = acris_once.get()
    reasons = acris_report.without_obligation

    assert set(reasons) == set(acris_report.entities)
    assert len(reasons) == ACRIS_ENTITIES
    assert all("primary key" in reason.replace("_", " ").lower() for reason in reasons.values())


# Rendered terms


def test_seeded_sample_is_stable_across_two_runs(acris: Landed, acris_once: _Once) -> None:
    acris_report = acris_once.get()
    again = acris.dry_run(load_concept_source_spec(ACRIS_SPEC), "acris-again")

    first = {name: v.seeded for name, v in acris_report.vocabularies.items()}
    second = {name: v.seeded for name, v in again.vocabularies.items()}
    assert first == second
    assert {name: len(seeded) for name, seeded in first.items()} == {
        name: min(SEEDED, count) for name, count in ACRIS_TERMS.items()
    }
    assert isinstance(report().REPORT_SEED, int)


def test_rendered_terms_list_the_longest_labels(acris_once: _Once) -> None:
    for name, vocabulary in acris_once.get().vocabularies.items():
        lengths = [len(term.display or "") for term in vocabulary.longest]
        assert len(lengths) == min(LONGEST, ACRIS_TERMS[name])
        assert lengths == sorted(lengths, reverse=True)
        assert isinstance(vocabulary.near_duplicates, list)
        assert isinstance(vocabulary.null_required, list)


# Report is a typed artifact


def test_report_round_trips_through_the_ledger(tmp_path: Path) -> None:
    landed = _land(tmp_path, land_tabular, A2_WORKBOOK, "land_a2")
    found = landed.dry_run(load_concept_source_spec(A2_SPEC), "round-trip")
    landed.staging.close()

    assert _reports_in(tmp_path / "ledger" / "round-trip") == [found]


def test_report_forbids_extra_fields(a2: Landed) -> None:
    found = a2.dry_run(load_concept_source_spec(A2_SPEC), "extra-field")

    with pytest.raises(ValidationError):
        report().DryRunReport.model_validate({**_plain(found), "unexpected": 1})


# Run ledger: artifacts after a dry run


def test_artifacts_after_a_dry_run(tmp_path: Path) -> None:
    from tests.unit.intake.report.report_ledger_env import a2_stages_into

    run_dir = tmp_path / "ledger" / "a2-stages"
    landed = _land(tmp_path, land_tabular, A2_WORKBOOK, "land_a2")
    written = a2_stages_into(landed.provider, landed.land, run_dir)
    found = report().dry_run(
        written.spec, landed.provider, landed.staging, written.ledger
    ).report
    written.verify_dry_run()
    landed.staging.close()

    written.assert_stage_artifacts_parse()
    assert _reports_in(run_dir) == [found]

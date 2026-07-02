"""US-012A: DuckDB end-to-end smoke — the whole spine wired into ``run_fit``.

Hermetic: a fake VocabStore drives the resolver, the source table + value-mapping
store + staging table all live in one in-process DuckDB (local, no network), and
the gold set is built in memory. The test exercises the full chain
resolve -> produce -> assemble -> compile -> staging-write -> QA -> eval and
asserts the shapes are MappingAssertion -> FieldMap -> MappingPlan (no
MappingProposal), a staging table is written, Gate D-lite passes, and the eval
report is produced.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import pytest

from sema.eval.mapping_goldset import GoldSet
from sema.eval.mapping_goldset_utils import GoldLabel, GoldRow
from sema.eval.staging_qa_utils import QAOutcome
from sema.models.planner.mapping_plan import MappingAssertion, MappingPlan
from sema.models.planner.patterns import MappingPattern
from sema.pipeline.fit_slice0 import FitResult, run_fit
from sema.pipeline.fit_slice0_utils import build_slice0_fit_request
from sema.resolve.engine import VocabularyResolver
from sema.resolve.vocab_store_utils import ConceptRow

pytestmark = pytest.mark.unit

_MANIFEST = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "sema"
    / "targets"
    / "manifests"
    / "omop_condition_slice0.yaml"
)
_STANDARD_ID = "45768916"


class _FakeVocabStore:
    """Returns configured ConceptRows; enough surface for the resolver."""

    def __init__(self) -> None:
        self._by_code = {
            ("OncoTree", "LUAD"): ConceptRow(
                id="777926",
                name="Lung Adenocarcinoma",
                domain="Condition",
                vocabulary="OncoTree",
                standard=None,
                code="LUAD",
                invalid_reason=None,
            )
        }
        self._maps_to = {
            "777926": [
                ConceptRow(
                    id=_STANDARD_ID,
                    name="Adenocarcinoma of lung",
                    domain="Condition",
                    vocabulary="SNOMED",
                    standard="S",
                    code="254626006",
                    invalid_reason=None,
                )
            ]
        }

        self._by_id = {
            row.id: row for rows in self._maps_to.values() for row in rows
        }

    def concept_by_code(self, vocabulary: str, code: str) -> ConceptRow | None:
        return self._by_code.get((vocabulary, code))

    def maps_to_targets(
        self,
        concept_id: str,
        *,
        relationship_id: str,
        standard_flag: str | None = None,
        only_valid: bool = False,
    ) -> list[ConceptRow]:
        return list(self._maps_to.get(concept_id, []))

    def concepts_by_ids(self, ids: list[str]) -> dict[str, ConceptRow | None]:
        return {i: self._by_id.get(i) for i in ids}


def _seed_source(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute('CREATE SCHEMA IF NOT EXISTS "study"')
    conn.execute('CREATE TABLE "study"."sample" (ONCOTREE_CODE VARCHAR)')
    conn.executemany(
        'INSERT INTO "study"."sample" VALUES (?)',
        [("LUAD",), ("LUAD",), ("ZZZZ",)],
    )


def _gold() -> GoldSet:
    return GoldSet(
        rows=[
            GoldRow("LUAD", int(_STANDARD_ID), GoldLabel.RESOLVED, 2),
            GoldRow("ZZZZ", None, GoldLabel.NO_MAP, 1),
        ]
    )


def _run(conn: duckdb.DuckDBPyConnection) -> FitResult:
    policy, request = build_slice0_fit_request(
        manifest_path=_MANIFEST,
        source_schema="study",
        source_table="sample",
        value_column="ONCOTREE_CODE",
        source_codes=["LUAD", "ZZZZ"],
        source_row_count=3,
        gold=_gold(),
    )
    resolver = VocabularyResolver(_FakeVocabStore(), policy)
    return run_fit(resolver, request, value_mapping_conn=conn, staging_conn=conn)


class TestRunFit:
    def test_returns_assertion_field_map_plan_shapes(
        self, tmp_path: Path
    ) -> None:
        conn = duckdb.connect(str(tmp_path / "fit.duckdb"))
        _seed_source(conn)
        result = _run(conn)
        assert isinstance(result.assertion, MappingAssertion)
        assert result.assertion.pattern is MappingPattern.VOCAB_LOOKUP
        assert isinstance(result.plan, MappingPlan)
        # the plan is built from FieldMaps projected off the assertions: the
        # VOCAB_LOOKUP target plus the two run-constant staging columns (§1.5(e)).
        patterns = {fm.pattern for fm in result.plan.field_maps}
        assert patterns == {MappingPattern.VOCAB_LOOKUP, MappingPattern.CONSTANT}

    def test_plan_covers_the_three_field_staging_obligation(
        self, tmp_path: Path
    ) -> None:
        from sema.models.planner.lifecycle import PlanVerdict

        conn = duckdb.connect(str(tmp_path / "fit.duckdb"))
        _seed_source(conn)
        result = _run(conn)
        # the live path now enforces the real §1.5(e) 3-field staging obligation
        assert len(result.plan.obligation.required_fields) == 3
        assert set(result.plan.obligation.required_fields).issubset(
            result.plan.covered_required_fields()
        )
        assert result.plan.derive_verdict() is PlanVerdict.compilable

    def test_writes_staging_table_with_row_count(self, tmp_path: Path) -> None:
        conn = duckdb.connect(str(tmp_path / "fit.duckdb"))
        _seed_source(conn)
        result = _run(conn)
        assert result.rows_staged == 3
        total = conn.execute(
            f'SELECT COUNT(*) FROM "{result.staging_schema}".'
            f'"{result.staging_table}"'
        ).fetchone()[0]
        assert total == 3

    def test_no_map_rows_are_null_resolved_are_populated(
        self, tmp_path: Path
    ) -> None:
        conn = duckdb.connect(str(tmp_path / "fit.duckdb"))
        _seed_source(conn)
        result = _run(conn)
        null_codes = conn.execute(
            "SELECT source_oncotree_code FROM "
            f'"{result.staging_schema}"."{result.staging_table}" '
            "WHERE condition_concept_id IS NULL"
        ).fetchall()
        assert {r[0] for r in null_codes} == {"ZZZZ"}

    def test_gate_d_lite_passes(self, tmp_path: Path) -> None:
        conn = duckdb.connect(str(tmp_path / "fit.duckdb"))
        _seed_source(conn)
        result = _run(conn)
        assert result.qa.outcome is QAOutcome.PASS, result.qa.as_dict()

    def test_eval_report_is_produced_at_full_coverage(
        self, tmp_path: Path
    ) -> None:
        conn = duckdb.connect(str(tmp_path / "fit.duckdb"))
        _seed_source(conn)
        result = _run(conn)
        assert result.report.coverage_fraction == 1.0
        # precision is structural (deterministic exact-code walk)
        assert result.report.score.distinct_code.mapped_precision == 1.0

    def test_value_mapping_store_matches_frozen_a_schema(
        self, tmp_path: Path
    ) -> None:
        from sema.resolve.value_mapping_store_utils import FROZEN_COLUMNS

        conn = duckdb.connect(str(tmp_path / "fit.duckdb"))
        _seed_source(conn)
        result = _run(conn)
        assert tuple(result.store_columns) == tuple(FROZEN_COLUMNS)

    def test_rerun_is_idempotent(self, tmp_path: Path) -> None:
        conn = duckdb.connect(str(tmp_path / "fit.duckdb"))
        _seed_source(conn)
        first = _run(conn)
        order = "ORDER BY source_oncotree_code, source_table"
        before = conn.execute(
            f'SELECT * FROM "{first.staging_schema}"."{first.staging_table}" '
            + order
        ).fetchall()
        _run(conn)
        after = conn.execute(
            f'SELECT * FROM "{first.staging_schema}"."{first.staging_table}" '
            + order
        ).fetchall()
        assert before == after

    def test_no_mapping_proposal_in_chain(self, tmp_path: Path) -> None:
        conn = duckdb.connect(str(tmp_path / "fit.duckdb"))
        _seed_source(conn)
        result = _run(conn)
        # the spine is MappingAssertion -> FieldMap -> MappingPlan only
        assert type(result.assertion).__name__ == "MappingAssertion"
        assert "Proposal" not in type(result.assertion).__name__
        for fm in result.plan.field_maps:
            assert "Proposal" not in type(fm).__name__


def _run_codes(
    conn: duckdb.DuckDBPyConnection, codes: list[str], gold: GoldSet
) -> FitResult:
    policy, request = build_slice0_fit_request(
        manifest_path=_MANIFEST,
        source_schema="study",
        source_table="sample",
        value_column="ONCOTREE_CODE",
        source_codes=codes,
        source_row_count=3,
        gold=gold,
    )
    resolver = VocabularyResolver(_FakeVocabStore(), policy)
    return run_fit(resolver, request, value_mapping_conn=conn, staging_conn=conn)


def test_strict_report_ignores_stale_store_rows_absent_from_this_run(
    tmp_path: Path,
) -> None:
    # bug-369 F1 follow-up: the strict contradiction check must be scoped to
    # THIS run's mappings, not the whole historical store. A prior run left a
    # stale NO_MAP row for OTHER; OTHER is no longer in the source table, yet a
    # human labelled it RESOLVED. The current run (LUAD, ZZZZ) never touches
    # OTHER, so it must not fail on that stale disagreement.
    conn = duckdb.connect(str(tmp_path / "fit.duckdb"))
    _seed_source(conn)
    gold = GoldSet(
        rows=[
            GoldRow("LUAD", int(_STANDARD_ID), GoldLabel.RESOLVED, 2),
            GoldRow("ZZZZ", None, GoldLabel.NO_MAP, 1),
            GoldRow("OTHER", 999999, GoldLabel.RESOLVED, 1),
        ]
    )
    # Run 1 leaves a stale OTHER -> NO_MAP row in the shared store.
    _run_codes(conn, ["LUAD", "ZZZZ", "OTHER"], gold)
    # Run 2: OTHER has dropped out of the source; only LUAD, ZZZZ resolve now.
    result = _run_codes(conn, ["LUAD", "ZZZZ"], gold)
    assert result.report.has_labelled_contradiction() is False


def test_build_request_reads_binding_from_manifest() -> None:
    policy, request = build_slice0_fit_request(
        manifest_path=_MANIFEST,
        source_schema="study",
        source_table="sample",
        value_column="ONCOTREE_CODE",
        source_codes=["LUAD"],
        source_row_count=1,
        gold=_gold(),
    )
    # SOURCE vocabulary comes from the resolver policy (R9); the binding's
    # TARGET vocabulary is SNOMED.
    assert policy.source_vocabulary == "OncoTree"
    assert policy.target_domain == "Condition"
    assert request.resolve_context.target_field == "condition_concept_id"


def _unused(_: Any) -> None:  # pragma: no cover - keep imports tidy
    pass

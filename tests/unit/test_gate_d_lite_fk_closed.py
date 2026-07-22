"""S1-07 — Gate-D-lite extension over the FK-closed OMOP shape.

Runs the three new checks (FK closure, required-field non-null, missing-key
accounting) over a real materialized person + condition table, and proves each
FAILs on the defect it guards.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from sema.compile.compiler_utils import StagingDecision
from sema.compile.fk_closed_compiler import FkClosedCompiler
from sema.compile.fk_closed_compiler_utils import (
    ChildSourceSpec,
    ChildTableSpec,
    ParentTableSpec,
    RegistryJoinSpec,
)
from sema.eval.staging_qa import count_column_nulls, run_fk_closed_qa
from sema.eval.staging_qa_utils import (
    QAOutcome,
    check_fk_closure,
    check_missing_key_disposition,
    check_required_not_null,
)
from sema.resolve.identity_registry import (
    DEFAULT_SCHEMA,
    DEFAULT_TABLE,
    IdentityRegistry,
)

pytestmark = pytest.mark.unit

_REGISTRY = RegistryJoinSpec(
    schema=DEFAULT_SCHEMA,
    table=DEFAULT_TABLE,
    namespace_column="source_namespace",
    key_column="source_entity_key",
    id_column="entity_id",
)
_PERSON = ParentTableSpec(schema="omop", table="person", id_column="person_id")
_CONDITION = ChildTableSpec(
    schema="omop",
    table="condition_occurrence",
    pk_column="condition_occurrence_id",
    fk_column="person_id",
    value_column="condition_concept_id",
    null_columns=("condition_start_date",),
    row_ref_column="source_row_ref",
    patient_key_column="source_patient_key",
    scope_schema_column="source_schema",
    scope_table_column="source_table",
)
_REQUIRED = ("condition_occurrence_id", "person_id", "condition_concept_id")
_SOURCE = ChildSourceSpec(
    schema="cbio_msk",
    table="sample",
    value_column="oncotree_code",
    row_ref_column="sample_id",
    patient_key_column="patient_id",
)
_DECISIONS = [StagingDecision("LUAD", 777926, "RESOLVED", None, "auto_accepted")]


@pytest.fixture()
def materialized(tmp_path: Path) -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(str(tmp_path / "poc.duckdb"))
    conn.execute('CREATE SCHEMA "cbio_msk"')
    conn.execute(
        'CREATE TABLE "cbio_msk"."sample" '
        "(sample_id VARCHAR, patient_id VARCHAR, oncotree_code VARCHAR)"
    )
    # one clean row + one blank-key row (routes to review).
    conn.execute(
        "INSERT INTO \"cbio_msk\".\"sample\" VALUES "
        "('S-1', 'P-1', 'LUAD'), ('S-2', '', 'LUAD')"
    )
    IdentityRegistry(conn).get_or_create([("cbio_msk", "P-1")], run_id="s1-07")
    FkClosedCompiler(no_map_default=0).materialize(
        conn,
        parent=_PERSON,
        child=_CONDITION,
        source=_SOURCE,
        registry=_REGISTRY,
        decisions=_DECISIONS,
    )
    return conn


def _qa(conn: duckdb.DuckDBPyConnection) -> object:
    return run_fk_closed_qa(
        conn,
        parent=_PERSON,
        child=_CONDITION,
        source=_SOURCE,
        required_fields=_REQUIRED,
        source_row_count=2,
    )


class TestPassingReport:
    def test_all_checks_pass_on_clean_shape(
        self, materialized: duckdb.DuckDBPyConnection
    ) -> None:
        report = _qa(materialized)
        assert report.passed
        names = {c.name for c in report.checks}
        assert names == {"fk_closure", "required_not_null", "missing_key_disposition"}

    def test_missing_key_accounting_balances(
        self, materialized: duckdb.DuckDBPyConnection
    ) -> None:
        report = _qa(materialized)
        disp = next(c for c in report.checks if c.name == "missing_key_disposition")
        assert disp.details == {
            "written": 1,
            "missing_key": 1,
            "source": 2,
            "accounted": 2,
        }

    def test_nullable_date_not_gated(
        self, materialized: duckdb.DuckDBPyConnection
    ) -> None:
        # condition_start_date is 100% NULL (D4) but is NOT in required_fields,
        # so required_not_null passes; the null count is still observable.
        nulls = count_column_nulls(
            materialized, "omop", "condition_occurrence", ("condition_start_date",)
        )
        assert nulls == {"condition_start_date": 1}
        report = _qa(materialized)
        assert report.passed


class TestFailingChecks:
    def test_fk_closure_fails_on_orphan(
        self, materialized: duckdb.DuckDBPyConnection
    ) -> None:
        materialized.execute(
            "INSERT INTO omop.condition_occurrence VALUES "
            "(999, 424242, 1, NULL, 'S-X', 'P-X', 'cbio_msk', 'sample')"
        )
        report = _qa(materialized)
        assert not report.passed
        fk = next(c for c in report.checks if c.name == "fk_closure")
        assert fk.outcome is QAOutcome.FAIL
        assert fk.details["orphan_rows"] == 1

    def test_required_not_null_fails_on_null_required_field(self) -> None:
        check = check_required_not_null({"person_id": 3, "condition_occurrence_id": 0})
        assert check.outcome is QAOutcome.FAIL
        assert check.details["offenders"] == {"person_id": 3}

    def test_fk_closure_pure_check_passes_on_zero(self) -> None:
        assert check_fk_closure(0).outcome is QAOutcome.PASS

    def test_missing_key_disposition_fails_when_unbalanced(self) -> None:
        check = check_missing_key_disposition(
            written_rows=5, missing_key_rows=1, source_rows=10
        )
        assert check.outcome is QAOutcome.FAIL

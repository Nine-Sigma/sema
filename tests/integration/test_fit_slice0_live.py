"""US-012A live test: the whole spine end-to-end on ~/.sema/poc.duckdb.

Skip-guarded on the DuckDB mirror. Runs resolve -> produce -> assemble ->
compile -> staging-write -> Gate D-lite QA -> eval for a real cBioPortal study
against DuckDB and asserts: a staging table is written with row count == source
row count, Gate D-lite passes, the eval report is produced, the value-mapping
store matches the §1.5(a) frozen schema, and the staging table matches §1.5(b).
No Databricks objects are written (the source study is copied into a temp work
DuckDB; US-013 is the live Databricks gate).
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from sema.eval.mapping_goldset import GoldSet, load_gold_set
from sema.eval.staging_qa_utils import QAOutcome
from sema.models.planner.mapping_plan import MappingAssertion, MappingPlan
from sema.models.planner.patterns import MappingPattern
from showcase.cbioportal_to_omop.slice0_fit import run_fit
from showcase.cbioportal_to_omop.slice0_fit_utils import build_slice0_fit_request, discover_study
from sema.resolve.engine import VocabularyResolver
from showcase.cbioportal_to_omop.omop_policy import OMOP_VOCAB_SCHEMA
from sema.resolve.value_mapping_store_utils import FROZEN_COLUMNS
from sema.resolve.vocab_store import open_duckdb_vocab_store, VocabStore

pytestmark = pytest.mark.integration

_DB = Path.home() / ".sema" / "poc.duckdb"
_GOLD = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "gold"
    / "oncotree_condition_slice0.jsonl"
)
_MANIFEST = (
    Path(__file__).resolve().parents[2]
    / "showcase" / "cbioportal_to_omop" / "manifests"
    / "omop_condition_slice0.yaml"
)
_VALUE_COLUMN = "ONCOTREE_CODE"


def _copy_source(
    work: duckdb.DuckDBPyConnection, schema: str, table: str
) -> None:
    work.execute(f"ATTACH '{_DB}' AS poc (READ_ONLY)")
    work.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
    work.execute(
        f'CREATE TABLE "{schema}"."{table}" AS '
        f'SELECT "{_VALUE_COLUMN}" FROM poc."{schema}"."{table}" '
        f'WHERE "{_VALUE_COLUMN}" IS NOT NULL'
    )


@pytest.mark.skipif(not _DB.exists(), reason="~/.sema/poc.duckdb not present")
def test_fit_chain_end_to_end_on_duckdb(tmp_path: Path) -> None:
    pdb = duckdb.connect(str(_DB), read_only=True)
    found = discover_study(pdb, value_column=_VALUE_COLUMN)
    assert found is not None, "no cbioportal sample table with ONCOTREE_CODE"
    src_schema, src_table = found
    pdb.close()

    # Source lives in a temp work DuckDB so poc.duckdb is never mutated.
    work = duckdb.connect(str(tmp_path / "fit.duckdb"))
    _copy_source(work, src_schema, src_table)

    from showcase.cbioportal_to_omop.slice0_fit_utils import enumerate_source

    codes, row_count = enumerate_source(
        work, schema=src_schema, table=src_table, value_column=_VALUE_COLUMN
    )
    gold = GoldSet(rows=load_gold_set(_GOLD)) if _GOLD.exists() else GoldSet(rows=[])

    policy, request = build_slice0_fit_request(
        manifest_path=_MANIFEST,
        source_schema=src_schema,
        source_table=src_table,
        value_column=_VALUE_COLUMN,
        source_codes=codes,
        source_row_count=row_count,
        gold=gold,
    )
    # Vocabulary stays in poc.duckdb (read-only); store + staging in the work db.
    vstore = open_duckdb_vocab_store(str(_DB), schema=OMOP_VOCAB_SCHEMA)
    resolver = VocabularyResolver(vstore, policy)
    result = run_fit(
        resolver, request, value_mapping_conn=work, staging_conn=work
    )

    # Shapes: MappingAssertion -> FieldMap -> MappingPlan (no MappingProposal).
    assert isinstance(result.assertion, MappingAssertion)
    assert result.assertion.pattern is MappingPattern.VOCAB_LOOKUP
    assert isinstance(result.plan, MappingPlan)

    # A staging table is written with row count == source row count.
    assert result.rows_staged == row_count
    staged = work.execute(
        f'SELECT COUNT(*) FROM "{result.staging_schema}".'
        f'"{result.staging_table}"'
    ).fetchone()[0]
    assert staged == row_count

    # Gate D-lite passes and the eval report is produced.
    assert result.qa.outcome is QAOutcome.PASS, result.qa.as_dict()
    assert result.report is not None

    # §1.5(a) store schema and §1.5(b) staging projection.
    assert tuple(result.store_columns) == tuple(FROZEN_COLUMNS)
    null_targets = work.execute(
        "SELECT COUNT(*) FROM "
        f'"{result.staging_schema}"."{result.staging_table}" '
        "WHERE condition_concept_id IS NULL "
        "AND resolution_status != 'NO_MAP'"
    ).fetchone()[0]
    assert null_targets == 0
    vstore.close()

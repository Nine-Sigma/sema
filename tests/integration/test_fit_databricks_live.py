"""US-013 LIVE: the whole Slice-0 chain on real Databricks (skip-guarded).

Run by a human with workspace credentials — never wired into CI. Reuses the
US-012A ``sema fit`` CLI surface with ``--backend databricks``: it resolves the
distinct ONCOTREE_CODE for a real study against ``workspace.vocabulary_omop``,
writes the §1.5(b) staging table to ``workspace`` via an atomic Delta
``REPLACE WHERE`` (scoped on source_schema/source_table), and runs Gate D-lite +
the eval report. The value-mapping store stays on a local DuckDB (US-005).

Acceptance of the metrics requires human-labelled 100% gold coverage
(US-002/US-012); this test asserts the pipeline *runs and is idempotent*, not
the acceptance verdict.

Required env (DatabricksConfig, env_prefix ``DATABRICKS_``):
    DATABRICKS_HOST, DATABRICKS_HTTP_PATH, DATABRICKS_TOKEN
Optional:
    DATABRICKS_FIT_SCHEMA (source study schema, default ``msk_chord_2024``)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from click.testing import CliRunner

from sema.cli import cli
from sema.cli_fit_utils import open_databricks_cursor
from sema.models.config import DatabricksConfig

pytestmark = pytest.mark.integration

_MANIFEST = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "sema"
    / "targets"
    / "manifests"
    / "omop_condition_slice0.yaml"
)
_STUDY = os.environ.get("DATABRICKS_FIT_SCHEMA", "msk_chord_2024")
_CONFIGURED = bool(
    os.environ.get("DATABRICKS_HOST")
    and os.environ.get("DATABRICKS_HTTP_PATH")
    and os.environ.get("DATABRICKS_TOKEN")
)

skip_live = pytest.mark.skipif(
    not _CONFIGURED, reason="Databricks workspace credentials not configured"
)


def _is_not_found(text: str) -> bool:
    return "NOT_FOUND" in text or "cannot be found" in text


def _invoke(store_path: Path) -> dict[str, object]:
    result = CliRunner().invoke(
        cli,
        [
            "fit",
            "--manifest", str(_MANIFEST),
            "--backend", "databricks",
            "--catalog", "workspace",
            "--study-schema", _STUDY,
            "--duckdb", str(store_path),
        ],
    )
    if result.exit_code != 0 and _is_not_found(result.output):
        pytest.skip(
            f"source {_STUDY}.sample (or workspace.vocabulary_omop) not provisioned; "
            "set DATABRICKS_FIT_SCHEMA to a real cBioPortal study"
        )
    assert result.exit_code == 0, result.output
    return json.loads(result.output)  # type: ignore[no-any-return]


@skip_live
def test_databricks_fit_runs_and_is_idempotent(tmp_path: Path) -> None:
    first = _invoke(tmp_path / "store.duckdb")
    assert first["rows_staged"] == first["source_row_count"]
    assert first["gate_d_lite"]["outcome"] == "PASS"  # type: ignore[index]
    assert "eval" in first

    # Re-running the command reproduces the same staged row count (idempotent).
    second = _invoke(tmp_path / "store2.duckdb")
    assert second["rows_staged"] == first["rows_staged"]
    assert second["gate_d_lite"]["outcome"] == "PASS"  # type: ignore[index]


@skip_live
def test_databricks_staging_has_no_person_id_column() -> None:
    cursor = open_databricks_cursor(
        DatabricksConfig(), catalog="workspace", schema="sema_staging"
    )
    try:
        cursor.execute("DESCRIBE TABLE `sema_staging`.`condition_staging`")
        columns = {str(row[0]).lower() for row in cursor.fetchall()}
    except Exception as exc:  # noqa: BLE001 — staging not yet written in this workspace
        if _is_not_found(str(exc)):
            pytest.skip("staging table not yet written; run the fit test first")
        raise
    assert "person_id" not in columns

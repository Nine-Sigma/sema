"""The dry run imports through the run's staging, whatever the inline cap (task 3.6d).

Pinned contract: a settled spec whose vocabularies exceed ``term_storage.inline_cap``
reaches Review with a dry-run verdict and report instead of raising for a missing staging
database. The dry run stages only in a scratch namespace it drops, so the run's staging
afterwards holds neither a scratch namespace nor any term under the real namespace.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sema.binding.spec import ConceptSourceSpec
from sema.intake.graph_stages import STAGING_FILE
from tests.unit.intake.catalog_env import connector_over, warehouse_catalog, write_descriptor
from tests.unit.intake.graph_env import GraphRun, graph, routed_client, settings, staged_namespaces
from tests.unit.intake.warehouse_env import CATALOG, SCHEMA

pytestmark = pytest.mark.unit

EVERY_VOCABULARY_ABOVE = {"inline_cap": 0}
WAITING, PASS = "waiting_review", "pass"


def _discovered_above_the_cap(root: Path) -> GraphRun:
    """The zero-call catalog with every vocabulary above its inline cap."""
    con, metadata = warehouse_catalog(status=False)
    client = routed_client()
    descriptor = write_descriptor(root / "inputs", term_storage=EVERY_VOCABULARY_ABOVE)
    try:
        found = graph().discover(
            None, warehouse=connector_over(con, metadata), catalog=CATALOG, schema=SCHEMA,
            descriptor=descriptor, client=client, **settings(root),
        )
    finally:
        con.close()
    return GraphRun(root, found.run_id, client, found)


def test_a_spec_above_the_inline_cap_dry_runs_through_the_run_staging(tmp_path: Path) -> None:
    run = _discovered_above_the_cap(tmp_path)

    payload = run.outcome.payload
    assert run.outcome.node == "review"
    assert run.nodes()[-2:] == ["dry_run", "review"]
    assert [kind for kind, stage in run.steps() if stage == "dry_run"] == ["node", "verdict"]
    assert [verdict["status"] for verdict in run.verdicts("dry_run")] == [PASS]
    assert payload["dry_run"] is not None
    assert run.row().status == WAITING
    spec = ConceptSourceSpec.model_validate(payload["spec"])
    assert spec.term_storage is not None and spec.term_storage.inline_cap == 0
    named, terms = staged_namespaces(run.ledger_dir / STAGING_FILE, spec.external_namespace)
    assert [name for name in named if name != spec.external_namespace] == []
    assert terms == 0
    assert run.client.invoke.call_count == 0

"""CLI: `sema target import --spec --out --evidence-out` against the DuckDB fixture."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from sema.cli_target import target_group
from sema.targets.registry import _clear_for_tests
from tests.unit.binding.fixtures.concept_library import SPEC_PATH, build_library

pytestmark = pytest.mark.unit


class _DuckDBConnector:
    def __init__(self) -> None:
        self._con = build_library()

    def execute_query(self, query: str) -> list[tuple[Any, ...]]:
        return self._con.execute(query).fetchall()


@pytest.fixture(autouse=True)
def _isolate_registry() -> Iterator[None]:
    _clear_for_tests()
    yield
    _clear_for_tests()


def test_target_import_writes_manifest_and_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # raising=False and late imports: before the seam exists, the test must fail
    # on the missing command rather than on the patch or a missing model.
    monkeypatch.setattr(
        "sema.cli_target._import_connector", _DuckDBConnector, raising=False
    )
    out, evidence_out = tmp_path / "manifest.yaml", tmp_path / "authored_evidence.json"
    result = CliRunner().invoke(
        target_group,
        [
            "import",
            "--spec",
            str(SPEC_PATH),
            "--out",
            str(out),
            "--evidence-out",
            str(evidence_out),
        ],
    )

    assert result.exit_code == 0, result.output
    from sema.binding.models import AuthoredEvidence
    from sema.targets.adapters.manifest import ManifestTargetAdapter
    from sema.targets.normalizer import TargetModelNormalizer

    normalized = TargetModelNormalizer.normalize(ManifestTargetAdapter(out))
    concept_props = [
        p
        for e in normalized.entities
        for p in e.properties
        if p.property_kind.value == "CONCEPT"
    ]
    assert len(concept_props) == 4
    evidence = AuthoredEvidence.model_validate_json(evidence_out.read_text())
    assert len(evidence.column_bindings) == 2
    assert len(evidence.value_mappings) == 3

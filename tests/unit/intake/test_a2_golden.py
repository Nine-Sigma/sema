"""A2: land the three-sheet workbook and import it with the hand-written spec."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from sema.ingest.duckdb_staging import Staging
from sema.targets.adapters.manifest_parser import parse_manifest_raw

pytestmark = pytest.mark.unit

FIXTURES = Path(__file__).parents[2] / "fixtures" / "intake" / "tabular"
WORKBOOK = FIXTURES / "codes_a2.xlsx"
SPEC = FIXTURES / "codes_a2.spec.yaml"
GOLDEN = FIXTURES / "codes_a2.manifest.yaml"


def import_a2(tmp_path: Path) -> Any:
    from sema.binding.importer import import_concept_source
    from sema.binding.spec import load_concept_source_spec
    from sema.intake.land.loaders.tabular import land_tabular
    from sema.intake.land.relations import RelationProvider

    with Staging(str(tmp_path / "staging.duckdb"), schemas=()) as staging:
        land_tabular(WORKBOOK, staging, schema="land_a2")
        return import_concept_source(
            load_concept_source_spec(SPEC),
            RelationProvider(warehouse=None, staging=staging),
            dataset_map={},
            staging=staging,
        )


def test_a2_manifest_equals_golden(tmp_path: Path) -> None:
    result = import_a2(tmp_path)

    assert yaml.safe_dump(result.manifest, sort_keys=False) == GOLDEN.read_text()
    parsed = parse_manifest_raw(yaml.safe_load(GOLDEN.read_text()))
    assert len(parsed.terms) == 8


def test_a2_retired_code_is_in_the_sidecar(tmp_path: Path) -> None:
    result = import_a2(tmp_path)

    assert [(r.kind, r.vocabulary, r.code, r.replaced_by) for r in result.authored.retirements] == [
        ("term", "VS_COLOR", "TEAL", "BLUE")
    ]

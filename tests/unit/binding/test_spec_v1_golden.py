"""Version 1 spec compatibility: the fixture imports to the pre-version-2 manifest."""

from __future__ import annotations

from typing import Any

import pytest
import yaml

from sema.targets.adapters.manifest_parser import parse_manifest_raw
from tests.unit.binding.fixtures.concept_library import (
    FIXTURES_DIR,
    SPEC_PATH,
    build_library,
    load_dataset_map,
    provider_for,
)

pytestmark = pytest.mark.unit

GOLDEN_MANIFEST = FIXTURES_DIR / "golden" / "syn_library_v1.manifest.yaml"
GOLDEN_EVIDENCE = FIXTURES_DIR / "golden" / "syn_library_v1.evidence.json"


def _import() -> Any:
    from sema.binding.importer import import_concept_source
    from sema.binding.spec import load_concept_source_spec

    con = build_library()
    try:
        return import_concept_source(
            load_concept_source_spec(SPEC_PATH),
            provider_for(con),
            dataset_map=load_dataset_map(),
        )
    finally:
        con.close()


def test_version_1_manifest_text_equals_golden() -> None:
    dumped = yaml.safe_dump(_import().manifest, sort_keys=False)

    assert dumped == GOLDEN_MANIFEST.read_text()


def test_version_1_parsed_manifest_equals_golden() -> None:
    result = _import()

    parsed = parse_manifest_raw(yaml.safe_load(yaml.safe_dump(result.manifest)))
    golden = parse_manifest_raw(yaml.safe_load(GOLDEN_MANIFEST.read_text()))
    assert parsed == golden


def test_version_1_evidence_equals_golden() -> None:
    dumped = _import().authored.model_dump_json(indent=2)

    assert dumped == GOLDEN_EVIDENCE.read_text()

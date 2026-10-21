"""Dry-run verifier: the import's output checked for duplicates, parseability, and lifecycle."""

from __future__ import annotations

import copy
import dataclasses
from pathlib import Path
from typing import Any

import pytest

from sema.binding.importer import import_concept_source
from sema.binding.spec import load_concept_source_spec
from tests.unit.intake.structure_env import A2_SPEC
from tests.unit.intake.verify.conftest import Landed
from tests.unit.intake.verify.verify_env import (
    DRY_RUN_CHECKS,
    assert_named,
    assert_passes,
    failures,
    tools_for,
    verifier,
)

pytestmark = pytest.mark.unit

DUPLICATED = ("VS_COLOR", "BLUE")
RETIRED_CODE = "TEAL"


@pytest.fixture(scope="module")
def a2_import(a2: Landed) -> Any:
    spec = load_concept_source_spec(A2_SPEC)
    return import_concept_source(
        spec, a2.source.tools.provider, dataset_map={}, staging=a2.staging
    )


def _verify(imported: Any, a2: Landed, structure: Any, ledger: Path) -> Any:
    spec = load_concept_source_spec(A2_SPEC)
    tools = tools_for(a2.source, ledger, stage="dry_run", structure=structure)
    return verifier("dry_run").verify_dry_run(spec, imported, structure=structure, tools=tools)


def _with_manifest(imported: Any, edit: Any) -> Any:
    manifest = copy.deepcopy(imported.manifest)
    edit(manifest)
    return dataclasses.replace(imported, manifest=manifest)


def _term(manifest: dict[str, Any], vocabulary: str, code: str) -> dict[str, Any]:
    terms = manifest["terms"]
    matched: list[dict[str, Any]] = [
        t for t in terms if (t["vocabulary"], t["code"]) == (vocabulary, code)
    ]
    (found,) = matched
    return found


def test_checks_are_the_documented_list() -> None:
    assert set(verifier("dry_run").CHECKS) == DRY_RUN_CHECKS


def test_a2_dry_run_passes(
    a2: Landed, a2_structure: Any, a2_import: Any, tmp_path: Path
) -> None:
    verdict = _verify(a2_import, a2, a2_structure, tmp_path / "ledger" / "verify")

    assert_passes(verdict, DRY_RUN_CHECKS)


def test_two_active_terms_sharing_vocabulary_and_code_fail(
    a2: Landed, a2_structure: Any, a2_import: Any, tmp_path: Path
) -> None:
    def duplicate(manifest: dict[str, Any]) -> None:
        twin = dict(_term(manifest, *DUPLICATED), display="Blue again")
        manifest["terms"].append(twin)

    imported = _with_manifest(a2_import, duplicate)
    verdict = _verify(imported, a2, a2_structure, tmp_path / "ledger" / "verify")

    (found,) = failures(verdict, "unique_active_code")
    assert (found.evidence["vocabulary"], found.evidence["code"]) == DUPLICATED
    assert_named(verdict, DRY_RUN_CHECKS)


def test_manifest_the_adapter_rejects_fails(
    a2: Landed, a2_structure: Any, a2_import: Any, tmp_path: Path
) -> None:
    def unsupported(manifest: dict[str, Any]) -> None:
        manifest["manifest_version"] = 999

    imported = _with_manifest(a2_import, unsupported)
    verdict = _verify(imported, a2, a2_structure, tmp_path / "ledger" / "verify")

    assert failures(verdict, "manifest_parses")
    assert_named(verdict, DRY_RUN_CHECKS)


def test_retired_term_whose_replacement_is_not_active_fails(
    a2: Landed, a2_structure: Any, a2_import: Any, tmp_path: Path
) -> None:
    (retired,) = a2_import.authored.retirements
    assert (retired.code, retired.replaced_by) == (RETIRED_CODE, DUPLICATED[1])

    def drop_replacement(manifest: dict[str, Any]) -> None:
        replacement = _term(manifest, *DUPLICATED)
        manifest["terms"] = [t for t in manifest["terms"] if t is not replacement]

    imported = _with_manifest(a2_import, drop_replacement)
    verdict = _verify(imported, a2, a2_structure, tmp_path / "ledger" / "verify")

    (found,) = failures(verdict, "replaced_by_active")
    assert (found.evidence["code"], found.evidence["replaced_by"]) == (
        RETIRED_CODE,
        DUPLICATED[1],
    )
    assert_named(verdict, DRY_RUN_CHECKS)

"""Spec assembly over loader-fixed bindings (task 1.9): no hint, sections from ``SECTIONS``.

A3 is the oracle case: its structure and roles, with no hint, assemble a spec whose import
equals the hand-written ``ontology.spec.yaml`` import on the manifest and on term staging.
Term staging is compared at the descriptor's cap (nothing staged) and at a cap of one, so
every term is staged and the staged tables are compared row for row.
"""

from __future__ import annotations

import difflib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml

from sema.binding.importer import import_concept_source
from sema.binding.spec import RELATION_SECTIONS, ConceptSourceSpec, load_concept_source_spec
from sema.binding.spec_sections import TermStorageSpec
from sema.ingest.duckdb_staging import Staging
from sema.intake.roles.eligibility_utils import RunDescriptor
from sema.intake.land.loaders.athena import land_athena_bundle
from sema.intake.land.loaders.semsql import land_semsql
from sema.intake.land.relations import RelationProvider
from sema.intake.structure.shapes import athena_bundle, semsql
from tests.support.athena_bundle import export_schema, make_minimal_bundle
from tests.unit.intake.eligibility_env import descriptor, eligibility, run_roles
from tests.unit.intake.structure.shapes.shapes_env import Source
from tests.unit.intake.structure_env import A3_DB, fresh_tools, land, run

pytestmark = pytest.mark.unit

A3_FIXTURES = A3_DB.parent
A3_SPEC = A3_FIXTURES / "ontology.spec.yaml"
A3_RUN = A3_FIXTURES / "ontology.run.yaml"
ATHENA_RUN = {
    "id": "athena-minimal",
    "identity_system": "ATH",
    "release": "1",
    "scheme": {"qualified_name": "ath.vocabularies", "description": "A minimal vocabulary bundle"},
}
STAGE_EVERY_TERM = 1
FILLABLE_SECTIONS = (*RELATION_SECTIONS, "code_lists")


@dataclass
class Assembled:
    staging: Staging
    source: Source
    structure: Any
    roles: Any
    spec: ConceptSourceSpec | None


def _assemble(root: Path, name: str, lander: Any, path: Path, run_descriptor: Any) -> Assembled:
    staging, source = land(root, name, lander, path)
    structure = run(source, fresh_tools(source, root / "ledger" / "structure"))
    roles = run_roles(source, structure, root / "ledger" / "roles")
    spec = eligibility().assemble_spec(run_descriptor, roles)
    return Assembled(staging, source, structure, roles, spec)


@pytest.fixture(scope="module")
def a3(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Assembled]:
    root = tmp_path_factory.mktemp("assembly_a3")
    assembled = _assemble(root, "a3", land_semsql, A3_DB, descriptor(A3_RUN))
    yield assembled
    assembled.staging.close()


@pytest.fixture(scope="module")
def athena(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Assembled]:
    root = tmp_path_factory.mktemp("assembly_athena")
    bundle = root / "bundle"
    make_minimal_bundle(bundle)
    run_descriptor = RunDescriptor.model_validate(ATHENA_RUN)
    assembled = _assemble(root, "ath", land_athena_bundle, bundle, run_descriptor)
    yield assembled
    assembled.staging.close()


def _spec_diff(assembled: ConceptSourceSpec | None, written: ConceptSourceSpec) -> str:
    def dump(spec: ConceptSourceSpec | None) -> list[str]:
        raw = spec.model_dump(mode="json", by_alias=True, exclude_none=True) if spec else {}
        return yaml.safe_dump(raw, sort_keys=True).splitlines()

    lines = difflib.unified_diff(dump(written), dump(assembled), "written", "assembled", lineterm="")
    return "\n".join(lines)


def _import(spec: ConceptSourceSpec, staging: Staging) -> tuple[dict[str, Any], str]:
    provider = RelationProvider(warehouse=None, staging=staging)
    result = import_concept_source(spec, provider, dataset_map={}, staging=staging)
    return result.manifest, export_schema(staging, spec.external_namespace)


def _capped(spec: ConceptSourceSpec, cap: int | None) -> ConceptSourceSpec:
    if cap is None:
        return spec
    storage = spec.term_storage.model_dump() | {"inline_cap": cap}
    return spec.model_copy(update={"term_storage": TermStorageSpec.model_validate(storage)})


def _shape(spec: ConceptSourceSpec) -> list[tuple[str, str]]:
    return sorted((entry.section, entry.template) for entry in spec.shape or [])


def _filled(spec: ConceptSourceSpec) -> set[str]:
    return {name for name in FILLABLE_SECTIONS if getattr(spec, name) is not None}


def test_a3_structure_binds_semsql_alone_with_no_hint(a3: Assembled) -> None:
    assert a3.structure.ambiguities == ()
    assert {entry.binding.template for entry in a3.structure.bound} == {semsql.ID}


def test_a3_with_no_hint_assembles_a_spec(a3: Assembled) -> None:
    written, spec = load_concept_source_spec(A3_SPEC), a3.spec

    assert spec is not None, _spec_diff(spec, written)


def test_a3_assembled_shape_names_semsql_for_every_loader_section(a3: Assembled) -> None:
    spec = a3.spec
    assert spec is not None, "no spec assembled over the semsql binding"

    assert _shape(spec) == sorted((section, semsql.ID) for section in semsql.SECTIONS)
    assert _filled(spec) == set(semsql.SECTIONS)


@pytest.mark.parametrize("cap", [None, STAGE_EVERY_TERM], ids=["descriptor_cap", "stage_all"])
def test_a3_assembled_import_equals_the_written_spec_import(
    a3: Assembled, cap: int | None
) -> None:
    written, spec = load_concept_source_spec(A3_SPEC), a3.spec
    diff = _spec_diff(spec, written)
    assert spec is not None, diff

    manifest, staged = _import(_capped(spec, cap), a3.staging)
    expected_manifest, expected_staged = _import(_capped(written, cap), a3.staging)

    assert yaml.safe_dump(manifest, sort_keys=False) == yaml.safe_dump(
        expected_manifest, sort_keys=False
    ), diff
    assert staged == expected_staged, diff


def test_a3_stage_all_cap_stages_every_term(a3: Assembled) -> None:
    written = _capped(load_concept_source_spec(A3_SPEC), STAGE_EVERY_TERM)

    _, staged = _import(written, a3.staging)

    assert "## concept\n" in staged, "precondition: the cap of one stages the terms"


def test_athena_structure_binds_the_bundle_with_no_hint(athena: Assembled) -> None:
    assert athena.structure.ambiguities == ()
    assert {entry.binding.template for entry in athena.structure.bound} == {athena_bundle.ID}


def test_athena_with_no_hint_assembles_its_loader_sections(athena: Assembled) -> None:
    spec = athena.spec
    assert spec is not None, "no spec assembled over the athena_bundle binding"

    assert _shape(spec) == sorted(
        (section, athena_bundle.ID) for section in athena_bundle.SECTIONS
    )
    assert _filled(spec) == set(athena_bundle.SECTIONS)

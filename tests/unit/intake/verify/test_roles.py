"""Roles verifier: a spec's role bindings re-proved against eligibility and the landed data."""

from __future__ import annotations

import csv
import dataclasses
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

from sema.binding.spec import ConceptSourceSpec, load_concept_source_spec
from sema.ingest.duckdb_staging import Staging
from sema.intake.land.loaders.tabular import land_tabular
from tests.unit.intake.eligibility_env import COLOURS_SQL, memory_source, run_roles
from tests.unit.intake.structure.shapes.shapes_env import Source, staging_source
from tests.unit.intake.structure_env import A2_SPEC, fresh_tools, ref, run
from tests.unit.intake.verify.conftest import Landed
from tests.unit.intake.verify.verify_env import (
    ROLES_CHECKS,
    assert_named,
    assert_passes,
    edited_spec,
    failures,
    tools_for,
    verifier,
)

pytestmark = pytest.mark.unit

A3_SPEC = Path(__file__).parents[3] / "fixtures" / "intake" / "semsql" / "ontology.spec.yaml"
LIFECYCLE_AND_MAPPING_CHECKS = {"status_values", "replaced_by_resolves", "external_refs_resolve"}
ROLE_FLOORS = {"DISPLAY_NULL_RATE_FLOOR"}
DATED_ROWS = (
    ("code", "label", "valid_from", "valid_to"),
    ("D1", "First dated entry", "2020-01-01", "2021-01-01"),
    ("D2", "Second dated entry", "2020-02-01", "2021-02-01"),
    ("D3", "Third dated entry", "2022-03-01", "2021-03-01"),
    ("D4", "Fourth dated entry", "2020-04-01", "2021-04-01"),
    ("D5", "Fifth dated entry", "2020-05-01", "2021-05-01"),
)
INVERTED_CODE = "D3"
DATED_SPEC = {
    "concept_source": {
        "spec_version": 2,
        "id": "dated",
        "identity_system": "dated",
        "release": "1",
        "default_provider": "staging",
        "shape": [{"section": "code_lists.dated", "template": "concept_list"}],
        "code_lists": [
            {
                "vocabulary_id": "dated",
                "relation": "land_d.dated",
                "roles": {
                    "code": "code",
                    "display": "label",
                    "valid_from": "valid_from",
                    "valid_to": "valid_to",
                },
            }
        ],
    }
}


def _verify(source: Source, roles: Any, ledger: Path, spec: ConceptSourceSpec | None) -> Any:
    tools = tools_for(source, ledger, stage="roles", structure=roles.structure)
    return verifier("roles").verify_roles(roles, tools, spec=spec)


@pytest.fixture(scope="module")
def a2_roles(a2: Landed, a2_structure: Any) -> Any:
    return run_roles(a2.source, a2_structure, a2.root / "ledger" / "a2_roles")


@pytest.fixture(scope="module")
def a3_roles(a3: Landed) -> Any:
    structured = a3.structure("a3_roles_structure")
    return run_roles(a3.source, structured, a3.root / "ledger" / "a3_roles")


def _a2_edited(tmp_path: Path, **value_set_roles: str) -> ConceptSourceSpec:
    def edit(section: dict[str, Any]) -> None:
        section["value_sets"]["roles"].update(value_set_roles)

    return edited_spec(A2_SPEC, tmp_path / "spec", edit)


def _dated(tmp_path: Path, rows: tuple[tuple[str, ...], ...]) -> Iterator[Any]:
    path = tmp_path / "dated.csv"
    with path.open("w", newline="") as handle:
        csv.writer(handle).writerows(rows)
    spec_path = tmp_path / "dated.spec.yaml"
    spec_path.write_text(yaml.safe_dump(DATED_SPEC, sort_keys=False))
    with Staging(str(tmp_path / "staging.duckdb"), schemas=()) as staging:
        land = land_tabular(path, staging, schema="land_d")
        source = staging_source(staging, land, tmp_path / "ledger" / "landing")
        yield source, load_concept_source_spec(spec_path)


@pytest.fixture
def dated(tmp_path: Path) -> Iterator[tuple[Source, ConceptSourceSpec]]:
    yield from _dated(tmp_path, DATED_ROWS)


@pytest.fixture
def ordered(tmp_path: Path) -> Iterator[tuple[Source, ConceptSourceSpec]]:
    yield from _dated(tmp_path, tuple(row for row in DATED_ROWS if row[0] != INVERTED_CODE))


def test_checks_are_the_documented_list() -> None:
    assert set(verifier("roles").CHECKS) == ROLES_CHECKS


def test_a2_spec_passes_with_its_floors_recorded(
    a2: Landed, a2_roles: Any, tmp_path: Path
) -> None:
    spec = load_concept_source_spec(A2_SPEC)
    verdict = _verify(a2.source, a2_roles, tmp_path / "ledger" / "verify", spec)

    assert_passes(verdict, ROLES_CHECKS)
    assert ROLE_FLOORS <= set(verdict.floors)


def test_code_bound_to_a_column_with_nulls_is_outside_its_eligibility_set(
    a2: Landed, a2_roles: Any, tmp_path: Path
) -> None:
    spec = _a2_edited(tmp_path, code="successor")
    verdict = _verify(a2.source, a2_roles, tmp_path / "ledger" / "verify", spec)

    found = failures(verdict, "in_eligibility_set")
    assert [(f.relation, f.column, f.evidence["role"]) for f in found] == [
        (ref(a2.source, "codes_a2__Values"), "successor", "code")
    ]
    assert_named(verdict, ROLES_CHECKS)


def test_one_column_holding_code_and_display_fails(
    a2: Landed, a2_roles: Any, tmp_path: Path
) -> None:
    spec = _a2_edited(tmp_path, display="code")
    verdict = _verify(a2.source, a2_roles, tmp_path / "ledger" / "verify", spec)

    found = failures(verdict, "one_role_per_column")
    values = ref(a2.source, "codes_a2__Values")
    assert [(f.relation, f.column) for f in found] == [(values, "code")]
    assert_named(verdict, ROLES_CHECKS)


def test_replaced_by_that_resolves_to_no_code_fails(
    a2: Landed, a2_roles: Any, tmp_path: Path
) -> None:
    spec = _a2_edited(tmp_path, replaced_by="rank")
    verdict = _verify(a2.source, a2_roles, tmp_path / "ledger" / "verify", spec)

    (found,) = failures(verdict, "replaced_by_resolves")
    assert (found.relation, found.column) == (ref(a2.source, "codes_a2__Values"), "rank")
    assert_named(verdict, ROLES_CHECKS)


def test_ambiguity_list_missing_a_multi_member_role_fails(
    a2: Landed, a2_roles: Any, tmp_path: Path
) -> None:
    assert a2_roles.ambiguities, "precondition: A2 roles list multi-member sets"
    emptied = dataclasses.replace(a2_roles, ambiguities=())
    spec = load_concept_source_spec(A2_SPEC)
    verdict = _verify(a2.source, emptied, tmp_path / "ledger" / "verify", spec)

    listed = {a.subject.rpartition(".")[0] for a in a2_roles.ambiguities}
    found = failures(verdict, "ambiguity_list_complete")
    assert {f.relation for f in found} == listed
    assert_named(verdict, ROLES_CHECKS)


def test_unresolved_section_fails(tmp_path: Path) -> None:
    con, source = memory_source(COLOURS_SQL, "tc", tmp_path)
    structured = run(source, fresh_tools(source, tmp_path / "ledger" / "structure"))
    roles = run_roles(source, structured, tmp_path / "ledger" / "roles")
    verdict = _verify(source, roles, tmp_path / "ledger" / "verify", None)
    con.close()

    (found,) = failures(verdict, "section_resolved")
    assert found.relation == "tc.colours"
    assert_named(verdict, ROLES_CHECKS)


def test_a3_lifecycle_and_mapping_checks_pass(a3: Landed, tmp_path: Path) -> None:
    structured = a3.structure("a3_structure")
    roles = run_roles(a3.source, structured, tmp_path / "ledger" / "roles")
    spec = load_concept_source_spec(A3_SPEC)
    verdict = _verify(a3.source, roles, tmp_path / "ledger" / "verify", spec)

    assert LIFECYCLE_AND_MAPPING_CHECKS <= ROLES_CHECKS
    found = [f for f in verdict.failures if f.check in LIFECYCLE_AND_MAPPING_CHECKS]
    assert found == [], [(f.check, f.relation, f.column, f.evidence) for f in found]
    assert_named(verdict, ROLES_CHECKS)


def test_valid_from_after_valid_to_fails(
    dated: tuple[Source, ConceptSourceSpec], tmp_path: Path
) -> None:
    source, spec = dated
    structured = run(source, fresh_tools(source, tmp_path / "ledger" / "structure"), spec)
    roles = run_roles(source, structured, tmp_path / "ledger" / "roles", hint=spec)
    verdict = _verify(source, roles, tmp_path / "ledger" / "verify", spec)

    (found,) = failures(verdict, "valid_dates")
    assert found.relation == ref(source, "dated")
    assert found.column in {"valid_from", "valid_to"}
    assert_named(verdict, ROLES_CHECKS)


def test_dated_rows_in_order_pass_the_date_check(
    ordered: tuple[Source, ConceptSourceSpec], tmp_path: Path
) -> None:
    source, spec = ordered
    structured = run(source, fresh_tools(source, tmp_path / "ledger" / "structure"), spec)
    roles = run_roles(source, structured, tmp_path / "ledger" / "roles", hint=spec)
    verdict = _verify(source, roles, tmp_path / "ledger" / "verify", spec)

    assert not failures(verdict, "valid_dates")
    assert_named(verdict, ROLES_CHECKS)


def test_a3_spec_passes_with_its_loader_section_and_constant_shared_key(
    a3: Landed, a3_roles: Any, tmp_path: Path
) -> None:
    spec = load_concept_source_spec(A3_SPEC)
    assert spec.concepts is not None
    assert spec.concepts.group_by == "ontology", "precondition: A3 groups by the constant key"

    verdict = _verify(a3.source, a3_roles, tmp_path / "ledger" / "verify", spec)

    assert_passes(verdict, ROLES_CHECKS)


@pytest.mark.parametrize(
    ("group_by", "shared"),
    [("subject", "subject"), ("ontology", "label")],
    ids=["non_constant_group_by", "not_the_group_by_key"],
)
def test_concept_key_and_name_sharing_a_non_constant_column_fail(
    a3: Landed, a3_roles: Any, tmp_path: Path, group_by: str, shared: str
) -> None:
    def edit(section: dict[str, Any]) -> None:
        section["concepts"].update(
            group_by=group_by, roles={"concept_id": shared, "name": shared}
        )

    spec = edited_spec(A3_SPEC, tmp_path / "spec", edit)
    verdict = _verify(a3.source, a3_roles, tmp_path / "ledger" / "verify", spec)

    found = failures(verdict, "one_role_per_column")
    terms = ref(a3.source, "semsql_terms")
    assert [(f.relation, f.column, f.evidence["roles"]) for f in found] == [
        (terms, shared, ["concept_id", "name"])
    ]
    assert_named(verdict, ROLES_CHECKS)

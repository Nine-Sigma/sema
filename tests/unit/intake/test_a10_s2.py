"""A10 end to end without a model (task 13.1): land, structure, roles, the four verifiers, and
the dry run under ``metadata_only`` into one ledger and one registry row.

The hinted run takes ``acris.spec.yaml`` as the hint for Structure and Roles and the A10 run
descriptor for assembly. The absent-code-table run removes ``Document_Control_Codes.csv``
from a copy of the directory; its hint is the hand-written spec without that code list's
entry and shape line, since a hint cannot name a relation that did not land.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

from sema.binding.spec import ConceptSourceSpec, load_concept_source_spec
from sema.intake.ledger import Ledger
from sema.intake.profile import EvidencePolicy, ProfileTools
from sema.intake.structure import run_structure
from tests.unit.intake.eligibility_env import (
    ACRIS_RUN,
    GROUPED_CODE_TABLE,
    MISSPELLED_PROPERTY,
    copy_without,
    descriptor,
    normalized,
    write_hint,
)
from tests.unit.intake.pipeline_env import (
    EXPLORATORY_STAGES,
    FINAL_STAGE,
    PROVIDER,
    TIER,
    Landed,
    StageRun,
    describe,
    landed_run,
    read_ambiguity_list,
    run_stages,
    statement_relations,
)
from tests.unit.intake.structure_env import ACRIS, ACRIS_SPEC, CODE_TABLES

pytestmark = pytest.mark.unit

SCHEMA = "land_acris"
DICTIONARY_SHEETS = ("__Column_Info", "__Dataset_Info")
UNRESOLVED = "unresolved_reference"
UNRESOLVED_COUNT = 2


def _hint_without_grouped_table(directory: Path) -> ConceptSourceSpec:
    raw = yaml.safe_load(ACRIS_SPEC.read_text())
    section = raw["concept_source"]
    section["code_lists"] = [
        e for e in section["code_lists"] if e["vocabulary_id"] != GROUPED_CODE_TABLE
    ]
    section["shape"] = [s for s in section["shape"] if GROUPED_CODE_TABLE not in s["section"]]
    return write_hint(directory, raw)


@pytest.fixture(scope="module")
def acris(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Landed]:
    with landed_run(ACRIS, tmp_path_factory.mktemp("a10_s2"), SCHEMA) as landed:
        yield landed


@pytest.fixture(scope="module")
def hinted(acris: Landed) -> StageRun:
    return run_stages(acris, descriptor(ACRIS_RUN), load_concept_source_spec(ACRIS_SPEC))


@pytest.fixture(scope="module")
def dropped(tmp_path_factory: pytest.TempPathFactory) -> Iterator[StageRun]:
    root = tmp_path_factory.mktemp("a10_s2_nodcc")
    files = copy_without(root / "files", GROUPED_CODE_TABLE)
    hint = _hint_without_grouped_table(root / "hint")
    with landed_run(files, root, SCHEMA) as landed:
        yield run_stages(landed, descriptor(ACRIS_RUN), hint)


def _domains_of(run: StageRun, property_name: str) -> list[str | None]:
    assert run.imported is not None, "no spec assembled, so no Table Schemas"
    return [
        f.x_sema_domain
        for schema in run.imported.table_schemas.values()
        for f in schema.fields
        if f.name == property_name
    ]


def test_land_structure_and_roles_verdicts_pass(hinted: StageRun) -> None:
    assert hinted.land_verdict.status == "pass", describe(hinted.land_verdict)
    assert hinted.structure_verdict.status == "pass", describe(hinted.structure_verdict)
    assert hinted.roles_verdict.status == "pass", describe(hinted.roles_verdict)
    assert hinted.dry_run_verdict is not None
    assert hinted.dry_run_verdict.status == "pass", describe(hinted.dry_run_verdict)


def test_assembled_spec_equals_the_hand_written_spec(hinted: StageRun) -> None:
    assert hinted.spec is not None, [(a.subject, a.members) for a in hinted.roles.ambiguities]

    assert normalized(hinted.spec) == normalized(load_concept_source_spec(ACRIS_SPEC))


def test_exploratory_value_reads_are_over_dictionary_relations(hinted: StageRun) -> None:
    landed = hinted.landed
    dictionaries = {
        landed.qualified(r.name)
        for r in landed.land.relations
        if r.name.endswith(DICTIONARY_SHEETS)
    }
    lines = landed.value_bearing(EXPLORATORY_STAGES)

    assert lines, "Roles reads the dictionary text, so some statement is value-bearing"
    assert all(line["schema_relation"] for line in lines), [
        line["statement"] for line in lines if not line["schema_relation"]
    ]
    for line in lines:
        named = statement_relations(line["statement"])
        assert named and named <= dictionaries, line["statement"]


def test_registry_reaches_dry_run_with_zero_model_calls(hinted: StageRun) -> None:
    landed = hinted.landed
    record = landed.record()

    assert hinted.report is not None
    assert (record.stage, record.model_calls) == (FINAL_STAGE, 0)
    assert landed.model_calls() == []
    assert (landed.ledger.directory / "artifacts" / "dry_run_report.json").is_file()


def test_without_the_hint_the_five_code_tables_bind_concept_list_by_code(
    acris: Landed, tmp_path: Path
) -> None:
    # Amended 2026-10-21: each code table's columns equal the properties its bound
    # dictionary lists, so no hint is needed to settle them.
    ledger = Ledger(tmp_path).run("unhinted")
    tools = ProfileTools(
        acris.provider, EvidencePolicy(tier=TIER), ledger, stage="structure"
    )
    result = run_structure(acris.land, tools, provider=PROVIDER, hint=None)

    code_bound = {
        name
        for entry in result.bound
        if (entry.binding.template, entry.source) == ("concept_list", "code")
        for name in entry.binding.relations
    }
    assert result.ambiguities == ()
    assert code_bound == {acris.qualified(name) for name in CODE_TABLES}
    assert result.proposal_targets == ()


def test_hinted_run_binds_the_misspelled_property_to_the_grouped_table(
    hinted: StageRun,
) -> None:
    assert _domains_of(hinted, MISSPELLED_PROPERTY) == [GROUPED_CODE_TABLE] * UNRESOLVED_COUNT


def test_absent_code_table_leaves_two_unresolved_references(dropped: StageRun) -> None:
    unresolved = [a for a in dropped.roles_verdict.ambiguities if UNRESOLVED in a.hints]

    assert len(unresolved) == UNRESOLVED_COUNT, [a.subject for a in unresolved]
    assert all(MISSPELLED_PROPERTY in a.subject for a in unresolved)
    assert all(a.choice is None for a in unresolved)


def test_absent_code_table_leaves_no_domain_on_the_misspelled_property(
    dropped: StageRun,
) -> None:
    domains = _domains_of(dropped, MISSPELLED_PROPERTY)

    assert len(domains) == UNRESOLVED_COUNT
    assert domains == [None] * UNRESOLVED_COUNT


def test_ambiguity_lists_reach_the_ledger(hinted: StageRun) -> None:
    ledger = hinted.landed.ledger
    lists: dict[str, Any] = {
        stage: read_ambiguity_list(ledger, stage) for stage in ("structure", "roles")
    }

    assert lists["structure"].count == len(hinted.structure.ambiguities)
    assert lists["roles"].count == len(hinted.roles.ambiguities)

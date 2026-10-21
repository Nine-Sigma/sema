"""Structure verifier: a structure proposal re-proved against the landed relations."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from sema.ingest.duckdb_staging import Staging
from sema.intake.loaders.tabular import land_tabular
from sema.intake.shapes import concept_list
from sema.intake.shapes.base import SlotBinding, profile_relations
from tests.unit.intake.shapes.shapes_env import Source, staging_source, write_workbook
from tests.unit.intake.structure_env import (
    CODE_TABLES,
    GROUPED_CODE_TABLE,
    fresh_tools,
    ref,
    run,
)
from tests.unit.intake.verify.conftest import Landed
from tests.unit.intake.verify.verify_env import (
    STRUCTURE_CHECKS,
    assert_named,
    assert_passes,
    failures,
    tools_for,
    verifier,
    without_relation,
)

pytestmark = pytest.mark.unit

GROUPS = ("g1", "g2", "g3")
OPTIONS: list[list[object]] = [["code", "option"], ["A", "Alpha"], ["B", "Beta"], ["C", "Gamma"]]
CHANGED_OPTIONS = OPTIONS[:-1] + [["C", "Other"]]
DIFFERING_COLUMN = "option"
BOILERPLATE = {"reason": "boilerplate", "would_match": "concept_list"}


def _verify(proposal: Any, source: Source, ledger: Path) -> Any:
    tools = tools_for(source, ledger, stage="structure")
    return verifier("structure").verify_structure(
        proposal, source.land, tools, provider=source.provider
    )


def _sheets(group: str, options: list[list[object]]) -> dict[str, list[list[object]]]:
    return {
        "Main": [
            ["name", "description"],
            [f"{group}_a", f"First item of {group}"],
            [f"{group}_b", f"Second item of {group}"],
        ],
        "Opts": options,
    }


@pytest.fixture
def differing(tmp_path: Path) -> Iterator[Source]:
    directory = tmp_path / "books"
    directory.mkdir()
    for group in GROUPS:
        options = CHANGED_OPTIONS if group == GROUPS[-1] else OPTIONS
        write_workbook(directory / f"{group}.xlsx", _sheets(group, options))
    with Staging(str(tmp_path / "staging.duckdb"), schemas=()) as staging:
        land = land_tabular(directory, staging, schema="land_s")
        yield staging_source(staging, land, tmp_path / "ledger" / "landing")


def _entry(result: Any, relation: str) -> Any:
    (found,) = [e for e in result.bound if relation in e.binding.relations]
    return found


def _replace_entry(result: Any, old: Any, new: Any) -> Any:
    bound = tuple(new if entry is old else entry for entry in result.bound)
    return result.model_copy(update={"bound": bound})


def test_checks_are_the_documented_list() -> None:
    assert set(verifier("structure").CHECKS) == STRUCTURE_CHECKS


def test_a2_structure_output_passes(a2: Landed, a2_structure: Any, tmp_path: Path) -> None:
    from sema.intake.structure import stage_floors

    verdict = _verify(a2_structure, a2.source, tmp_path / "ledger" / "verify")

    assert_passes(verdict, STRUCTURE_CHECKS)
    assert set(stage_floors()) <= set(verdict.floors)


def test_hinted_acris_structure_output_passes(
    acris: Landed, acris_hinted: Any, tmp_path: Path
) -> None:
    verdict = _verify(acris_hinted, acris.source, tmp_path / "ledger" / "verify")

    assert_passes(verdict, STRUCTURE_CHECKS)


def test_unhinted_acris_output_with_its_five_ambiguities_passes(
    acris: Landed, acris_structure: Any, tmp_path: Path
) -> None:
    code_tables = sorted(ref(acris.source, name) for name in CODE_TABLES)
    verdict = _verify(acris_structure, acris.source, tmp_path / "ledger" / "verify")

    assert sorted(a.subject for a in acris_structure.ambiguities) == code_tables
    assert_passes(verdict, STRUCTURE_CHECKS)


def test_code_chosen_binding_omitted_from_the_ambiguity_list_fails(
    acris: Landed, acris_hinted: Any, tmp_path: Path
) -> None:
    grouped = ref(acris.source, GROUPED_CODE_TABLE)
    pinned = _entry(acris_hinted, grouped)
    proposal = _replace_entry(acris_hinted, pinned, pinned.model_copy(update={"source": "code"}))
    verdict = _verify(proposal, acris.source, tmp_path / "ledger" / "verify")

    (found,) = failures(verdict, "ambiguity_list_complete")
    assert found.relation == grouped
    assert_named(verdict, STRUCTURE_CHECKS)


def test_dropped_ambiguity_fails_the_partition_and_the_ambiguity_list(
    acris: Landed, acris_structure: Any, tmp_path: Path
) -> None:
    code_tables = sorted(ref(acris.source, name) for name in CODE_TABLES)
    proposal = acris_structure.model_copy(update={"ambiguities": ()})
    verdict = _verify(proposal, acris.source, tmp_path / "ledger" / "verify")

    partition = failures(verdict, "partition_complete")
    listed = failures(verdict, "ambiguity_list_complete")
    assert sorted(f.relation for f in partition) == code_tables
    assert sorted(f.relation for f in listed) == code_tables
    assert_named(verdict, STRUCTURE_CHECKS)


def test_template_outside_the_closed_set_fails(
    a2: Landed, a2_structure: Any, tmp_path: Path
) -> None:
    hierarchy = ref(a2.source, "codes_a2__Hierarchy")
    entry = _entry(a2_structure, hierarchy)
    renamed = entry.binding.model_copy(update={"template": "no_such_template"})
    proposal = _replace_entry(a2_structure, entry, entry.model_copy(update={"binding": renamed}))
    verdict = _verify(proposal, a2.source, tmp_path / "ledger" / "verify")

    (found,) = failures(verdict, "unknown_template")
    assert found.relation == hierarchy
    assert_named(verdict, STRUCTURE_CHECKS)


def test_single_relation_binding_over_a_claimed_relation_fails_precedence(
    a2: Landed, a2_structure: Any, tmp_path: Path
) -> None:
    concepts = ref(a2.source, "codes_a2__Concepts")
    tools = fresh_tools(a2.source, tmp_path / "ledger" / "candidates")
    profiles = profile_relations(a2.source.land, tools, provider=a2.source.provider)
    (profile,) = [p for p in profiles if p.ref.name == concepts]
    single = concept_list.candidates([profile], tools)
    assert single, "precondition: concept_list is eligible on the claimed relation alone"
    claimed = _entry(a2_structure, concepts)
    proposal = _replace_entry(
        a2_structure, claimed, claimed.model_copy(update={"binding": single[0]})
    )
    verdict = _verify(proposal, a2.source, tmp_path / "ledger" / "verify")

    assert concepts in [f.relation for f in failures(verdict, "precedence_applied")]
    assert_named(verdict, STRUCTURE_CHECKS)


def test_ignore_reason_outside_the_closed_list_fails(
    acris: Landed, acris_hinted: Any, tmp_path: Path
) -> None:
    first = acris_hinted.ignored[0]
    renamed = first.model_copy(update={"slots": {**first.slots, "reason": "form_metadata"}})
    proposal = acris_hinted.model_copy(
        update={"ignored": (renamed, *acris_hinted.ignored[1:])}
    )
    verdict = _verify(proposal, acris.source, tmp_path / "ledger" / "verify")

    (found,) = failures(verdict, "ignore_reason_known")
    assert found.relation == first.relations[0]
    assert_named(verdict, STRUCTURE_CHECKS)


def test_boilerplate_claim_whose_counterpart_differs_fails_the_profile_recheck(
    differing: Source, tmp_path: Path
) -> None:
    result = run(differing, fresh_tools(differing, tmp_path / "ledger" / "structure"))
    changed = differing.ref(f"{GROUPS[-1]}__Opts")
    claim = SlotBinding(template="ignore", slots=dict(BOILERPLATE), relations=(changed,))
    kept = without_relation(result, changed)
    proposal = result.model_copy(update={**kept, "ignored": (*kept["ignored"], claim)})
    verdict = _verify(proposal, differing, tmp_path / "ledger" / "verify")

    (found,) = failures(verdict, "ignore_profile_holds")
    assert (found.relation, found.column) == (changed, DIFFERING_COLUMN)
    assert_named(verdict, STRUCTURE_CHECKS)


def test_structure_verifier_reads_no_value(
    a2: Landed, a2_structure: Any, tmp_path: Path
) -> None:
    ledger = tmp_path / "ledger" / "verify"
    _verify(a2_structure, a2.source, ledger)
    lines = Source(a2.source.land, a2.source.tools, "staging", ledger).value_bearing()

    assert lines == []

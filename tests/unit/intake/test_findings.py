"""A data fact gates; a measured reading is a finding (S5 decisions 3, 18, 22, 23; task 3.2).

Pinned contract (rulings R1 to R7), over invented relations in DuckDB:

- A. A finding-class constant keeps the candidate or member and attaches a finding named
  ``group_by_ratio``, ``display_null_rate``, ``definition_mean_length``,
  ``entity_distinct_ratio``, ``boilerplate_share``, or ``property_distinct_ratio`` with
  the measured value (``SlotBinding.findings`` on a binding; the eligibility's dump for a
  column role).
- B. A gate removes the candidate: a null group key; a relation identical in fewer than
  ``BOILERPLATE_GROUP_SHARE_FLOOR`` of the other groups is not ignored (decision 22); a code
  table a landed dictionary describes keeps no ``entity_model`` candidate, and the run
  records ``described_by`` with the dictionary (decision 23).
- C. ``Verdict.floor_classes`` classes every listed constant ``gate`` or ``finding``.
- D. A roles verdict lists ``display_null_rate`` in ``Verdict.findings``, not in failures.
- E. Guard: the A10 replay reaches Review with an empty structure list (decision 23).
- F. The structure prompt and the Review entry of an open structure choice state each
  member's findings by name and value (decision 3).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from sema.intake.resolve_utils import member_bindings, open_entries, render_prompt
from sema.intake.verify.roles import verify_roles
from sema.intake.verify.structure import verify_structure
from tests.unit.intake.findings_env import (
    DENSE, DENSE_COLUMNS, DENSE_RATIO, DENSE_ROWS, DICTIONARY, DICTIONARY_COLUMNS,
    DICTIONARY_ROWS, FIELD_COLUMNS, FIELD_RATIO, FIELD_ROWS, FIELDS, HEADER, HEADER_COLUMNS,
    HEADER_ROWS, HOLED_ROWS, KIND_COLUMNS, KIND_ROWS, KINDS, SPARSE, SPARSE_COLUMNS,
    SPARSE_RATE, SPARSE_ROWS, TERSE, TERSE_COLUMNS, TERSE_MEAN, TERSE_ROWS, create,
    findings_of, landed, mentions, options_groups, qualified, stated,
)
from tests.unit.intake.no_hint_env import KEYS_MOVED, replay, utils
from tests.unit.intake.role_entries_env import concept_list, pinned, roles_of
from tests.unit.intake.shapes.shapes_env import Source, candidates
from tests.unit.intake.structure_env import fresh_tools, run
from tests.unit.intake.verify.verify_env import tools_for

pytestmark = pytest.mark.unit

DENORMALIZED, ENTITY_MODEL, IGNORE = "concept_with_values_denormalized", "entity_model", "ignore"
FINDING, GATE = "finding", "gate"
CLASSES = {
    "DENORMALIZED_GROUP_RATIO_FLOOR": FINDING, "ID_DISTINCT_RATIO_FLOOR": FINDING,
    "ENTITY_DISTINCT_RATIO_CAP": FINDING, "BOILERPLATE_GROUP_SHARE_FLOOR": GATE,
    "JOIN_RESOLVE_FLOOR": GATE, "LABEL_COVERAGE_FLOOR": GATE,
}


def _over(source: Source, template: str, table: str) -> list[Any]:
    return candidates(template, [source.profile(table)], source)


def _grouped_by(found: list[Any], column: str) -> list[Any]:
    return [b for b in found if b.slots.get("group_by") == column]


def test_a_group_ratio_above_the_floor_keeps_the_candidate_with_the_finding(
    tmp_path: Path,
) -> None:
    with landed(tmp_path, create(DENSE, DENSE_COLUMNS, DENSE_ROWS)) as source:
        found = _grouped_by(_over(source, DENORMALIZED, DENSE), "set_key")

    assert found, "no concept_with_values_denormalized candidate over set_key"
    assert stated(found[0], "group_by_ratio") == pytest.approx(DENSE_RATIO)


def test_an_entity_ratio_above_the_cap_keeps_the_candidate_with_the_finding(
    tmp_path: Path,
) -> None:
    with landed(tmp_path, create(FIELDS, FIELD_COLUMNS, FIELD_ROWS)) as source:
        found = [
            b for b in _over(source, ENTITY_MODEL, FIELDS)
            if b.slots.get("entity") == "table_label"
        ]

    assert found, "no single_relation entity_model candidate over table_label"
    assert stated(found[0], "entity_distinct_ratio") == pytest.approx(FIELD_RATIO)


def test_an_undescribed_code_table_keeps_the_entity_model_candidate_with_the_finding(
    tmp_path: Path,
) -> None:
    with landed(tmp_path, create(KINDS, KIND_COLUMNS, KIND_ROWS)) as source:
        found = [
            b for b in _over(source, ENTITY_MODEL, KINDS)
            if b.slots.get("property") == "kind_code"
        ]

    assert found, "no single_relation entity_model candidate with kind_code as property"
    assert stated(found[0], "property_distinct_ratio") == pytest.approx(1.0)


def _ignored(root: Path, identical: int, groups: int) -> dict[str, Any]:
    with landed(root, *options_groups(identical, groups)) as source:
        profiles = list(source.profiles())
        found = candidates(IGNORE, profiles, source)
    return {binding.relations[0]: binding for binding in found}


def test_an_ignored_boilerplate_relation_states_its_share(tmp_path: Path) -> None:
    ignored = _ignored(tmp_path, identical=3, groups=3)

    binding = ignored.get(qualified("g1__options"))
    assert binding is not None and binding.slots["reason"] == "boilerplate", ignored
    assert float(stated(binding, "boilerplate_share")) == pytest.approx(1.0)


def _eligibility(roles: Any, relation: str, role: str) -> Any:
    (found,) = [
        e for e in roles.eligibilities
        if e.relation == relation and e.role == role and e.subject is None
    ]
    return found


def _roles(source: Source, root: Path, binding: Any) -> Any:
    structure = pinned(source, [binding], root / "pinned")
    return roles_of(source, structure, root / "roles")


def test_a_sparse_display_column_stays_a_member_with_the_finding(tmp_path: Path) -> None:
    with landed(tmp_path, create(SPARSE, SPARSE_COLUMNS, SPARSE_ROWS)) as source:
        relation = qualified(SPARSE)
        roles = _roles(source, tmp_path, concept_list(relation, f"code_lists.{SPARSE}"))

    display = _eligibility(roles, relation, "display")
    assert "item_label" in display.members, display
    assert mentions(display, "display_null_rate", "item_label", str(SPARSE_RATE)), display


def test_a_short_definition_column_stays_a_member_with_the_finding(tmp_path: Path) -> None:
    with landed(tmp_path, create(TERSE, TERSE_COLUMNS, TERSE_ROWS)) as source:
        relation = qualified(TERSE)
        roles = _roles(source, tmp_path, concept_list(relation, "concepts"))

    definition = _eligibility(roles, relation, "definition")
    assert "definition" in definition.members, definition
    assert mentions(definition, "definition_mean_length", str(TERSE_MEAN)), definition


def test_a_null_group_key_is_never_a_denormalized_candidate(tmp_path: Path) -> None:
    """Regression guard: the null test is a gate on HEAD and stays one."""
    with landed(tmp_path, create(DENSE, DENSE_COLUMNS, HOLED_ROWS)) as source:
        found = _grouped_by(_over(source, DENORMALIZED, DENSE), "set_key")

    assert found == []


def test_a_unique_group_key_is_never_a_denormalized_candidate(tmp_path: Path) -> None:
    """Decision 28: a group of one row makes every per-group test vacuous (a gate)."""
    with landed(tmp_path, create(KINDS, KIND_COLUMNS, KIND_ROWS)) as source:
        found = _grouped_by(_over(source, DENORMALIZED, KINDS), "kind_code")

    assert found == []


def test_a_unique_entity_column_is_never_an_entity_model_candidate(tmp_path: Path) -> None:
    """Decision 28: an entity of one row makes "unique per entity" vacuous (a gate)."""
    with landed(tmp_path, create(KINDS, KIND_COLUMNS, KIND_ROWS)) as source:
        found = [
            b for b in _over(source, ENTITY_MODEL, KINDS) if b.slots.get("entity") == "kind_code"
        ]

    assert found == []


def test_identity_in_fewer_than_the_share_of_groups_is_not_boilerplate(tmp_path: Path) -> None:
    """Regression guard (decision 22): the share floor stays a gate."""
    ignored = _ignored(tmp_path, identical=2, groups=5)

    assert qualified("g1__options") not in ignored, ignored
    assert qualified("g2__options") not in ignored, ignored


def _described_source(root: Path) -> Any:
    return landed(
        root,
        create(KINDS, KIND_COLUMNS, KIND_ROWS),
        create(DICTIONARY, DICTIONARY_COLUMNS, DICTIONARY_ROWS),
        create(HEADER, HEADER_COLUMNS, HEADER_ROWS),
    )


def test_a_described_code_table_keeps_no_entity_model_and_records_its_dictionary(
    tmp_path: Path,
) -> None:
    with _described_source(tmp_path) as source:
        structure = run(source, fresh_tools(source, tmp_path / "structure"))
        verdict = verify_structure(structure, source.land, source.tools, provider="warehouse")

    kinds = qualified(KINDS)
    over = [e.binding for e in structure.bound if kinds in e.binding.relations]
    assert not [b for b in over if b.template == ENTITY_MODEL], over
    assert not [a for a in structure.ambiguities if a.subject == kinds and ENTITY_MODEL in
                " ".join(a.members)], structure.ambiguities
    recorded = [item for item in (structure, verdict) if mentions(item, "described_by", kinds)]
    assert recorded and mentions(recorded[0], qualified(DICTIONARY)), "described_by not recorded"


def test_the_structure_verdict_classes_every_constant(tmp_path: Path) -> None:
    with landed(tmp_path, create(DENSE, DENSE_COLUMNS, DENSE_ROWS)) as source:
        structure = run(source, fresh_tools(source, tmp_path / "structure"))
        verdict = verify_structure(structure, source.land, source.tools, provider="warehouse")

    classes: dict[str, str] = getattr(verdict, "floor_classes")
    assert set(classes) == set(verdict.floors), set(classes) ^ set(verdict.floors)
    assert set(classes.values()) <= {GATE, FINDING}, classes
    assert {name: classes.get(name) for name in CLASSES} == CLASSES


def _states_findings(lines: list[str], member: str, binding: Any) -> bool:
    readings = [f"{name} {value:.4g}" for name, value in findings_of(binding)]
    return any(member in line and all(r in line for r in readings) for line in lines)


def _open_choice(root: Path) -> tuple[Any, dict[str, Any]]:
    with landed(root, create(FIELDS, FIELD_COLUMNS, FIELD_ROWS)) as source:
        tools = fresh_tools(source, root / "structure")
        structure = run(source, tools)
        (entry,) = open_entries(structure, member_bindings(structure, source.profiles(), tools))
    carrying = {m: b for m, b in entry.bindings.items() if b.findings}
    assert carrying, entry.bindings
    return entry, carrying


def test_the_structure_prompt_states_each_members_findings(tmp_path: Path) -> None:
    entry, carrying = _open_choice(tmp_path)
    prompt = render_prompt([entry], (), "", ()).splitlines()

    unstated = [m for m, b in carrying.items() if not _states_findings(prompt, m, b)]
    assert not unstated, f"the prompt omits the findings of {unstated}"


def test_the_review_entry_lists_each_members_findings(tmp_path: Path) -> None:
    entry, carrying = _open_choice(tmp_path)
    hints = list(entry.ambiguity.hints)

    unlisted = [m for m, b in carrying.items() if not _states_findings(hints, m, b)]
    assert not unlisted, f"the Review entry omits the findings of {unlisted}: {hints}"


def test_the_roles_verdict_lists_a_finding_beside_its_failures(tmp_path: Path) -> None:
    """A guard once task 2.2 lands (it adds ``Verdict.findings``)."""
    with landed(tmp_path, create(SPARSE, SPARSE_COLUMNS, SPARSE_ROWS)) as source:
        relation = qualified(SPARSE)
        roles = _roles(source, tmp_path, concept_list(relation, f"code_lists.{SPARSE}"))
        tools = tools_for(source, tmp_path / "verify", stage="roles", structure=roles.structure)
        verdict = verify_roles(roles, tools, spec=None, relations=(relation,))

    listed = [(f.name, f.subject) for f in getattr(verdict, "findings")]
    assert ("display_null_rate", f"{relation}.item_label") in listed, listed
    assert not [f for f in verdict.failures if f.check == "display_null_rate"], verdict.failures


def test_the_a10_replay_keeps_its_structure_list_empty(tmp_path: Path) -> None:
    """Guard (decision 23): a known source, replayed at zero calls."""
    try:
        outcome = replay("a10", tmp_path).outcome
    except utils().ReplayMismatchError:
        # Decision 26: a10's Roles keys moved until the 4.2 re-record; Structure still ran.
        if "a10" not in KEYS_MOVED:
            raise
        (artifacts,) = {p.parent for p in tmp_path.glob("*/artifacts/structure_ambiguities.json")}
        metrics = json.loads((artifacts / "run_metrics.json").read_text())
    else:
        artifacts = tmp_path / outcome.run_id / "artifacts"
        assert outcome.payload is not None
        metrics = outcome.payload["run_metrics"]

    assert metrics["open_structure"] == 0
    assert json.loads((artifacts / "structure_ambiguities.json").read_text())["ambiguities"] == []


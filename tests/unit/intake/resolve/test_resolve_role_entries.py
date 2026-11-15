"""Every open role entry reaches the model on its own field, among unclaimed members.

Pinned contract (known-source misses F1, F2, F4, team lead 2026-10-21):

- F1. Two open entries for the same role on one relation (a denormalized relation's
  concept-side and value-set-side ``status``) each get their own role-map field, the
  repeat qualified by section, and each entry reads its own answer (S5 decision 1).
- F2. Every claim in force (``code`` or ``lexicon``) is stated in the prompt; a model
  choice of a claimed column fails ``one_role_per_column`` and the claim stays in force.
- F4. An open ``<relation>.section`` entry is asked in its relation's role map, also when
  it is the relation's only open entry; a stated model choice settles the section through
  ``resolve_sections``; an abstention leaves it open with the model's reason.

The mocked model answers every field of the output model it is handed over the members
its prompt states, keeping each claim in force.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from sema.intake.roles.eligibility_utils import StatedChoice
from sema.intake.resolve.output_models import role_field
from sema.intake.resolve.output_models_utils import ABSTAIN
from sema.intake.verify.roles import ONE_ROLE_PER_COLUMN, verify_roles
from tests.unit.intake.resolve_roles_env import (
    MEMORY_TABLES,
    VALUE_TABLES,
    Synthetic,
    asked_columns,
    distinct,
    lines_with,
)
from tests.unit.intake.resolve_roles_env import answering as answering_roles
from tests.unit.intake.resolve_roles_env import prompts as sent_prompts
from tests.unit.intake.role_entries_env import (
    CATALOG,
    CATALOG_SQL,
    GROUP_BAND,
    GROUP_KEY,
    ITEM_BAND,
    ITEM_CONCEPT_CODE,
    ITEM_DISPLAY,
    KIN,
    KIN_SQL,
    LOOSE,
    NONE,
    PLAIN,
    REASON,
    TERMS,
    TERMS_SQL,
    abstaining,
    answering,
    claims,
    concept_list,
    denormalized,
    eligibility,
    entry,
    offered,
    output_models,
    pinned,
    prompts,
    resolved,
    roles_of,
    source_of,
)
from tests.unit.intake.structure_env import fresh_tools

pytestmark = pytest.mark.unit

CONCEPTS, CODE_LISTS = "concepts", "code_lists"
SECTION_MEMBERS = frozenset({CONCEPTS, CODE_LISTS})
CONCEPT_STATUS, VALUE_STATUS = f"{CATALOG}.status", f"{CATALOG}.value_sets.status"
STATUS_FIELDS = (role_field("status"), role_field("value_sets_status"))
SECTION_FIELD = role_field("section")


@pytest.fixture
def catalog(tmp_path: Path) -> Iterator[tuple[Any, Any, Any]]:
    con, source = source_of(CATALOG_SQL, "dn", tmp_path)
    structure = pinned(source, [denormalized(CATALOG, GROUP_KEY)], tmp_path / "pinned")
    yield source, structure, roles_of(source, structure, tmp_path / "roles")
    con.close()


def _terms(root: Path, sections: dict[str, str | None]) -> Iterator[tuple[Any, Any, Any]]:
    con, source = source_of(TERMS_SQL, "cc", root)
    bindings = [concept_list(relation, section) for relation, section in sections.items()]
    structure = pinned(source, bindings, root / "pinned")
    yield source, structure, roles_of(source, structure, root / "roles")
    con.close()


@pytest.fixture
def code_list(tmp_path: Path) -> Iterator[tuple[Any, Any, Any]]:
    yield from _terms(tmp_path, {TERMS: "code_lists.terms"})


@pytest.fixture
def kin(tmp_path: Path) -> Iterator[tuple[Any, Any, Any]]:
    con, source = source_of(KIN_SQL, "kn", tmp_path)
    structure = pinned(source, [concept_list(r, None) for r in KIN], tmp_path / "pinned")
    yield source, structure, roles_of(source, structure, tmp_path / "roles")
    con.close()


@pytest.fixture
def unsectioned(tmp_path: Path) -> Iterator[tuple[Any, Any, Any]]:
    yield from _terms(tmp_path, {LOOSE: None, PLAIN: None})


# --- F1: one field per open entry -------------------------------------------------------


def test_catalog_has_two_open_status_entries_with_different_members(
    catalog: tuple[Any, Any, Any],
) -> None:
    """The fixture holds the collision: one role, two open entries, two member sets."""
    _, _, roles = catalog
    concept = set(offered(entry(roles, CONCEPT_STATUS)))
    value = set(offered(entry(roles, VALUE_STATUS)))

    assert GROUP_BAND in concept and ITEM_BAND not in concept and NONE in concept
    assert ITEM_BAND in value and GROUP_BAND not in value and NONE in value
    assert entry(roles, CONCEPT_STATUS).choice is None
    assert entry(roles, VALUE_STATUS).choice is None


def _bands(field: str, members: tuple[str, ...]) -> str:
    return {STATUS_FIELDS[0]: GROUP_BAND, STATUS_FIELDS[1]: NONE}.get(field, ABSTAIN)


def test_repeated_role_gets_one_field_per_entry_over_its_own_members(
    catalog: tuple[Any, Any, Any], tmp_path: Path
) -> None:
    source, structure, roles = catalog
    client = answering(_bands)

    resolved(source, structure, roles, client, tmp_path)

    first = output_models(client)[0]
    assert set(STATUS_FIELDS) <= set(first.model_fields)
    lines = [lines_with(prompts(client)[0], f"`{field}`") for field in STATUS_FIELDS]
    assert all(len(found) == 1 for found in lines), lines


def test_each_repeated_role_entry_settles_from_its_own_answer(
    catalog: tuple[Any, Any, Any], tmp_path: Path
) -> None:
    source, structure, roles = catalog
    client = answering(_bands)

    result, _ = resolved(source, structure, roles, client, tmp_path)

    concept, value = entry(result, CONCEPT_STATUS), entry(result, VALUE_STATUS)
    assert (concept.choice, concept.source) == (GROUP_BAND, "model")
    assert (value.choice, value.source) == (NONE, "model")
    assert concept.call_key == value.call_key is not None
    assert eligibility(result, CATALOG, None, "status").choice == GROUP_BAND
    assert eligibility(result, CATALOG, "value_sets", "status").choice == NONE
    assert client.invoke.call_count == 1


def test_calls_without_a_repeated_role_settle_from_their_answers(tmp_path: Path) -> None:
    """A non-repeating call asks one map per signature and each answer settles its role."""
    asked = {}
    for name, tables in (("memory", MEMORY_TABLES), ("values", VALUE_TABLES)):
        built = Synthetic(tables, tmp_path / name)
        client = answering_roles(distinct({"display": "wim"}))
        result, _ = resolved(
            built.source, built.structure, built.roles, client, tmp_path / f"{name}_r"
        )
        asked[name] = [asked_columns(p) for p in sent_prompts(client)], result
        built.close()

    memory, values = asked["memory"], asked["values"]
    assert sorted(memory[0]) == [("tarn", "vell", "mox"), ("zorb", "quax")]
    assert values[0] == [("kelt", "wim", "rusk")]
    display = eligibility(values[1], "tq.eta", None, "display")
    assert (display.choice, display.source) == ("wim", "model")


# --- F2: claims in force are never offered ----------------------------------------------


def test_code_list_open_role_offers_no_column_a_deterministic_claim_holds(
    code_list: tuple[Any, Any, Any], tmp_path: Path
) -> None:
    source, structure, roles = code_list
    held = {column for (rel, _, _), column in claims(roles).items() if rel == TERMS}
    assert {ITEM_CONCEPT_CODE, ITEM_DISPLAY} <= held
    client = answering(abstaining)

    resolved(source, structure, roles, client, tmp_path)

    (prompt,) = prompts(client)
    claimed = prompt.partition("## Claims in force")[2].partition("## ")[0]
    for column in held:
        assert f"`{column}`" in claimed, column


def _claimed_first(held: set[str]) -> Any:
    """A required role takes a claimed column; an optional one answers ``none``."""

    def _pick(field: str, members: tuple[str, ...]) -> str:
        if NONE in members:
            return NONE
        return next((m for m in members if m in held), ABSTAIN)

    return _pick


def test_code_list_claims_survive_a_model_that_picks_a_claimed_column(
    code_list: tuple[Any, Any, Any], tmp_path: Path
) -> None:
    source, structure, roles = code_list
    before = claims(roles)
    held = {column for (rel, _, _), column in before.items() if rel == TERMS}

    result, _ = resolved(source, structure, roles, answering(_claimed_first(held)), tmp_path)

    for (relation, section, role), column in before.items():
        settled = eligibility(result, relation, section, role)
        assert settled.choice == column, (role, settled.choice, settled.source)
    assert eligibility(result, TERMS, None, "concept_code").choice == ITEM_CONCEPT_CODE
    assert eligibility(result, TERMS, None, "code").choice not in held


def test_model_choice_of_a_claimed_column_fails_one_role_per_column(
    code_list: tuple[Any, Any, Any], tmp_path: Path
) -> None:
    source, structure, _ = code_list
    stated = {f"{TERMS}.code": StatedChoice(choice=ITEM_CONCEPT_CODE, source="model",
                                             confidence=0.9)}

    roles = roles_of(source, structure, tmp_path / "stated", choices=stated)

    assert eligibility(roles, TERMS, None, "concept_code").choice == ITEM_CONCEPT_CODE
    tools = fresh_tools(source, tmp_path / "verify", stage="roles", structure=structure)
    verdict = verify_roles(roles, tools, spec=None, relations=(TERMS,))
    failing = [f for f in verdict.failures if f.check == ONE_ROLE_PER_COLUMN]
    assert [f.column for f in failing] == [ITEM_CONCEPT_CODE], verdict.failures


# --- F4: section entries reach the model ------------------------------------------------


def _section_or_abstain(choice: str) -> Any:
    def _pick(field: str, members: tuple[str, ...]) -> str:
        return choice if set(members) == SECTION_MEMBERS else ABSTAIN

    return _pick


def _section_fields(model: Any) -> list[str]:
    return [f for f in model.model_fields if f == SECTION_FIELD]


def test_open_section_entries_are_listed(unsectioned: tuple[Any, Any, Any]) -> None:
    _, _, roles = unsectioned

    for relation in (LOOSE, PLAIN):
        listed = entry(roles, f"{relation}.section")
        assert set(listed.members) == SECTION_MEMBERS
        assert listed.choice is None


def test_open_section_is_asked_with_its_relations_open_roles(
    unsectioned: tuple[Any, Any, Any], tmp_path: Path
) -> None:
    source, structure, roles = unsectioned
    client = answering(_section_or_abstain(CONCEPTS))

    resolved(source, structure, roles, client, tmp_path)

    loose_calls = [
        m for m, p in zip(output_models(client), prompts(client), strict=True)
        if "quax" in asked_columns(p)
    ]
    assert loose_calls, "the relation's open column roles were not asked"
    assert len(_section_fields(loose_calls[0])) == 1


def test_open_section_alone_is_still_asked(
    unsectioned: tuple[Any, Any, Any], tmp_path: Path
) -> None:
    source, structure, roles = unsectioned
    plain_open = [
        a for a in roles.ambiguities
        if a.subject.startswith(f"{PLAIN}.") and a.choice is None and a.source is None
    ]
    assert [a.subject for a in plain_open if offered(a)] == [f"{PLAIN}.section"]
    client = answering(_section_or_abstain(CONCEPTS))

    resolved(source, structure, roles, client, tmp_path)

    plain_calls = [
        m for m, p in zip(output_models(client), prompts(client), strict=True)
        if "plain_code" in asked_columns(p)
    ]
    assert len(plain_calls) == 1 and _section_fields(plain_calls[0])


def test_stated_section_choice_settles_the_binding(
    unsectioned: tuple[Any, Any, Any], tmp_path: Path
) -> None:
    source, structure, roles = unsectioned
    client = answering(_section_or_abstain(CONCEPTS))

    result, _ = resolved(source, structure, roles, client, tmp_path)

    for relation in (LOOSE, PLAIN):
        listed = entry(result, f"{relation}.section")
        assert (listed.choice, listed.source) == (CONCEPTS, "model"), relation
        assert listed.call_key is not None
        (bound,) = [e for e in result.structure.bound if relation in e.binding.relations]
        assert bound.binding.section == CONCEPTS


def test_abstained_section_stays_open_with_the_models_reason(
    unsectioned: tuple[Any, Any, Any], tmp_path: Path
) -> None:
    source, structure, roles = unsectioned

    result, _ = resolved(source, structure, roles, answering(abstaining), tmp_path)

    for relation in (LOOSE, PLAIN):
        listed = entry(result, f"{relation}.section")
        assert (listed.choice, listed.source) == (None, None)
        assert listed.reason == REASON, relation
        (bound,) = [e for e in result.structure.bound if relation in e.binding.relations]
        assert bound.binding.section is None


def test_sources_with_no_section_entry_send_no_section_field(
    code_list: tuple[Any, Any, Any], catalog: tuple[Any, Any, Any], tmp_path: Path
) -> None:
    """Guard: a relation whose section is stated never gains a section field."""
    for name, (source, structure, roles) in (("terms", code_list), ("catalog", catalog)):
        client = answering(abstaining)
        resolved(source, structure, roles, client, tmp_path / name)
        assert not [m for m in output_models(client) if _section_fields(m)], name


def _section_calls(client: Any) -> list[Any]:
    return [m for m in output_models(client) if _section_fields(m)]


def test_kin_relation_inherits_the_section_choice_through_memory(
    kin: tuple[Any, Any, Any], tmp_path: Path
) -> None:
    source, structure, roles = kin
    assert all(entry(roles, f"{r}.section").choice is None for r in KIN)
    client = answering(_section_or_abstain(CONCEPTS))

    result, _ = resolved(source, structure, roles, client, tmp_path)

    assert len(_section_calls(client)) == 1
    listed = [entry(result, f"{r}.section") for r in KIN]
    assert [a.choice for a in listed] == [CONCEPTS, CONCEPTS]
    assert sorted(str(a.source) for a in listed) == ["memory", "model"]
    for relation in KIN:
        (bound,) = [e for e in result.structure.bound if relation in e.binding.relations]
        assert bound.binding.section == CONCEPTS, relation


def test_kin_relations_both_stay_open_when_the_model_abstains(
    kin: tuple[Any, Any, Any], tmp_path: Path
) -> None:
    source, structure, roles = kin
    client = answering(abstaining)

    result, _ = resolved(source, structure, roles, client, tmp_path)

    assert len(_section_calls(client)) == 1
    for relation in KIN:
        listed = entry(result, f"{relation}.section")
        assert (listed.choice, listed.source) == (None, None), relation
        (bound,) = [e for e in result.structure.bound if relation in e.binding.relations]
        assert bound.binding.section is None

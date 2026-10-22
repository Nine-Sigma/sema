"""Every open role entry reaches the model on its own field, among unclaimed members.

Pinned contract (known-source misses F1, F2, F4, team lead 2026-10-21):

- F1. Two open entries for the same role on one relation (a denormalized relation's
  concept-side and value-set-side ``status``) each get their own enum field over their own
  members, and each entry reads its own answer. A call with no repeated role renders the
  same prompt under the same call key as before, so committed recordings still replay.
- F2. No open role offers a column a deterministic claim (``code`` or ``lexicon``) holds on
  the same relation, whatever the order the roles settle in; a model choice of such a
  column fails ``one_role_per_column`` and the claim stays in force.
- F4. An open ``<relation>.section`` entry is asked in its relation's roles call with its
  members as the enum, also when it is the relation's only open entry; a stated model
  choice settles the section through ``resolve_sections``; an abstention leaves it open
  with the model's reason.

The mocked model answers every enum field of the output model it is handed, so these tests
read members and answers, never field names.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from sema.intake.eligibility_utils import StatedChoice
from sema.intake.ledger_utils import prompt_hash
from sema.intake.output_models import ROLES_CHOICE
from sema.intake.output_models_utils import ABSTAIN
from sema.intake.verify.roles import ONE_ROLE_PER_COLUMN, verify_roles
from tests.unit.intake.model_calls_env import MODEL
from tests.unit.intake.resolve_roles_env import (
    MEMORY_TABLES,
    VALUE_TABLES,
    Synthetic,
    distinct,
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
    enum_fields,
    field_lines,
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
# Call keys the synthetic sources below produced at 7b48f3d, before any of these fixes.
MEMORY_KEYS = {
    "35302b5dbe3b52fea9d3e061f2e9c5b0583ba7dfc4d138ced5b4e9a2eb077a9d",
    "6ae566d7d831ec7b6dfaf9adb15c06b657b08c2833073901019b325b254a53c7",
}
VALUE_FIRST_KEY = "e666c60ccc16149a3046877a8b032e9bda008073ce4b791b7d9ec60d976e0092"


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

    assert set(offered(entry(roles, CONCEPT_STATUS))) == {GROUP_BAND, NONE}
    assert set(offered(entry(roles, VALUE_STATUS))) == {ITEM_BAND, NONE}
    assert entry(roles, CONCEPT_STATUS).choice is None
    assert entry(roles, VALUE_STATUS).choice is None


def _bands(field: str, members: tuple[str, ...]) -> str:
    if GROUP_BAND in members:
        return GROUP_BAND
    return NONE if ITEM_BAND in members else ABSTAIN


def test_repeated_role_gets_one_field_per_entry_over_its_own_members(
    catalog: tuple[Any, Any, Any], tmp_path: Path
) -> None:
    source, structure, roles = catalog
    client = answering(_bands)

    resolved(source, structure, roles, client, tmp_path)

    first = output_models(client)[0]
    enums = sorted(sorted(members) for members in enum_fields(first).values())
    assert enums == [sorted({GROUP_BAND, NONE}), sorted({ITEM_BAND, NONE})]
    lines = field_lines(prompts(client)[0])
    assert len(lines) == 2
    assert len(set(lines)) == 2, lines


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


def test_calls_without_a_repeated_role_keep_their_prompts_and_keys(tmp_path: Path) -> None:
    """Guard: the recorded replay keys of a non-repeating call do not move."""
    keys = {}
    for name, tables in (("memory", MEMORY_TABLES), ("values", VALUE_TABLES)):
        built = Synthetic(tables, tmp_path / name)
        client = answering_roles(distinct({"display": "wim"}))
        resolved(built.source, built.structure, built.roles, client, tmp_path / f"{name}_r")
        keys[name] = [prompt_hash(MODEL, p, ROLES_CHOICE) for p in sent_prompts(client)]
        built.close()

    assert set(keys["memory"]) == MEMORY_KEYS
    assert keys["values"][0] == VALUE_FIRST_KEY


# --- F2: claims in force are never offered ----------------------------------------------


def test_code_list_open_role_offers_no_column_a_deterministic_claim_holds(
    code_list: tuple[Any, Any, Any], tmp_path: Path
) -> None:
    source, structure, roles = code_list
    held = {column for (rel, _, _), column in claims(roles).items() if rel == TERMS}
    assert {ITEM_CONCEPT_CODE, ITEM_DISPLAY} <= held
    client = answering(abstaining)

    resolved(source, structure, roles, client, tmp_path)

    for model in output_models(client):
        for field, members in enum_fields(model).items():
            assert not held & set(members), (field, members)


def _claimed_first(held: set[str]) -> Any:
    def _pick(field: str, members: tuple[str, ...]) -> str:
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
    return [f for f, members in enum_fields(model).items() if set(members) == SECTION_MEMBERS]


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

    loose_calls = [m for m in output_models(client) if len(enum_fields(m)) > 1]
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

    sections_only = [m for m in output_models(client) if list(enum_fields(m)) == _section_fields(m)]
    assert len(sections_only) == 1


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

"""Identity roles ``title``, ``vocabulary_id``, ``concept_code`` (Dean, 2026-10-21).

Pinned contract:

- The lexicon declares each role with tokens and a description. They are role-map fields
  like any other (S5 decision 18a, reversing the 2026-10-21 name gate): every column is a
  member, plus ``none``; a name match is a lexicon fact; a constant column keeps its finding.
- ``title`` is a concept role (``ConceptRoles``); ``vocabulary_id`` and ``concept_code`` are
  code-list roles (``CodeListRoles``, so also ``ValueSetRoles``). All three are optional,
  and with no name match the model may name any column or ``none``.
- A name match settles the role with ``source: lexicon``; ``code`` and ``display`` still
  settle, and a lone label column named like a title stays the ``name``.
- A chosen column reaches the spec field the importer reads (``title`` seeds the concept's
  synonyms; ``vocabulary_id`` with ``concept_code`` gives each term an external ref);
  ``none`` reaches no field.
- A1, scoped to ``concepts`` and ``value_sets``, holds with only ``data_type``,
  ``sensitive``, and ``display_fallback`` trimmed from the written spec.
"""

from __future__ import annotations

import copy
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import duckdb
import pytest
import yaml

from sema.binding.importer import import_concept_source
from sema.binding.spec import ConceptSourceSpec
from sema.binding.spec_sections import CodeListRoles, ConceptRoles, ValueSetRoles
from sema.ingest.duckdb_staging import Staging
from sema.intake.roles import lexicon
from sema.intake.run.decisions import Decision, apply_structure_decisions
from sema.intake.roles.eligibility_utils import RunDescriptor, StatedChoice
from sema.intake.roles.lexicon import NO_REFERENCE
from sema.intake.land.loaders.database import land_database
from sema.intake.structure.shapes.base import SlotBinding
from sema.intake.structure.structure import BoundEntry
from tests.unit.intake.eligibility_env import by_role, descriptor, eligibility, write_hint
from tests.unit.intake.eligibility_env import run_roles as roles_over
from tests.unit.intake.structure.shapes.shapes_env import (
    FIXTURES,
    Source,
    staging_source,
    warehouse_source,
)
from tests.unit.intake.structure_env import fresh_tools, run

pytestmark = pytest.mark.unit

NONE = NO_REFERENCE
TITLE, VOCABULARY, CONCEPT_CODE = "title", "vocabulary_id", "concept_code"
IDENTITY_ROLES = (TITLE, VOCABULARY, CONCEPT_CODE)
CONCEPTS, VALUE_SETS = "concepts", "value_sets"
CODE_LIST_SECTION = "code_lists.terms"
HUMAN = "human"

SCHEMA = "tk"
TOPICS, TERMS = f"{SCHEMA}.topics", f"{SCHEMA}.terms"
CAPTIONS, LONE_TITLE = f"{SCHEMA}.captions", f"{SCHEMA}.lone_title"
BARE_CONCEPTS, BARE_TERMS = f"{SCHEMA}.bare_concepts", f"{SCHEMA}.bare_terms"
PELL = f"{SCHEMA}.pell_shape"
TWENTY = " FROM range(1, 21) r(i)"


def _padded(prefix: str) -> str:
    return f"'{prefix}' || lpad(CAST(i AS VARCHAR), 2, '0')"


SQL = (
    f"CREATE SCHEMA {SCHEMA}",
    # Both topic_label and topic_title carry name tokens, so name stays open here.
    f"CREATE TABLE {TOPICS} AS SELECT {_padded('T')} AS topic_id,"
    " 'Topic ' || CAST(i AS VARCHAR) AS topic_label,"
    " 'Heading for topic number ' || CAST(i AS VARCHAR) AS topic_title,"
    " 'A sentence that explains what topic ' || CAST(i AS VARCHAR) || ' covers'"
    f" AS topic_definition{TWENTY}",
    # term_vocabulary_id is constant; term_concept_code repeats (15 distinct of 20).
    f"CREATE TABLE {TERMS} AS SELECT {_padded('M')} AS term_code,"
    " 'Term entry ' || CAST(i AS VARCHAR) AS term_display,"
    " 'SYSA' AS term_vocabulary_id,"
    " 'K' || CAST(i % 15 AS VARCHAR) AS term_concept_code,"
    f" CASE WHEN i % 2 = 0 THEN 'A' ELSE 'B' END AS term_band{TWENTY}",
    # No column carries a title token: cap_caption is text, cap_rank an integer, cap_flag
    # a constant.
    f"CREATE TABLE {CAPTIONS} AS SELECT {_padded('C')} AS cap_id,"
    " 'Lbl ' || CAST(i AS VARCHAR) AS cap_label,"
    " 'Cap ' || CAST(i % 7 AS VARCHAR) AS cap_caption,"
    f" CAST(i AS INTEGER) AS cap_rank, 'X' AS cap_flag{TWENTY}",
    f"CREATE TABLE {LONE_TITLE} AS SELECT {_padded('E')} AS entry_id,"
    f" 'Entry ' || CAST(i AS VARCHAR) AS entry_title{TWENTY}",
    f"CREATE TABLE {BARE_CONCEPTS} AS SELECT {_padded('B')} AS bare_id,"
    f" 'Bare ' || CAST(i AS VARCHAR) AS bare_label{TWENTY}",
    f"CREATE TABLE {BARE_TERMS} AS SELECT {_padded('D')} AS bare_code,"
    f" 'Bare term ' || CAST(i AS VARCHAR) AS bare_display{TWENTY}",
    # The optional-roles fixture shape: `pell` is low-cardinality and carries no role word.
    f"CREATE TABLE {PELL} AS SELECT {_padded('K')} AS kelt,"
    " 'Wim entry ' || CAST(i AS VARCHAR) AS wim,"
    f" CASE WHEN i % 2 = 0 THEN 'B' ELSE 'A' END AS pell{TWENTY}",
)
TOPIC_NAME = "topic_label"
CHOSEN = {
    (TOPICS, TITLE): "topic_title",
    (TERMS, VOCABULARY): "term_vocabulary_id",
    (TERMS, CONCEPT_CODE): "term_concept_code",
}
DESCRIPTOR = RunDescriptor.model_validate({
    "id": "topics",
    "identity_system": "topics",
    "release": "1",
    "scheme": {"qualified_name": "topics.all", "description": "Synthetic topics"},
})


def _binding(relation: str, section: str | None) -> SlotBinding:
    return SlotBinding(
        template="concept_list", slots={"relation": relation}, relations=(relation,),
        section=section,
    )


@pytest.fixture(scope="module")
def source(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Source]:
    root = tmp_path_factory.mktemp("identity_roles")
    con = duckdb.connect(":memory:")
    for statement in SQL:
        con.execute(statement)
    yield warehouse_source(con, SCHEMA, root / "ledger" / "landing")
    con.close()


def _members(source: Source, relation: str) -> set[str]:
    (profile,) = [p for p in source.profiles() if p.ref.name == relation]
    return {*(column.name for column in profile.columns), NONE}


def _roles(source: Source, relation: str, section: str | None) -> dict[str, Any]:
    found = eligibility().compute(_binding(relation, section), source.profiles())
    return by_role(found, relation)


def _assembled(source: Source, root: Path, choices: Mapping[str, str]) -> Any:
    found = run(source, fresh_tools(source, root / "structure"))
    bound = tuple(
        BoundEntry(binding=_binding(relation, section), confidence=1.0, source=HUMAN)
        for relation, section in ((TOPICS, CONCEPTS), (TERMS, CODE_LIST_SECTION))
    )
    structure = found.model_copy(update={"bound": bound, "ambiguities": ()})
    stated = {
        subject: StatedChoice(choice=choice, source=HUMAN, confidence=1.0)
        for subject, choice in {f"{TOPICS}.name": TOPIC_NAME, **choices}.items()
    }
    result = roles_over(source, structure, root / "roles", choices=stated)
    spec = eligibility().assemble_spec(DESCRIPTOR, result)
    assert spec is not None, [(a.subject, a.members, a.choice) for a in result.ambiguities]
    return spec


@pytest.mark.parametrize("role", IDENTITY_ROLES)
def test_identity_role_is_declared_in_the_lexicon(role: str) -> None:
    declared = {entry.name: entry for entry in lexicon.ROLES}

    assert role in declared
    assert declared[role].tokens
    assert declared[role].description.strip()


@pytest.mark.parametrize(
    ("role", "carriers", "others"),
    [
        (TITLE, (ConceptRoles,), (CodeListRoles, ValueSetRoles)),
        (VOCABULARY, (CodeListRoles, ValueSetRoles), (ConceptRoles,)),
        (CONCEPT_CODE, (CodeListRoles, ValueSetRoles), (ConceptRoles,)),
    ],
)
def test_identity_role_is_optional_on_the_models_that_carry_it(
    role: str, carriers: tuple[type, ...], others: tuple[type, ...]
) -> None:
    for model in carriers:
        assert not model.model_fields[role].is_required(), model.__name__
    for model in others:
        assert role not in model.model_fields, model.__name__


def test_concept_targets_offer_title_and_code_list_targets_the_vocabulary_pair(
    source: Source,
) -> None:
    concept_roles = _roles(source, TOPICS, CONCEPTS)
    code_list_roles = _roles(source, TERMS, CODE_LIST_SECTION)
    open_roles = _roles(source, TERMS, None)

    assert TITLE in concept_roles
    assert not {VOCABULARY, CONCEPT_CODE} & set(concept_roles)
    assert {VOCABULARY, CONCEPT_CODE} <= set(code_list_roles)
    assert TITLE not in code_list_roles
    assert set(IDENTITY_ROLES) <= set(open_roles)
    for roles in (concept_roles, code_list_roles, open_roles):
        for name in set(IDENTITY_ROLES) & set(roles):
            assert roles[name].members[-1] == NONE, name


@pytest.mark.parametrize(
    ("relation", "section", "role"),
    [(TOPICS, CONCEPTS, TITLE), (TERMS, CODE_LIST_SECTION, VOCABULARY),
     (TERMS, CODE_LIST_SECTION, CONCEPT_CODE)],
)
def test_identity_role_members_are_every_column_and_the_name_match_is_a_fact(
    source: Source, relation: str, section: str, role: str
) -> None:
    """Columns matching no token (``topic_label``, ``term_band``, ``term_code``) stay in."""
    entry = _roles(source, relation, section)[role]

    assert set(entry.members) == _members(source, relation) and entry.members[-1] == NONE


@pytest.mark.parametrize(
    ("relation", "section", "role"),
    [(TOPICS, CONCEPTS, TITLE), (TERMS, CODE_LIST_SECTION, VOCABULARY),
     (TERMS, CODE_LIST_SECTION, CONCEPT_CODE)],
)
def test_a_lexicon_hint_settles_each_identity_role(
    source: Source, relation: str, section: str, role: str
) -> None:
    settled = _roles(source, relation, section)[role]
    column = CHOSEN[(relation, role)]

    assert (settled.choice, settled.source) == (column, "lexicon")
    assert any(h.source == "lexicon" and h.matches == (column,) for h in settled.hints)


def test_earlier_roles_keep_their_columns(source: Source) -> None:
    terms = _roles(source, TERMS, CODE_LIST_SECTION)
    lone = _roles(source, LONE_TITLE, CONCEPTS)

    assert (terms["code"].choice, terms["display"].choice) == ("term_code", "term_display")
    assert lone["name"].choice == "entry_title"


@pytest.mark.parametrize(
    ("relation", "section", "names", "claimed"),
    [(BARE_CONCEPTS, CONCEPTS, (TITLE,), True), (CAPTIONS, CONCEPTS, (TITLE,), False),
     (BARE_TERMS, CODE_LIST_SECTION, (VOCABULARY, CONCEPT_CODE), True),
     (PELL, CODE_LIST_SECTION, (VOCABULARY, CONCEPT_CODE), False)],
)
def test_a_relation_without_identity_names_asks_them_unless_every_column_is_claimed(
    source: Source, relation: str, section: str, names: tuple[str, ...], claimed: bool
) -> None:
    """A free column leaves the role open; with every column claimed it settles ``none``."""
    roles = _roles(source, relation, section)
    listed = {a.subject for a in eligibility().ambiguities(roles.values())}

    for name in names:
        assert set(roles[name].members) == _members(source, relation), name
        settled = (roles[name].choice, roles[name].source)
        assert settled == ((NONE, "code") if claimed else (None, None)), name
        assert (f"{relation}.{name}" in listed) is not claimed, name


def test_the_optional_roles_shape_opens_its_identity_roles(source: Source) -> None:
    roles = _roles(source, PELL, CODE_LIST_SECTION)
    listed = eligibility().ambiguities(roles.values())
    opened = ("code", "display", "status", VOCABULARY, CONCEPT_CODE)

    assert set(IDENTITY_ROLES) - {TITLE} <= set(roles)
    open_subjects = {a.subject for a in listed if a.choice is None}
    assert {f"{PELL}.{role}" for role in opened} <= open_subjects


def test_chosen_identity_columns_reach_the_spec_and_the_import(
    source: Source, tmp_path: Path
) -> None:
    spec = _assembled(source, tmp_path, {})

    assert spec.concepts.roles.title == CHOSEN[(TOPICS, TITLE)]
    (listed,) = spec.code_lists
    assert listed.roles.vocabulary_id == CHOSEN[(TERMS, VOCABULARY)]
    assert listed.roles.concept_code == CHOSEN[(TERMS, CONCEPT_CODE)]
    manifest = import_concept_source(spec, source.tools.provider, dataset_map={}).manifest
    first_term = manifest["terms"][0]
    assert first_term["external_refs"] == [{"system": "SYSA", "id": "K1"}]
    (scheme,) = manifest["entities"]
    assert scheme["properties"][0]["synonyms"] == ["Heading for topic number 1"]


def test_a_none_choice_reaches_no_spec_field(source: Source, tmp_path: Path) -> None:
    nones = {f"{relation}.{role}": NONE for relation, role in CHOSEN}

    spec = _assembled(source, tmp_path, nones)

    assert spec.concepts.roles.title is None
    (listed,) = spec.code_lists
    assert (listed.roles.vocabulary_id, listed.roles.concept_code) == (None, None)
    for roles in (spec.concepts.roles, listed.roles):
        assert NONE not in roles.model_dump(exclude_none=True).values()
    manifest = import_concept_source(spec, source.tools.provider, dataset_map={}).manifest
    assert all(term["external_refs"] == [] for term in manifest["terms"])


# A1 with the identity roles computed: only the roles no lexicon entry declares are trimmed
# from the written spec (the denormalized assembly test trims the identity roles too).
DENORMALIZED = "concept_with_values_denormalized"
LIBRARY = FIXTURES / "partner_sl"
LIBRARY_DB = LIBRARY / "library.duckdb"
LIBRARY_SPEC = LIBRARY / "standard_library.spec.yaml"
LIBRARY_RUN = LIBRARY / "standard_library.run.yaml"
SPEC_HEAD = ("spec_version", "id", "identity_system", "release", "default_provider", "scheme")
SIDES = (CONCEPTS, VALUE_SETS)
UNCOMPUTABLE = {CONCEPTS: ("data_type", "sensitive"), VALUE_SETS: ("display_fallback",)}
SIDE_IDENTITY = {CONCEPTS: (TITLE,), VALUE_SETS: (VOCABULARY, CONCEPT_CODE)}


@pytest.fixture(scope="module")
def library(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Source]:
    root = tmp_path_factory.mktemp("a1_identity")
    staging = Staging(str(root / "staging.duckdb"), schemas=())
    landed = land_database(LIBRARY_DB, staging, schema="land_a1")
    yield staging_source(staging, landed, root / "ledger" / "landing")
    staging.close()


def _written() -> dict[str, Any]:
    raw: dict[str, Any] = yaml.safe_load(LIBRARY_SPEC.read_text())["concept_source"]
    for side, names in UNCOMPUTABLE.items():
        for name in names:
            raw[side]["roles"].pop(name)
    return {key: raw[key] for key in (*SPEC_HEAD, *SIDES)}


def _shape_hint(directory: Path) -> ConceptSourceSpec:
    """The scoped A1 hint, also leaving out the concept side's identity roles."""
    raw = copy.deepcopy(_written())
    raw.pop(VALUE_SETS)
    for name in SIDE_IDENTITY[CONCEPTS]:
        raw[CONCEPTS]["roles"].pop(name)
    raw["shape"] = [{"section": CONCEPTS, "template": DENORMALIZED}]
    return write_hint(directory, {"concept_source": raw})


def _side_role(subject: str, relation: str) -> tuple[str, str]:
    parts = subject.removeprefix(f"{relation}.").split(".")
    side = VALUE_SETS if VALUE_SETS in parts else CONCEPTS
    (role,) = [part for part in parts if part not in SIDES]
    return side, role


def _oracle_choices(listed: Any, relation: str) -> dict[str, StatedChoice]:
    oracle = {side: dict(_written()[side]["roles"]) for side in SIDES}
    choices = {}
    for ambiguity in listed.ambiguities:
        if not ambiguity.subject.startswith(f"{relation}."):
            continue
        side, role = _side_role(ambiguity.subject, relation)
        choice = oracle[side].get(role, NONE)
        assert choice in ambiguity.members, (ambiguity.subject, choice, ambiguity.members)
        choices[ambiguity.subject] = StatedChoice(choice=choice, source=HUMAN, confidence=1.0)
    return choices


def _manifest(spec: ConceptSourceSpec, source: Source) -> str:
    imported = import_concept_source(spec, source.tools.provider, dataset_map={})
    return yaml.safe_dump(imported.manifest, sort_keys=False)


def test_a1_imports_equal_with_only_uncomputable_roles_trimmed(
    library: Source, tmp_path: Path
) -> None:
    hinted = run(library, fresh_tools(library, tmp_path / "structure"), _shape_hint(tmp_path))
    ignores = [Decision(stage="structure", subject=a.subject, choice="ignore")
               for a in hinted.ambiguities]
    tools = fresh_tools(library, tmp_path / "decided")
    structure = apply_structure_decisions(
        hinted, ignores, library.land, tools, provider=library.provider
    )
    (entry,) = [e for e in structure.bound if e.binding.template == DENORMALIZED]
    relation = entry.binding.slots["relation"]
    listed = roles_over(library, structure, tmp_path / "listed")
    identity = [
        e for e in listed.eligibilities if e.relation == relation and e.role in IDENTITY_ROLES
    ]
    assert sorted(e.role for e in identity) == sorted(IDENTITY_ROLES)
    for eligible in identity:
        assert eligible.source == "lexicon", (eligible.role, eligible.members)

    roles = roles_over(
        library, structure, tmp_path / "stated", choices=_oracle_choices(listed, relation)
    )
    spec = eligibility().assemble_spec(descriptor(LIBRARY_RUN), roles)

    assert spec is not None, [(a.subject, a.members) for a in roles.ambiguities]
    written = ConceptSourceSpec.model_validate(_written())
    assert spec.concepts is not None and written.concepts is not None
    assert spec.value_sets is not None and written.value_sets is not None
    assert spec.concepts.roles.title == written.concepts.roles.title
    for name in SIDE_IDENTITY[VALUE_SETS]:
        assert getattr(spec.value_sets.roles, name) == getattr(written.value_sets.roles, name)
    scoped = spec.model_copy(update={"shape": None})
    assert _manifest(scoped, library) == _manifest(written, library)

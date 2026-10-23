"""A bound concept_with_values_denormalized assembles a spec (Dean, 2026-10-21).

Pinned contract:

- ``eligibility.role_targets`` gives a denormalized binding two targets over its one
  relation: ``(ConceptRoles,)`` and ``(ValueSetRoles,)``, each slotting the key role to
  ``group_by``.
- Concept roles take the group's own attributes: columns ``group_by`` functionally
  determines. Value-set roles take the rows: columns it does not determine. A constant
  column is determined by every key, so either side may take it.
- The two sides stay apart: ``eligibility_utils.subject_of`` of a value-set role over the
  relation carries the ``value_sets`` section token, so no two entries share a subject.
- ``assemble`` emits ``concepts`` (``group_by`` as the key) and ``value_sets`` over the
  same relation, the shape the hand-written standard-library spec uses.
- A1, scoped by Dean to ``concepts`` and ``value_sets``: the library fixture, with the
  denormalized binding stated by a shape-only hint and role choices stated only where
  Roles lists an entry, assembles a spec whose import of those sections equals the
  hand-written spec's, trimmed to the roles eligibility computes (see ``UNCOMPUTED_ROLES``).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest
import yaml

from sema.binding.importer import import_concept_source
from sema.binding.spec import ConceptSourceSpec
from sema.ingest.duckdb_staging import Staging
from sema.intake.decisions import Decision, apply_structure_decisions
from sema.intake.eligibility_utils import RunDescriptor, StatedChoice, subject_of
from sema.intake.lexicon import KEY_ROLE, NO_REFERENCE
from sema.intake.loaders.database import land_database
from sema.intake.shapes.base import SlotBinding
from sema.intake.shapes.precedence import member
from tests.unit.intake.eligibility_env import (
    descriptor,
    eligibility,
    memory_source,
    run_roles,
    write_hint,
)
from tests.unit.intake.shapes.shapes_env import FIXTURES, Source, staging_source
from tests.unit.intake.structure_env import fresh_tools, run

pytestmark = pytest.mark.unit

ID = "concept_with_values_denormalized"
CONCEPTS, VALUE_SETS = "concepts", "value_sets"
SIDES = (CONCEPTS, VALUE_SETS)
STRUCTURE, ROLES, IGNORE = "structure", "roles", "ignore"

CATALOG = "dn.catalog"
GROUP_KEY = "group_key"
GROUP_ATTRIBUTES = {"group_label", "group_description"}
ROW_COLUMNS = {"item_code", "item_label"}
# Three groups of three rows; every group attribute is a function of group_key and the item
# columns are unique, so concept_list and the denormalized template are both eligible.
CATALOG_SQL = (
    "CREATE SCHEMA dn",
    "CREATE TABLE dn.catalog AS SELECT * FROM (VALUES"
    " ('G1', 'Colour', 'The colour a surface shows under daylight', 'C01', 'Crimson'),"
    " ('G1', 'Colour', 'The colour a surface shows under daylight', 'C02', 'Teal'),"
    " ('G1', 'Colour', 'The colour a surface shows under daylight', 'C03', 'Ochre'),"
    " ('G2', 'Size', 'The overall extent of an object in its longest axis', 'S01', 'Petite'),"
    " ('G2', 'Size', 'The overall extent of an object in its longest axis', 'S02', 'Medium'),"
    " ('G2', 'Size', 'The overall extent of an object in its longest axis', 'S03', 'Grand'),"
    " ('G3', 'Shape', 'The outline an object traces when seen from above', 'H01', 'Round'),"
    " ('G3', 'Shape', 'The outline an object traces when seen from above', 'H02', 'Square'),"
    " ('G3', 'Shape', 'The outline an object traces when seen from above', 'H03', 'Oval'))"
    " t(group_key, group_label, group_description, item_code, item_label)",
)
CATALOG_ORACLE = {
    CONCEPTS: {"name": "group_label", "definition": "group_description"},
    VALUE_SETS: {"code": "item_code", "display": "item_label"},
}
CATALOG_DESCRIPTOR = RunDescriptor.model_validate({
    "id": "catalog",
    "identity_system": "catalog",
    "release": "1",
    "scheme": {"qualified_name": "catalog.groups", "description": "Grouped item codes"},
})
SCHEMELESS_DESCRIPTOR = RunDescriptor(id="catalog", identity_system="grouped", release="2026.1")

LIBRARY = FIXTURES / "partner_sl"
LIBRARY_DB = LIBRARY / "library.duckdb"
LIBRARY_SPEC = LIBRARY / "standard_library.spec.yaml"
LIBRARY_RUN = LIBRARY / "standard_library.run.yaml"
SCOPED_SECTIONS = (CONCEPTS, VALUE_SETS)
SPEC_HEAD = ("spec_version", "id", "identity_system", "release", "default_provider", "scheme")


def _denormalized(relation: str, group_by: str) -> SlotBinding:
    return SlotBinding(
        template=ID,
        section=CONCEPTS,
        slots={"relation": relation, "group_by": group_by},
        relations=(relation,),
    )


def _side_role(subject: str, relation: str) -> tuple[str, str]:
    parts = subject.removeprefix(f"{relation}.").split(".")
    side = VALUE_SETS if VALUE_SETS in parts else CONCEPTS
    (role,) = [part for part in parts if part not in SIDES]
    return side, role


def _side_of(entry: Any) -> str:
    return _side_role(subject_of(entry), entry.relation)[0]


def _stated(roles: Any, relation: str, oracle: Mapping[str, Mapping[str, str]]) -> dict[str, Any]:
    """The oracle's column for each listed entry, ``none`` where the oracle names none."""
    choices = {}
    for ambiguity in roles.ambiguities:
        if not ambiguity.subject.startswith(f"{relation}."):
            continue
        side, role = _side_role(ambiguity.subject, relation)
        choice = oracle[side].get(role, NO_REFERENCE)
        assert choice in ambiguity.members, (ambiguity.subject, choice, ambiguity.members)
        choices[ambiguity.subject] = StatedChoice(choice=choice, source="human", confidence=1.0)
    return choices


def _decided(source: Source, structure: Any, decisions: list[Decision], ledger: Path) -> Any:
    tools = fresh_tools(source, ledger)
    return apply_structure_decisions(
        structure, decisions, source.land, tools, provider=source.provider
    )


def _assembled(
    source: Source,
    structure: Any,
    root: Path,
    oracle: Mapping[str, Mapping[str, str]],
    run_descriptor: RunDescriptor,
    relation: str,
) -> tuple[Any, ConceptSourceSpec | None]:
    listed = run_roles(source, structure, root / "listed")
    choices = _stated(listed, relation, oracle)
    roles = run_roles(source, structure, root / "stated", choices=choices)
    return roles, eligibility().assemble_spec(run_descriptor, roles)


@pytest.fixture
def catalog(tmp_path: Path) -> Iterator[Source]:
    con, source = memory_source(CATALOG_SQL, "dn", tmp_path)
    yield source
    con.close()


@pytest.fixture
def catalog_structure(catalog: Source, tmp_path: Path) -> Any:
    structured = run(catalog, fresh_tools(catalog, tmp_path / "structure"))
    choice = member(_denormalized(CATALOG, GROUP_KEY))
    decision = Decision(stage=STRUCTURE, subject=CATALOG, choice=choice)
    return _decided(catalog, structured, [decision], tmp_path / "decided")


def test_role_targets_give_concepts_and_value_sets_keyed_on_group_by() -> None:
    targets = eligibility().role_targets(_denormalized("s.r", "g"))

    assert sorted(t.models[0].__name__ for t in targets) == ["ConceptRoles", "ValueSetRoles"]
    for target in targets:
        assert (target.template, target.relation) == (ID, "s.r")
        assert len(target.models) == 1
        assert target.slotted == {KEY_ROLE: "g"}


def test_denormalized_has_an_assembly_rule() -> None:
    from sema.intake.eligibility_spec_utils import BUILDERS

    assert ID in BUILDERS


def test_catalog_binds_the_stated_denormalized_binding(catalog_structure: Any) -> None:
    (entry,) = [e for e in catalog_structure.bound if CATALOG in e.binding.relations]

    assert entry.binding.template == ID
    assert entry.binding.slots["group_by"] == GROUP_KEY
    assert catalog_structure.ambiguities == ()


def test_concept_roles_take_group_attributes_and_value_roles_take_rows(
    catalog: Source, catalog_structure: Any, tmp_path: Path
) -> None:
    roles = run_roles(catalog, catalog_structure, tmp_path / "roles")
    entries = [e for e in roles.eligibilities if e.relation == CATALOG and e.subject is None]
    sides = {_side_of(e) for e in entries}

    assert sides == set(SIDES)
    for entry in entries:
        columns = set(entry.members) - {NO_REFERENCE}
        if entry.role == KEY_ROLE:
            assert columns == {GROUP_KEY}, subject_of(entry)
        elif _side_of(entry) == CONCEPTS:
            assert not columns & ROW_COLUMNS, subject_of(entry)
        else:
            assert not columns & GROUP_ATTRIBUTES, subject_of(entry)


def test_concept_and_value_set_roles_have_distinct_subjects(
    catalog: Source, catalog_structure: Any, tmp_path: Path
) -> None:
    roles = run_roles(catalog, catalog_structure, tmp_path / "roles")
    subjects = [a.subject for a in roles.ambiguities if a.stage == ROLES]
    keyed = [e for e in roles.eligibilities if e.relation == CATALOG and e.role != KEY_ROLE]
    unkeyed = [subject_of(e) for e in keyed if e.subject is None]

    assert len(subjects) == len(set(subjects))
    assert len(unkeyed) == len(set(unkeyed))
    value_roles = {e.role for e in keyed if e.subject is None and _side_of(e) == VALUE_SETS}
    assert {"code", "display"} <= value_roles


def test_catalog_assembles_concepts_and_value_sets_over_one_relation(
    catalog: Source, catalog_structure: Any, tmp_path: Path
) -> None:
    roles, spec = _assembled(
        catalog, catalog_structure, tmp_path, CATALOG_ORACLE, CATALOG_DESCRIPTOR, CATALOG
    )

    assert spec is not None, [(a.subject, a.members) for a in roles.ambiguities]
    assert spec.concepts is not None and spec.value_sets is not None
    assert spec.concepts.relation.name == spec.value_sets.relation.name == CATALOG
    assert spec.concepts.group_by == GROUP_KEY
    concept_roles, value_roles = spec.concepts.roles, spec.value_sets.roles
    assert getattr(concept_roles, KEY_ROLE) == getattr(value_roles, KEY_ROLE) == GROUP_KEY
    assert (concept_roles.name, concept_roles.definition) == ("group_label", "group_description")
    assert (value_roles.code, value_roles.display) == ("item_code", "item_label")
    assert [(e.section, e.template) for e in spec.shape or []] == [(CONCEPTS, ID)]


def test_catalog_without_a_stated_scheme_derives_one_from_the_descriptor(
    catalog: Source, catalog_structure: Any, tmp_path: Path
) -> None:
    run_descriptor = SCHEMELESS_DESCRIPTOR
    roles, spec = _assembled(
        catalog, catalog_structure, tmp_path, CATALOG_ORACLE, run_descriptor, CATALOG
    )

    assert spec is not None, [(a.subject, a.members) for a in roles.ambiguities]
    assert spec.scheme is not None
    assert spec.scheme.qualified_name == f"{run_descriptor.identity_system}.{run_descriptor.id}"
    assert run_descriptor.id in spec.scheme.description
    assert run_descriptor.release in spec.scheme.description


def test_a_stated_scheme_wins_unchanged(
    catalog: Source, catalog_structure: Any, tmp_path: Path
) -> None:
    _, spec = _assembled(
        catalog, catalog_structure, tmp_path, CATALOG_ORACLE, CATALOG_DESCRIPTOR, CATALOG
    )

    assert spec is not None
    assert spec.scheme == CATALOG_DESCRIPTOR.scheme


def test_catalog_spec_imports_one_value_set_per_group(
    catalog: Source, catalog_structure: Any, tmp_path: Path
) -> None:
    _, spec = _assembled(
        catalog, catalog_structure, tmp_path, CATALOG_ORACLE, CATALOG_DESCRIPTOR, CATALOG
    )
    assert spec is not None

    manifest = import_concept_source(spec, catalog.tools.provider, dataset_map={}).manifest

    assert len(manifest["vocabularies"]) == 3
    assert len(manifest["terms"]) == 9
    (scheme,) = manifest["entities"]
    assert {p["concept_ref"]["id"] for p in scheme["properties"]} == {"G1", "G2", "G3"}


# A1, scoped to concepts and value_sets (Dean, 2026-10-21), and narrowed (team lead,
# 2026-10-21) to the roles eligibility computes today, so the test is red only for the
# denormalized assembly gap. `lexicon.ROLES` holds none of the roles below; measured by
# importing the scoped written spec with each one removed:
#   concepts.title                -> manifest changes (title seeds concept synonyms)
#   concepts.data_type            -> no change (the column holds the default type)
#   concepts.sensitive            -> no change (the importer does not read it)
#   value_sets.display_fallback   -> no change (display is never null)
#   value_sets.vocabulary_id      -> manifest changes (term external refs)
#   value_sets.concept_code       -> manifest changes (term external refs)
# The three manifest-changing roles are a role-eligibility gap routed to Dean.
UNCOMPUTED_ROLES = {
    CONCEPTS: ("title", "data_type", "sensitive"),
    VALUE_SETS: ("display_fallback", "vocabulary_id", "concept_code"),
}


@pytest.fixture(scope="module")
def library(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[Staging, Source]]:
    root = tmp_path_factory.mktemp("a1")
    staging = Staging(str(root / "staging.duckdb"), schemas=())
    landed = land_database(LIBRARY_DB, staging, schema="land_a1")
    yield staging, staging_source(staging, landed, root / "ledger" / "landing")
    staging.close()


def _library_raw() -> dict[str, Any]:
    """The written spec with the roles eligibility never computes removed."""
    raw: dict[str, Any] = yaml.safe_load(LIBRARY_SPEC.read_text())["concept_source"]
    for side, names in UNCOMPUTED_ROLES.items():
        for name in names:
            raw[side]["roles"].pop(name)
    return raw


def _scoped(raw: Mapping[str, Any]) -> dict[str, Any]:
    return {key: raw[key] for key in (*SPEC_HEAD, *SCOPED_SECTIONS)}


def _shape_only_hint(directory: Path) -> ConceptSourceSpec:
    """Shape-only hint: it states the denormalized binding and nothing Roles reads.

    With no hint, precedence drops the denormalized member over the data element relation
    (multi-relation candidates claim it), so a structure decision cannot choose it.
    """
    raw = _scoped(_library_raw())
    raw.pop(VALUE_SETS)
    raw["shape"] = [{"section": CONCEPTS, "template": ID}]
    return write_hint(directory, {"concept_source": raw})


def _library_oracle() -> dict[str, dict[str, str]]:
    raw = _library_raw()
    return {side: dict(raw[side]["roles"]) for side in SCOPED_SECTIONS}


def _manifest(spec: ConceptSourceSpec, source: Source) -> str:
    imported = import_concept_source(spec, source.tools.provider, dataset_map={})
    return yaml.safe_dump(imported.manifest, sort_keys=False)


def test_a1_assembles_concepts_and_value_sets_that_import_equal_to_the_written_spec(
    library: tuple[Staging, Source], tmp_path: Path
) -> None:
    _, source = library
    hinted = run(source, fresh_tools(source, tmp_path / "structure"), _shape_only_hint(tmp_path))
    decisions = [
        Decision(stage=STRUCTURE, subject=a.subject, choice=IGNORE) for a in hinted.ambiguities
    ]
    structure = _decided(source, hinted, decisions, tmp_path / "decided")
    (entry,) = [e for e in structure.bound if e.binding.template == ID]
    relation = entry.binding.slots["relation"]

    roles, spec = _assembled(
        source, structure, tmp_path, _library_oracle(), descriptor(LIBRARY_RUN), relation
    )

    assert spec is not None, [(a.subject, a.members) for a in roles.ambiguities]
    written = ConceptSourceSpec.model_validate(_scoped(_library_raw()))
    scoped = spec.model_copy(update={"shape": None})
    assert spec.concepts is not None and written.concepts is not None
    assert spec.concepts.group_by == written.concepts.group_by
    assert _manifest(scoped, source) == _manifest(written, source)

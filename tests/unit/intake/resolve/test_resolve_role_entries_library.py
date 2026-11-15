"""F1, F2, F4 on the committed standard-library fixture (A1), offline.

The structure is stated as the recorded A1 run's model settled it (run 12bca208): the data
element relation as a denormalized concept list grouped by the written spec's key, the
concept and mapping relations as concept lists with no section, the other two ignored.
Roles then lists two open ``status`` entries on the data element relation (one per side)
and an open ``.section`` entry for each concept list. The mocked model answers every field
it is handed, so nothing here depends on how a field is named beyond its role suffix.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

from sema.ingest.duckdb_staging import Staging
from sema.intake.land.loaders.database import land_database
from sema.intake.resolve.output_models import ROLE_MAP_PREFIX
from sema.intake.resolve.output_models_utils import ABSTAIN
from tests.unit.intake.role_entries_env import (
    REASON,
    abstaining,
    answering,
    claims,
    concept_list,
    decided,
    denormalized,
    eligibility,
    entry,
    member_of,
    offered,
    open_column_entries,
    output_models,
    resolved,
    prompts,
    roles_of,
)
from tests.unit.intake.resolve_roles_env import map_members
from tests.unit.intake.structure.shapes.shapes_env import FIXTURES, Source, staging_source
from tests.unit.intake.structure_env import fresh_tools, run

pytestmark = pytest.mark.unit

LIBRARY = FIXTURES / "partner_sl"
SCHEMA = "standard_library"
DATA_ELEMENT = f"{SCHEMA}.standard_library_data_element"
CONCEPT_LISTS = (f"{SCHEMA}.standard_library_concept", f"{SCHEMA}.data_element_mappings")
IGNORE = "ignore"
Library = tuple[Source, Any, Any]


def _group_by() -> str:
    raw = yaml.safe_load((LIBRARY / "standard_library.spec.yaml").read_text())
    return str(raw["concept_source"]["concepts"]["group_by"])


def _recorded_choice(ambiguity: Any) -> str:
    relation = ambiguity.subject
    if relation == DATA_ELEMENT:
        return member_of(denormalized(relation, _group_by()))
    if relation in CONCEPT_LISTS:
        return member_of(concept_list(relation, None))
    return IGNORE


@pytest.fixture(scope="module")
def library(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Library]:
    root = tmp_path_factory.mktemp("library_roles")
    staging = Staging(str(root / "staging.duckdb"), schemas=())
    landed = land_database(LIBRARY / "library.duckdb", staging, schema="land_a1")
    source = staging_source(staging, landed, root / "ledger" / "landing")
    found = run(source, fresh_tools(source, root / "structure"))
    structure = decided(source, found, _recorded_choice, root / "decided")
    yield source, structure, roles_of(source, structure, root / "roles")
    staging.close()


def _by_relation(roles: Any, structure: Any) -> dict[str, list[Any]]:
    relations = {name for e in structure.bound for name in e.binding.relations}
    grouped: dict[str, list[Any]] = {}
    for ambiguity in open_column_entries(roles):
        (relation,) = [r for r in relations if ambiguity.subject.startswith(f"{r}.")]
        grouped.setdefault(relation, []).append(ambiguity)
    return grouped


def test_library_lists_the_colliding_and_section_entries(library: Library) -> None:
    """The fixture still holds the three recorded misses before any model call."""
    _, structure, roles = library
    grouped = _by_relation(roles, structure)

    statuses = [a.subject for a in grouped[DATA_ELEMENT] if a.subject.endswith(".status")]
    assert len(statuses) == 2
    for relation in CONCEPT_LISTS:
        assert f"{relation}.section" in {a.subject for a in grouped[relation]}


def test_library_open_entries_each_get_a_map_field_offering_their_own_members(
    library: Library, tmp_path: Path
) -> None:
    source, structure, roles = library
    client = answering(abstaining)

    resolved(source, structure, roles, client, tmp_path)

    sent = [
        map_members(prompt)
        for prompt, model in zip(prompts(client), output_models(client), strict=True)
        if model.__name__.startswith(ROLE_MAP_PREFIX)
    ]
    for relation, entries in _by_relation(roles, structure).items():
        for listed in entries:
            role = listed.subject.rpartition(".")[2]
            assert any(
                field.endswith(f"_{role}") and set(offered(listed)) <= set(members)
                for fields in sent for field, members in fields.items()
            ), (relation, role, offered(listed))


def test_library_abstained_entries_stay_open_with_the_models_reason(
    library: Library, tmp_path: Path
) -> None:
    source, structure, roles = library

    result, _ = resolved(source, structure, roles, answering(abstaining), tmp_path)

    for listed in open_column_entries(roles):
        left = entry(result, listed.subject)
        assert (left.choice, left.reason) == (None, REASON), listed.subject


def test_library_claims_survive_a_model_that_picks_claimed_columns(
    library: Library, tmp_path: Path
) -> None:
    source, structure, roles = library
    before = claims(roles)
    held = set(before.values())

    def _claimed_first(field: str, members: tuple[str, ...]) -> str:
        return next((m for m in members if m in held), ABSTAIN)

    result, _ = resolved(source, structure, roles, answering(_claimed_first), tmp_path)

    moved = {
        key: eligibility(result, *key).choice
        for key, column in before.items()
        if eligibility(result, *key).choice != column
    }
    assert moved == {}, moved

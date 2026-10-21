"""Role eligibility: predicate sets, ordered hints, exclusion, mentions, spec assembly."""

from __future__ import annotations

import importlib.util
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

from sema.binding.importer import import_concept_source
from sema.binding.spec import load_concept_source_spec
from sema.binding.spec_sections import RelationRef
from sema.intake.ledger import Ledger
from sema.intake.profile import EvidencePolicy, ProfileRefusedError, ProfileTools
from tests.unit.intake.eligibility_env import (
    A2_GOLDEN, A2_RUN, A8_BUILDER, A8_CSV, ACRIS_RUN, COLOURS_SQL, CONCEPT_LIST,
    DECLARED_JOINS, GROUPED_CODE_TABLE, JOIN_COLUMNS, LEGALS_DESCRIPTION, MISSPELLED_PROPERTY,
    PROMPT_KEY_START, REAL_MASTER, REFERENCES, SECTION_MEMBERS,
    SPEC_TOKENS, Env, by_role, clear_embedder,
    close_embedder, concept_list_over, cosine, descriptor, eligibility, eligibility_utils,
    hint_sources, lexicon, memory_source, model_calls, normalized, per_property, run_roles,
)
from tests.unit.intake.structure_env import (
    ACRIS_SPEC, CODE_TABLES, fresh_tools, ref, run, sheet, sql_lines,
)

pytestmark = pytest.mark.unit

FLOORS = ("DISPLAY_NULL_RATE_FLOOR", "DEFINITION_MEAN_LENGTH_FLOOR", "STATUS_CARDINALITY_CAP")
FLOOR = 0.2


@pytest.fixture(scope="module")
def env(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Env]:
    held = Env(tmp_path_factory.mktemp("eligibility"))
    yield held
    held.close()


def _a8(env: Env, **kwargs: Any) -> Any:
    (profile,) = env.a8.source.profiles()
    binding = concept_list_over(profile.ref.name)
    return eligibility().compute(binding, env.a8.source.profiles(), **kwargs)


def _ranked(env: Env, **kwargs: Any) -> Any:
    binding = concept_list_over(env.ranked.ref("ranked"))
    return eligibility().compute(binding, env.ranked.profiles(), **kwargs)


def _embedding(entry: Any) -> Any:
    (hint,) = [hint for hint in entry.hints if hint.source == "embedding"]
    return hint


def test_a8_fixture_is_the_generator_output() -> None:
    spec = importlib.util.spec_from_file_location("build_ambiguity_a8", A8_BUILDER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.build() == A8_CSV.read_bytes()


def test_a8_lands_as_one_concept_list_with_its_section_open(env: Env) -> None:
    (entry,) = env.structured(env.a8, "a8_structure").bound

    assert (entry.binding.template, entry.binding.section) == (CONCEPT_LIST, None)


def test_a8_text_columns_are_above_the_definition_floor(env: Env) -> None:
    (profile,) = env.a8.source.profiles()
    floor = lexicon().DEFINITION_MEAN_LENGTH_FLOOR

    assert profile.row_count == 40
    assert profile.column("id").is_unique and profile.column("code").is_unique
    assert profile.column("label").mean_length > floor
    assert profile.column("description").mean_length > floor


def test_lexicon_declares_the_spec_tokens_in_a_fixed_role_order() -> None:
    lex = lexicon()
    table = {entry.name: entry for entry in lex.ROLES}
    order = [entry.name for entry in lex.ROLES]

    for role, tokens in SPEC_TOKENS.items():
        assert {lex.normalize(t) for t in table[role].tokens} == set(map(lex.normalize, tokens))
        assert table[role].description, role
    assert {"valid_from", "valid_to", "synonyms"} <= set(table)
    assert len(order) == len(set(order))
    assert order.index("code") < order.index("concept_id")
    assert order.index("name") < order.index("definition")


def test_floors_are_named_and_the_embedding_floor_is_unset() -> None:
    for name in FLOORS:
        assert isinstance(getattr(lexicon(), name), int | float), name
    assert eligibility_utils().EMBEDDING_MARGIN_FLOOR is None


def test_a8_sets_before_hints(env: Env) -> None:
    roles = by_role(_a8(env))

    assert set(roles["concept_id"].members) == {"id", "code"}
    assert set(roles["code"].members) == {"id", "code"}
    assert set(roles["name"].members) == {"label", "description"}
    assert set(roles["definition"].members) == {"label", "description"}


def test_a8_lexicon_settles_with_cross_role_exclusion(env: Env) -> None:
    roles = by_role(_a8(env))

    assert (roles["code"].choice, roles["code"].source) == ("code", "lexicon")
    assert (roles["concept_id"].choice, roles["concept_id"].excluded_by) == ("id", "code")
    assert roles["concept_id"].source in {"lexicon", "code"}
    assert (roles["name"].choice, roles["name"].source) == ("label", "lexicon")
    assert roles["definition"].choice == "description"
    assert roles["definition"].source in {"lexicon", "code"}
    assert not any("embedding" in hint_sources(entry) for entry in roles.values())


def test_a8_ambiguity_list_names_all_four_columns(env: Env) -> None:
    found = eligibility().ambiguities(_a8(env))
    chosen = {entry.choice: entry for entry in found}

    assert {entry.stage for entry in found} == {"roles"}
    assert {"id", "code", "label", "description"} <= set(chosen)
    assert set(chosen["id"].members) == {"id", "code"}
    assert chosen["id"].exclusions == {"code": "code"}
    assert any("lexicon" in hint for hint in chosen["label"].hints)


def test_lexicon_matches_whole_tokens_only() -> None:
    hits = eligibility_utils().lexicon_hits

    assert tuple(hits("code", ["concept_code", "serial"])) == ("concept_code",)
    assert tuple(hits("concept_id", ["concept_code", "serial"])) == ()
    assert tuple(hits("concept_id", ["ConceptId", "concept_code"])) == ("ConceptId",)


def test_constant_column_is_eligible_for_status_only(env: Env) -> None:
    binding = concept_list_over(env.ranked.ref("flagged"))
    found = eligibility().compute(binding, env.ranked.profiles())

    assert {entry.role for entry in found if "flag" in entry.members} == {"status"}


def test_embedding_with_a_clear_margin_settles_the_definition(env: Env) -> None:
    embedder = clear_embedder()
    found = _ranked(env, embedder=embedder, margin_floor=FLOOR)
    definition = by_role(found)["definition"]

    assert set(definition.members) == {"zorbit_text", "quaxle_body"}
    assert (definition.choice, definition.source) == ("zorbit_text", "embedding")
    assert _embedding(definition).rank[0][0] == "zorbit_text"
    assert _embedding(definition).margin == pytest.approx(1.0)
    (entry,) = [a for a in eligibility().ambiguities(found) if a.choice == "zorbit_text"]
    assert entry.source == "embedding"
    assert any("embedding" in text for text in entry.hints)
    assert embedder.calls >= 1


def test_embedding_below_the_floor_records_the_rank_without_a_choice(env: Env) -> None:
    found = _ranked(env, embedder=close_embedder(), margin_floor=FLOOR)
    definition = by_role(found)["definition"]
    hint = _embedding(definition)

    assert [column for column, _ in hint.rank] == ["zorbit_text", "quaxle_body"]
    expected = cosine([1.0, 0.9], [1.0, 0.0]) - cosine([0.9, 1.0], [1.0, 0.0])
    assert hint.margin == pytest.approx(expected)
    assert hint.margin < FLOOR
    assert definition.choice is None


def test_embedding_without_a_floor_records_the_rank_only(env: Env) -> None:
    definition = by_role(_ranked(env, embedder=clear_embedder()))["definition"]

    assert _embedding(definition).margin == pytest.approx(1.0)
    assert definition.choice is None
    assert not any("embedding" in hint_sources(entry) for entry in _ranked(env))


def test_text_of_a_relation_no_entity_model_names_is_refused(env: Env, tmp_path: Path) -> None:
    source = env.acris.source
    tools = fresh_tools(source, tmp_path / "roles", stage="roles", structure=env.a10)
    code_table = RelationRef(provider="staging", name=ref(source, GROUPED_CODE_TABLE))

    with pytest.raises(ProfileRefusedError, match=GROUPED_CODE_TABLE):
        tools.read_schema_text(code_table, ["DOC. TYPE DESCRIPTION"])
    assert sql_lines(tmp_path / "roles") == []


def test_sensitive_column_of_a_bound_dictionary_is_refused(env: Env, tmp_path: Path) -> None:
    source = env.acris.source
    dictionary = sheet(source, "Column_Info")[0]
    sensitive = frozenset({(dictionary, "Column Description")})
    policy = EvidencePolicy(tier="metadata_only", sensitive=sensitive)
    ledger = Ledger(tmp_path).run("roles")
    tools = ProfileTools(source.tools.provider, policy, ledger, stage="roles", structure=env.a10)
    columns = ["Column Name", "Column Description"]

    with pytest.raises(ProfileRefusedError, match="Column Description"):
        tools.read_schema_text(RelationRef(provider="staging", name=dictionary), columns)
    assert sql_lines(tmp_path / "roles") == []


def test_roles_read_text_only_from_schema_relations(env: Env) -> None:
    assert env.a10_roles is not None
    lines = sql_lines(env.root / "a10" / "roles")

    assert {line["stage"] for line in lines} == {"roles"}
    bearing = [line for line in lines if line["value_bearing"]]
    assert bearing and all(line["schema_relation"] for line in bearing)


def test_mentions_match_normalized_token_sequences() -> None:
    mentions = eligibility_utils().mentions
    names = ["Property_Types_Codes", "State_Codes", REAL_MASTER]
    linked = f"linked back to the {REAL_MASTER.upper()} dataset"

    found = mentions("see ACRIS Property Type Codes dataset for possible values", names)
    assert set(found) == {"Property_Types_Codes"}
    assert set(mentions(linked, names)) == {REAL_MASTER}
    assert set(mentions("Statement codes", names)) == set()


def test_property_type_note_binds_the_code_table_domain(env: Env) -> None:
    legals = [d for d in sheet(env.acris.source, "Column_Info") if "_Property_Legals_" in d]
    domains = per_property(env.pinned_roles.eligibilities, "domain")
    by_subject = {(entry.relation, entry.subject): entry for entry in domains}

    assert len(legals) == 2
    for dictionary in legals:
        entry = by_subject[(dictionary, "Property Type")]
        assert set(entry.members) - {"none"} == {"Property_Types_Codes"}
        assert entry.choice == "Property_Types_Codes"
        assert entry.source not in {None, "embedding", "human"}


def test_nine_properties_reference_the_five_code_tables(env: Env) -> None:
    domains = per_property(env.pinned_roles.eligibilities, "domain")
    chosen = [entry for entry in domains if entry.choice in CODE_TABLES]

    assert Counter(entry.subject for entry in chosen) == REFERENCES
    assert len({entry.choice for entry in chosen}) == len(CODE_TABLES)


def test_referenced_code_tables_resolve_to_code_lists(env: Env) -> None:
    source, roles = env.acris.source, env.pinned_roles
    lists = [e for e in roles.structure.bound if e.binding.template == CONCEPT_LIST]
    hint_shape = load_concept_source_spec(ACRIS_SPEC).shape or []

    assert {e.binding.relations[0]: e.binding.section for e in lists} == {
        ref(source, name): f"code_lists.{name}" for name in CODE_TABLES
    }
    assert {e.source for e in lists} == {"human"}
    rendered = {(entry.section, entry.template) for entry in roles.structure.to_shape()}
    assert rendered == {(entry.section, entry.template) for entry in hint_shape}
    assert not [a for a in roles.ambiguities if set(a.members) == SECTION_MEMBERS]


def test_standalone_list_section_is_ambiguous(tmp_path: Path) -> None:
    con, source = memory_source(COLOURS_SQL, "tc", tmp_path)
    structured = run(source, fresh_tools(source, tmp_path / "ledger" / "structure"))
    roles = run_roles(source, structured, tmp_path / "ledger" / "roles")
    con.close()

    (section,) = [a for a in roles.ambiguities if set(a.members) == SECTION_MEMBERS]
    assert section.subject.startswith("tc.colours")
    assert section.choice is None
    with pytest.raises(ValueError):
        roles.structure.to_shape()


def test_absent_code_table_leaves_two_unresolved_references(env: Env) -> None:
    dictionaries = sheet(env.dropped.source, "Column_Info")
    masters = [d for d in dictionaries if "_Property_Master_" in d]
    roles = env.dropped_roles
    unresolved = [a for a in roles.ambiguities if "unresolved_reference" in a.hints]
    misspelled = [a for a in unresolved if MISSPELLED_PROPERTY in a.subject]

    assert len(masters) == 2 and len(misspelled) == 2
    assert {next(m for m in masters if a.subject.startswith(m)) for a in misspelled} == set(
        masters
    )
    assert all(a.choice is None for a in misspelled)
    domains = per_property(roles.eligibilities, "domain")
    misspelled_domains = [e for e in domains if e.subject == MISSPELLED_PROPERTY]
    assert {e.choice for e in misspelled_domains} <= {None, "none"}


def test_legals_join_master_on_the_columns_the_header_names(env: Env) -> None:
    joins = env.a10_roles.joins
    (legals,) = [j for j in joins if j.from_entity == "ACRIS - Real Property Legals"]

    assert legals.to_entity == REAL_MASTER
    assert list(legals.columns) == JOIN_COLUMNS
    assert "Record Type" not in legals.columns
    assert legals.evidence == "declared"


def test_eight_declared_joins_over_the_directory(env: Env) -> None:
    joins = env.a10_roles.joins

    assert len(joins) == 8
    assert {(join.from_entity, join.to_entity) for join in joins} == DECLARED_JOINS
    assert all(list(join.columns) == JOIN_COLUMNS for join in joins)


def test_header_keys_are_found_by_the_lexicon_tokens(env: Env) -> None:
    headers = sheet(env.acris.source, "Dataset_Info")
    (legals,) = [h for h in headers if "_Real_Property_Legals_" in h]
    header = env.a10_roles.headers[legals]

    assert header.name == "ACRIS - Real Property Legals"
    assert header.description == LEGALS_DESCRIPTION
    assert header.values["Dataset Name"] == header.name
    assert "Detailed Description" in header.values
    assert header.values["Detailed Description"] is None
    assert any(key.startswith(PROMPT_KEY_START) for key in header.prose)
    assert "Agency Name" in header.prose
    assert not {"Dataset Name", "Dataset Description"} & set(header.prose)


def test_column_info_role_map_is_computed_once_and_reused_fourteen_times(env: Env) -> None:
    dictionaries = sheet(env.acris.source, "Column_Info")
    maps = {d: by_role(env.a10_roles.eligibilities, d) for d in dictionaries}
    reused = [d for d, roles in maps.items() if any(e.source == "memory" for e in roles.values())]

    assert all(maps.values())
    assert len(reused) == 14
    (first,) = set(dictionaries) - set(reused)
    for dictionary in reused:
        for entry in maps[dictionary].values():
            assert not {"lexicon", "embedding"} & set(hint_sources(entry)), entry.role
            if len(entry.members) > 1:
                assert entry.source == "memory", (dictionary, entry.role)
                assert entry.choice == maps[first][entry.role].choice


@pytest.mark.parametrize("path", [A2_RUN, ACRIS_RUN], ids=["a2", "a10"])
def test_run_descriptors_validate(path: Path) -> None:
    raw = yaml.safe_load(path.read_text())
    loaded = descriptor(path)
    fields = set(eligibility_utils().RunDescriptor.model_fields)

    assert {"id", "identity_system", "release", "scheme", "term_storage", "entity_case"} <= fields
    assert (loaded.id, loaded.identity_system) == (raw["id"], raw["identity_system"])
    assert str(loaded.release) == str(raw["release"])


def test_a2_assembles_a_spec_whose_import_equals_the_golden(env: Env) -> None:
    source = env.a2.source
    structured = env.structured(env.a2, "a2")
    ledger = Ledger(env.root / "a2").run("roles")
    full = EvidencePolicy(tier="full")
    tools = ProfileTools(source.tools.provider, full, ledger, stage="roles", structure=structured)
    roles = eligibility().run_roles(structured, source.profiles(), tools)

    spec = eligibility().assemble_spec(descriptor(A2_RUN), roles)
    assert spec is not None, [(a.subject, a.members) for a in roles.ambiguities]
    imported = import_concept_source(
        spec, source.tools.provider, dataset_map={}, staging=env.a2.staging
    )
    assert yaml.safe_dump(imported.manifest, sort_keys=False) == A2_GOLDEN.read_text()
    assert model_calls(env.root / "a2" / "roles") == []


def test_a10_with_the_hint_assembles_the_hand_written_spec(env: Env) -> None:
    roles = env.hinted_roles
    column_roles = {(e.relation, e.role): e for e in roles.eligibilities if e.subject is None}

    spec = eligibility().assemble_spec(descriptor(ACRIS_RUN), roles)
    assert spec is not None, [(a.subject, a.members) for a in roles.ambiguities]
    assert normalized(spec) == normalized(load_concept_source_spec(ACRIS_SPEC))
    grouped = column_roles[(ref(env.acris.source, GROUPED_CODE_TABLE), "code")]
    assert (grouped.choice, grouped.source) == ("DOC. TYPE", "human")
    assert model_calls(env.root / "hinted" / "roles") == []


def test_a8_ambiguity_list_is_never_empty(env: Env) -> None:
    structured = env.structured(env.a8, "a8_roles")
    roles = run_roles(env.a8.source, structured, env.root / "a8_roles" / "roles")
    a8_descriptor = eligibility_utils().RunDescriptor(id="a8", identity_system="a8", release="1")

    assert roles.ambiguities
    assert eligibility().assemble_spec(a8_descriptor, roles) is None

"""Structure stage: the fixed six-step order over landed relations, from aggregates only."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterator
from functools import cache
from pathlib import Path
from typing import Any

import duckdb
import pytest

from sema.binding.spec import load_concept_source_spec
from sema.binding.spec_sections import RelationRef
from sema.intake.loaders.athena import land_athena_bundle
from sema.intake.loaders.semsql import land_semsql
from sema.intake.loaders.tabular import land_tabular
from sema.intake.profile import ProfileRefusedError
from sema.intake.shapes.base import JOIN_RESOLVE_FLOOR
from tests.unit.intake.shapes.shapes_env import Source, warehouse_source
from tests.unit.intake.structure_env import (
    A2_BOUND,
    A2_SHAPE,
    A2_WORKBOOK,
    A3_DB,
    A3_SECTIONS,
    ACRIS,
    ACRIS_SPEC,
    CODE_TABLES,
    GROUPED_CODE_TABLE,
    HELDOUT,
    OUTLINE_SQL,
    STEPS,
    TWIN_SUBJECTS,
    assert_partition,
    a2_hint,
    bound_by_template,
    bound_relations,
    copy_code_tables,
    entries,
    entry_over,
    expected_stage_floors,
    fresh_tools,
    ignored_relations,
    joins,
    land,
    namespace,
    ref,
    reasons,
    resolving,
    run,
    sheet,
    sql_lines,
    stems,
    structure,
)

pytestmark = pytest.mark.unit

Result = Callable[[], Any]


@pytest.fixture(scope="module")
def tmp_module(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("structure")


def _landed(tmp: Path, name: str, lander: Any, path: Path) -> Iterator[Source]:
    staging, source = land(tmp / name, name, lander, path)
    yield source
    staging.close()


@pytest.fixture(scope="module")
def a2(tmp_module: Path) -> Iterator[Source]:
    yield from _landed(tmp_module, "a2", land_tabular, A2_WORKBOOK)


@pytest.fixture(scope="module")
def a3(tmp_module: Path) -> Iterator[Source]:
    yield from _landed(tmp_module, "a3", land_semsql, A3_DB)


@pytest.fixture(scope="module")
def acris(tmp_module: Path) -> Iterator[Source]:
    yield from _landed(tmp_module, "acris", land_tabular, ACRIS)


@pytest.fixture(scope="module")
def heldout(tmp_module: Path) -> Iterator[Source]:
    yield from _landed(tmp_module, "heldout", land_tabular, HELDOUT)


@pytest.fixture(scope="module")
def codes(tmp_module: Path) -> Iterator[Source]:
    directory = copy_code_tables(tmp_module / "codes_only" / "csv")
    yield from _landed(tmp_module, "codes", land_tabular, directory)


@pytest.fixture(scope="module")
def athena(tmp_module: Path) -> Iterator[Source]:
    from tests.support.athena_bundle import make_minimal_bundle

    bundle = tmp_module / "athena_bundle"
    make_minimal_bundle(bundle)
    yield from _landed(tmp_module, "ath", land_athena_bundle, bundle)


@pytest.fixture(scope="module")
def acris_ledger(tmp_module: Path) -> Path:
    return tmp_module / "acris_runs" / "structure"


@pytest.fixture(scope="module")
def a10(acris: Source, acris_ledger: Path) -> Result:
    return cache(lambda: run(acris, fresh_tools(acris, acris_ledger)))


@pytest.fixture(scope="module")
def a10_hinted(acris: Source, tmp_module: Path) -> Result:
    tools = fresh_tools(acris, tmp_module / "acris_runs" / "hinted")
    return cache(lambda: run(acris, tools, load_concept_source_spec(ACRIS_SPEC)))


@pytest.fixture(scope="module")
def a2_result(a2: Source, tmp_module: Path) -> Result:
    return cache(lambda: run(a2, fresh_tools(a2, tmp_module / "a2_runs" / "structure")))


def test_result_records_the_six_steps_in_order(a10: Result) -> None:
    assert structure().STEPS == STEPS
    assert tuple(a10().order) == STEPS


def test_result_is_a_closed_model_that_round_trips(a10: Result) -> None:
    model = structure().StructureResult

    result = a10()

    assert model.model_config.get("extra") == "forbid"
    assert model.model_validate_json(result.model_dump_json()) == result


def test_fifteen_dictionary_sheets_report_one_signature(acris: Source, a10: Result) -> None:
    dictionaries = sheet(acris, "Column_Info")

    assert len(dictionaries) == 15
    (signature,) = {tuple(map(tuple, a10().signatures[name])) for name in dictionaries}
    assert list(signature) == sorted(signature)
    assert ("Column Name", "VARCHAR") in signature
    assert set(a10().signatures) == namespace(acris)


def test_a2_join_graph_holds_one_edge_inside_its_one_group(a2: Source, a2_result: Result) -> None:
    concepts, values = ref(a2, "codes_a2__Concepts"), ref(a2, "codes_a2__Values")

    assert [set(group) for group in a2_result().groups] == [namespace(a2)]
    (edge,) = resolving(a2_result().edges, JOIN_RESOLVE_FLOOR)
    assert (edge.from_relation, edge.from_column) == (values, "concept_code")
    assert (edge.to_relation, edge.to_column) == (concepts, "concept_code")
    assert edge.within_group and not edge.merged


def test_five_code_tables_are_five_groups_with_no_edge(codes: Source, tmp_module: Path) -> None:
    result = run(codes, fresh_tools(codes, tmp_module / "codes_runs" / "structure"))

    assert sorted(map(tuple, result.groups)) == sorted((ref(codes, n),) for n in CODE_TABLES)
    assert resolving(result.edges, JOIN_RESOLVE_FLOOR) == []
    assert not any(edge.merged for edge in result.edges)


def test_acris_directory_is_fifteen_workbook_groups_and_five_single_groups(
    acris: Source, a10: Result
) -> None:
    sizes = Counter(len(group) for group in a10().groups)

    assert sizes == {4: 15, 1: 5}
    by_stem = {g[0].rpartition("__")[0]: set(g) for g in a10().groups if len(g) == 4}
    for stem in stems(acris):
        sheets = {f"{stem}__{s}" for s in ("Column_Info", "Dataset_Info")}
        sheets |= {f"{stem}__{s}" for s in ("Dropdown_Options", "Dataset_Revision_History")}
        assert by_stem[stem] == sheets, stem


def test_claimed_relations_record_evidence_edges_and_never_merge(
    acris: Source, a10: Result
) -> None:
    evidence = resolving(a10().edges, JOIN_RESOLVE_FLOOR)
    headers = sheet(acris, "Dataset_Info")
    dictionaries = sheet(acris, "Column_Info")

    assert not any(edge.merged for edge in a10().edges)
    assert any(joins(edge, headers[0], other) for edge in evidence for other in headers[1:])
    for subject in TWIN_SUBJECTS:
        real, personal = (
            next(d for d in dictionaries if f"_{kind}_Property_{subject}_" in d)
            for kind in ("Real", "Personal")
        )
        assert any(joins(edge, real, personal, "Column Name") for edge in evidence), subject


def test_a10_ignores_thirty_relations_by_structural_reason(acris: Source, a10: Result) -> None:
    found = reasons(a10())

    assert len(a10().ignored) == 30
    assert {found[name] for name in sheet(acris, "Dataset_Revision_History")} == {"no_rows"}
    assert {found[name] for name in sheet(acris, "Dropdown_Options")} == {"boilerplate"}
    for ignored in a10().ignored:
        assert ignored.template == "ignore"
        if ignored.slots["reason"] == "boilerplate":
            assert "concept_list" in ignored.slots["would_match"].split(",")


def test_ignored_relations_appear_in_no_candidate(acris: Source, a10: Result) -> None:
    ignored = set(ignored_relations(a10()))

    assert not ignored & set(bound_relations(a10()))
    for ambiguity in a10().ambiguities:
        assert not any(name in member for name in ignored for member in ambiguity.members)


def test_a10_binds_fifteen_per_file_entity_models_with_the_key_value_header(
    acris: Source, a10: Result
) -> None:
    models = entries(a10(), "entity_model")

    assert len(models) == 15
    by_dictionary = {entry.binding.slots["dictionary"]: entry for entry in models}
    for stem in stems(acris):
        entry = by_dictionary[f"{stem}__Column_Info"]
        assert entry.binding.slots["layout"] == "per_file"
        assert entry.binding.slots["header"] == f"{stem}__Dataset_Info"


def test_a10_binds_only_the_fifteen_entity_models(a10: Result) -> None:
    assert {entry.binding.template for entry in a10().bound} == {"entity_model"}
    assert len(a10().bound) == 15


def test_a10_lists_the_five_code_tables_with_their_members(acris: Source, a10: Result) -> None:
    by_subject = {ambiguity.subject: ambiguity for ambiguity in a10().ambiguities}

    assert len(a10().ambiguities) == len(CODE_TABLES)
    assert set(by_subject) == {ref(acris, name) for name in CODE_TABLES}
    for name in CODE_TABLES:
        ambiguity = by_subject[ref(acris, name)]
        # Scenario "A10 structure with one honest ambiguity" (amended 2026-10-21): the
        # grouped code table offers the same two members as the other code tables.
        expected = {"concept_list": 1, "entity_model": 1}
        assert ambiguity.stage == "structure"
        assert Counter(m.partition("(")[0] for m in ambiguity.members) == expected, name
        lone = f"entity_model(dictionary={ref(acris, name)}, layout=per_file)"
        assert lone in ambiguity.members, name


def test_code_bound_entries_carry_confidence_one(a10: Result) -> None:
    assert a10().bound
    for entry in a10().bound:
        assert entry.source == "code"
        assert entry.confidence == 1.0


def test_hand_written_spec_as_hint_settles_the_ambiguity(acris: Source, a10_hinted: Result) -> None:
    grouped = entry_over(a10_hinted(), ref(acris, GROUPED_CODE_TABLE))

    assert a10_hinted().ambiguities == ()
    assert grouped.binding.template == "concept_list"
    assert grouped.source == "human"
    assert_partition(acris, a10_hinted())


def test_hinted_a10_renders_the_hint_shape(a10_hinted: Result) -> None:
    hint = load_concept_source_spec(ACRIS_SPEC)
    rendered = {(entry.section, entry.template) for entry in a10_hinted().to_shape()}

    assert hint.shape is not None
    assert rendered == {(entry.section, entry.template) for entry in hint.shape}


def test_relation_eligible_for_two_templates_is_listed_with_both(tmp_path: Path) -> None:
    con = duckdb.connect(":memory:")
    for statement in OUTLINE_SQL:
        con.execute(statement)
    source = warehouse_source(con, "tw", tmp_path / "ledger" / "landing")
    result = run(source, fresh_tools(source, tmp_path / "ledger" / "structure"))
    con.close()

    assert bound_relations(result) == []
    (ambiguity,) = result.ambiguities
    assert ambiguity.subject == "tw.outline"
    # A text code beside a label is also a lone dictionary, so entity_model joins the two.
    assert {member.partition("(")[0] for member in ambiguity.members} == {
        "concept_list",
        "entity_model",
        "path_codes",
    }
    assert all("tw.outline" in member for member in ambiguity.members)


@pytest.mark.parametrize("name", ["a2", "a3", "acris", "heldout"])
def test_every_relation_is_accounted_for_exactly_once(
    name: str, request: pytest.FixtureRequest, tmp_path: Path
) -> None:
    source: Source = request.getfixturevalue(name)

    result = run(source, fresh_tools(source, tmp_path / "ledger" / "structure"))

    assert_partition(source, result)


@pytest.mark.parametrize("name", ["a3", "athena"])
def test_a_non_tabular_landing_is_one_group(
    name: str, request: pytest.FixtureRequest, tmp_path: Path
) -> None:
    source: Source = request.getfixturevalue(name)

    result = run(source, fresh_tools(source, tmp_path / "ledger" / "structure"))

    assert [set(group) for group in result.groups] == [namespace(source)]


def test_wrong_hint_fails_join_resolves_naming_the_key_and_rate(
    a2: Source, tmp_path: Path
) -> None:
    hint = a2_hint(tmp_path / "hint", value_key="status")
    module = structure()

    with pytest.raises(module.HintRejectedError) as caught:
        run(a2, fresh_tools(a2, tmp_path / "ledger" / "structure"), hint)

    (failure,) = [f for f in caught.value.failures if f.check == "join_resolves"]
    assert failure.column == "status"
    assert failure.relation == ref(a2, "codes_a2__Values")
    assert failure.evidence["resolve"] < JOIN_RESOLVE_FLOOR
    assert "join_resolves" in str(caught.value) and "status" in str(caught.value)


def test_a2_hint_pins_its_templates_as_human(a2: Source, tmp_path: Path) -> None:
    hint = a2_hint(tmp_path / "hint")

    result = run(a2, fresh_tools(a2, tmp_path / "ledger" / "structure"), hint)

    assert result.ambiguities == ()
    assert {entry.source for entry in result.bound} == {"human"}
    assert [(e.section, e.template) for e in result.to_shape()] == A2_SHAPE
    assert_partition(a2, result)


def test_a2_shape_is_code_bound_with_confidence_one(a2_result: Result) -> None:
    assert a2_result().ambiguities == ()
    assert bound_by_template(a2_result()) == A2_BOUND
    assert {(entry.confidence, entry.source) for entry in a2_result().bound} == {(1.0, "code")}
    assert [(e.section, e.template) for e in a2_result().to_shape()] == A2_SHAPE


def test_a3_shape_names_semsql_for_every_section_its_loader_fills(
    a3: Source, tmp_path: Path
) -> None:
    result = run(a3, fresh_tools(a3, tmp_path / "ledger" / "structure"))

    assert {(e.binding.template, e.confidence, e.source) for e in result.bound} == {
        ("semsql", 1.0, "code")
    }
    shape = sorted((entry.section, entry.template) for entry in result.to_shape())
    assert shape == sorted((section, "semsql") for section in A3_SECTIONS)

def test_structure_issues_no_value_bearing_statement(a10: Result, acris_ledger: Path) -> None:
    a10()
    lines = sql_lines(acris_ledger)

    assert lines
    assert {line["stage"] for line in lines} == {"structure"}
    assert [line for line in lines if line["value_bearing"]] == []


def test_floors_are_the_union_of_every_template_and_the_base(a10: Result) -> None:
    floors = {k: tuple(v) if isinstance(v, list) else v for k, v in a10().floors.items()}
    assert floors == expected_stage_floors()


def test_bound_dictionaries_and_headers_are_the_schema_relations(
    acris: Source, a10: Result, tmp_path: Path
) -> None:
    schema = set(sheet(acris, "Column_Info")) | set(sheet(acris, "Dataset_Info"))
    assert set(a10().schema_relations) == schema

    tools = fresh_tools(acris, tmp_path / "ledger" / "roles", stage="roles", structure=a10())
    dictionary = RelationRef(provider="staging", name=sheet(acris, "Column_Info")[0])
    code_table = RelationRef(provider="staging", name=ref(acris, GROUPED_CODE_TABLE))

    assert tools.read_schema_text(dictionary, ["Column Name"])
    (line,) = sql_lines(tmp_path / "ledger" / "roles")
    assert line["value_bearing"] and line["schema_relation"]
    with pytest.raises(ProfileRefusedError):
        tools.read_schema_text(code_table, ["DOC. TYPE"])


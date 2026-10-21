"""ignore: no_header, no_rows, boilerplate, computed from profiles before any template."""

from __future__ import annotations

import importlib
from collections.abc import Iterator
from pathlib import Path

import pytest

from sema.ingest.duckdb_staging import Staging
from tests.unit.intake.shapes.shapes_env import (
    Source,
    candidates,
    shapes,
    staging_source,
    verify,
    write_workbook,
)

pytestmark = pytest.mark.unit

ID = "ignore"
GROUPS = ("g1", "g2", "g3")
OPTIONS: list[list[object]] = [["code", "option"], ["A", "Alpha"], ["B", "Beta"], ["C", "Gamma"]]


def _sheets(
    group: str, *, options: list[list[object]] = OPTIONS
) -> dict[str, list[list[object]]]:
    return {
        "Main": [
            ["name", "description"],
            [f"{group}_a", f"First item of {group}"],
            [f"{group}_b", f"Second item of {group}"],
        ],
        "Opts": options,
        "Log": [["Version", "Date", "Change"]],
        "Figures": [[1, 2, 3], [4, 5, 6]],
    }


def _land(tmp_path: Path, books: dict[str, dict[str, list[list[object]]]]) -> Iterator[Source]:
    from sema.intake.loaders.tabular import land_tabular

    directory = tmp_path / "books"
    directory.mkdir()
    for stem, sheets in books.items():
        write_workbook(directory / f"{stem}.xlsx", sheets)
    with Staging(str(tmp_path / "staging.duckdb"), schemas=()) as staging:
        land = land_tabular(directory, staging, schema="land_s")
        yield staging_source(staging, land, tmp_path / "ledger" / "run")


@pytest.fixture
def three(tmp_path: Path) -> Iterator[Source]:
    yield from _land(tmp_path, {group: _sheets(group) for group in GROUPS})


@pytest.fixture
def differing(tmp_path: Path) -> Iterator[Source]:
    changed = OPTIONS[:-1] + [["C", "Other"]]
    books = {group: _sheets(group) for group in GROUPS[:2]}
    books["g3"] = _sheets("g3", options=changed)
    yield from _land(tmp_path, books)


@pytest.fixture
def single(tmp_path: Path) -> Iterator[Source]:
    sheets = _sheets("g1")
    sheets["Notes"] = [["note"], ["alpha"], ["alpha"], ["beta"]]
    yield from _land(tmp_path, {"g1": sheets})


def _reasons(source: Source) -> dict[str, str]:
    found = candidates(ID, source.profiles(), source)
    reasons = {}
    for ignored in found:
        (relation,) = ignored.relations
        reasons[relation] = ignored.slots["reason"]
    return reasons


def _named(source: Source, sheet: str) -> list[str]:
    return [source.ref(f"{group}__{sheet}") for group in GROUPS]


def test_zero_row_relation_is_no_rows_before_boilerplate(three: Source) -> None:
    reasons = _reasons(three)

    assert {reasons[name] for name in _named(three, "Log")} == {"no_rows"}


def test_headerless_relation_is_no_header_before_boilerplate(three: Source) -> None:
    reasons = _reasons(three)

    assert {reasons[name] for name in _named(three, "Figures")} == {"no_header"}


def test_headerless_relation_stays_ignored_and_is_a_proposal_target(three: Source) -> None:
    structure = importlib.import_module("sema.intake.structure")
    figures = _named(three, "Figures")

    result = structure.run_structure(three.land, three.tools, provider=three.provider)

    ignored = [b.relations[0] for b in result.ignored]
    reasons = {b.relations[0]: b.slots["reason"] for b in result.ignored}
    assert {name: reasons.get(name) for name in figures} == dict.fromkeys(figures, "no_header")
    assert len(ignored) == len(set(ignored))
    assert not set(figures) & {a.subject for a in result.ambiguities}
    assert not set(figures) & {n for e in result.bound for n in e.binding.relations}
    assert result.proposal_targets == tuple(figures)


def test_relation_identical_across_three_groups_is_boilerplate(three: Source) -> None:
    found = candidates(ID, three.profiles(), three)
    options = [b for b in found if b.relations[0] in _named(three, "Opts")]

    assert len(options) == 3
    for ignored in options:
        assert ignored.template == ID
        assert ignored.slots["reason"] == "boilerplate"
        assert "concept_list" in ignored.slots["would_match"].split(",")


def test_relations_with_per_file_content_are_not_ignored(three: Source) -> None:
    assert not set(_named(three, "Main")) & set(_reasons(three))


def test_relation_differing_in_one_column_in_one_group_is_not_boilerplate(
    differing: Source,
) -> None:
    reasons = _reasons(differing)

    assert not set(_named(differing, "Opts")) & set(reasons)
    assert {reasons[name] for name in _named(differing, "Log")} == {"no_rows"}


def test_single_group_source_has_no_boilerplate(single: Source) -> None:
    reasons = _reasons(single)

    assert "boilerplate" not in reasons.values()
    assert single.ref("g1__Opts") not in reasons


def test_label_signature_without_an_ignore_profile_is_not_ignored(single: Source) -> None:
    assert single.ref("g1__Notes") not in _reasons(single)


def test_acris_ignores_revision_history_as_no_rows_and_dropdowns_as_boilerplate(
    acris: Source,
) -> None:
    found = candidates(ID, acris.profiles(), acris)
    reasons = {b.relations[0]: b for b in found}
    revisions = [p.ref.name for p in acris.sheet("Dataset_Revision_History")]
    dropdowns = [p.ref.name for p in acris.sheet("Dropdown_Options")]

    assert len(found) == 30
    assert {reasons[name].slots["reason"] for name in revisions} == {"no_rows"}
    assert {reasons[name].slots["reason"] for name in dropdowns} == {"boilerplate"}
    for name in dropdowns:
        assert "concept_list" in reasons[name].slots["would_match"].split(",")
    kept = acris.sheet("Dataset_Info") + acris.sheet("Column_Info")
    assert not {p.ref.name for p in kept} & set(reasons)


def test_boilerplate_verify_passes_and_records_the_group_share_floor(three: Source) -> None:
    group = three.profiles()
    boilerplate = [
        b for b in candidates(ID, group, three) if b.slots["reason"] == "boilerplate"
    ]

    for ignored in boilerplate:
        verdict = verify(ID, ignored, group, three)
        assert verdict.status == "pass"
        share = verdict.floors["BOILERPLATE_GROUP_SHARE_FLOOR"]
        assert share == shapes(ID).BOILERPLATE_GROUP_SHARE_FLOOR
    assert boilerplate


def test_ignore_reads_no_value(three: Source) -> None:
    candidates(ID, three.profiles(), three)

    assert three.value_bearing() == []

"""The embedding margin measurement: floor rule, case extraction, held-out refusal.

Contract the tests pin (task 8.3; role-eligibility "Embedding rank narrows when the
margin is clear"):

- ``scripts.measure_embedding_margin_utils``:
  ``MarginCase(fixture, relation, role, members, top, expected, margin)``, frozen, with
  ``skipped`` (``expected`` is None or not in ``members``) and ``correct`` (not skipped
  and ``top == expected``); ``select_floor(cases) -> float``;
  ``expected_columns(spec) -> dict[(relation, role), column]``;
  ``margin_cases(fixture, eligibilities, profiles, expected, embedder) -> list[MarginCase]``;
  ``refuse_heldout(path) -> Path`` raising ``HeldOutSourceError`` (a ``ValueError``).
- ``scripts.measure_embedding_margin``: ``MeasurementFixture(name, path, spec)``,
  ``MEASUREMENT_FIXTURES``, ``measure_fixture(fixture, embedder, workdir)``, and
  ``main(argv) -> int`` building the embedder with ``cli_factories._get_embedder``.

Modules are imported lazily so a missing one fails each test on its own.
"""

from __future__ import annotations

import importlib
import math
import string
from collections.abc import Iterator, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from sema.binding.spec import load_concept_source_spec
from sema.intake import lexicon
from sema.intake.eligibility_utils import RoleEligibility, embedding_rank
from tests.unit.intake.eligibility_env import A8_CSV, Env
from tests.unit.intake.shapes.shapes_env import FIXTURES

pytestmark = pytest.mark.unit

A8_SPEC = A8_CSV.with_name("ambiguity_a8.spec.yaml")
A2_SPEC = FIXTURES / "tabular" / "codes_a2.spec.yaml"
A3_SPEC = FIXTURES / "semsql" / "ontology.spec.yaml"
ACRIS_SPEC = FIXTURES / "acris" / "acris.spec.yaml"
A8_RELATION = "land_a8.ambiguity_a8"
A8_TRUTH = {("concept_id", "id"), ("name", "label"), ("definition", "description")}
MEASURED = {"a2", "a3", "a8", "acris"}
HELD_OUT = ("heldout", "heldout2")


def utils() -> ModuleType:
    return importlib.import_module("scripts.measure_embedding_margin_utils")


def script() -> ModuleType:
    return importlib.import_module("scripts.measure_embedding_margin")


class LetterEmbedder:
    """Letter counts plus a constant: deterministic, never a zero vector."""

    def __init__(self) -> None:
        self.calls = 0

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls += 1
        return [self._vector(text) for text in texts]

    @staticmethod
    def _vector(text: str) -> list[float]:
        folded = text.casefold()
        return [float(folded.count(letter)) for letter in string.ascii_lowercase] + [1.0]


def case(
    margin: float,
    *,
    top: str = "a",
    expected: str | None = "a",
    members: tuple[str, ...] = ("a", "b"),
    role: str = "name",
) -> Any:
    return utils().MarginCase(
        fixture="f", relation="r", role=role, members=members,
        top=top, expected=expected, margin=margin,
    )


def wrong(margin: float) -> Any:
    return case(margin, top="b", expected="a")


@pytest.fixture(scope="module")
def env(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Env]:
    held = Env(tmp_path_factory.mktemp("margin"))
    yield held
    held.close()


@pytest.fixture(scope="module")
def a8_cases(tmp_path_factory: pytest.TempPathFactory) -> tuple[list[Any], LetterEmbedder]:
    (fixture,) = [f for f in script().MEASUREMENT_FIXTURES if f.name == "a8"]
    embedder = LetterEmbedder()
    found = script().measure_fixture(fixture, embedder, tmp_path_factory.mktemp("a8"))
    return found, embedder


# --- the floor rule ---------------------------------------------------------------


def test_case_flags_follow_the_expected_column() -> None:
    assert case(0.3).correct and not case(0.3).skipped
    assert not wrong(0.3).correct and not wrong(0.3).skipped
    assert case(0.3, expected=None).skipped
    assert case(0.3, expected="z").skipped and not case(0.3, expected="z").correct


def test_floor_clears_the_widest_wrong_margin() -> None:
    cases = [case(0.40), case(0.05), wrong(0.12), wrong(0.131), case(0.02)]

    assert utils().select_floor(cases) == pytest.approx(0.15)


def test_floor_over_an_exact_two_decimal_wrong_margin_adds_one_hundredth() -> None:
    assert utils().select_floor([case(0.5), wrong(0.07)]) == pytest.approx(0.08)
    assert utils().select_floor([case(0.5), wrong(0.29)]) == pytest.approx(0.30)


def test_floor_without_wrong_choices_is_the_narrowest_correct_margin() -> None:
    assert utils().select_floor([case(0.40), case(0.123), case(0.9)]) == pytest.approx(0.13)
    assert utils().select_floor([case(0.40), case(0.07)]) == pytest.approx(0.07)


def test_skipped_cases_do_not_move_the_floor() -> None:
    skipped = [
        case(0.95, top="b", expected=None),
        case(0.90, top="b", expected="z"),
        case(0.001, expected=None),
    ]

    assert utils().select_floor([case(0.2), wrong(0.1), *skipped]) == pytest.approx(0.11)
    assert utils().select_floor([case(0.2), case(0.3), *skipped]) == pytest.approx(0.2)


def test_floor_refuses_a_set_with_nothing_counted() -> None:
    with pytest.raises(ValueError):
        utils().select_floor([case(0.4, expected=None)])
    with pytest.raises(ValueError):
        utils().select_floor([])


# --- ground truth from the hand-written specs ------------------------------------


def test_expected_columns_read_section_roles_by_relation() -> None:
    truth = utils().expected_columns(load_concept_source_spec(A2_SPEC))

    assert truth[("land_a2.codes_a2__Concepts", "name")] == "concept_name"
    assert truth[("land_a2.codes_a2__Values", "display")] == "label"
    assert truth[("land_a2.codes_a2__Values", "status")] == "status"
    assert ("land_a2.codes_a2__Concepts", "status") not in truth


def test_expected_columns_read_details_and_value_sets_over_one_relation() -> None:
    truth = utils().expected_columns(load_concept_source_spec(A3_SPEC))

    assert truth[("land_a3.semsql_terms", "display")] == "label"
    assert truth[("land_a3.semsql_terms", "definition")] == "definition"
    assert truth[("land_a3.semsql_terms", "code")] == "subject"


def test_expected_columns_apply_entity_model_roles_to_every_entity_relation() -> None:
    spec = load_concept_source_spec(ACRIS_SPEC)
    truth = utils().expected_columns(spec)
    entities = spec.entity_model.entities if spec.entity_model else None
    assert entities

    for entry in entities:
        assert truth[(entry.relation.name, "property")] == "Column Name"
        assert truth[(entry.relation.name, "description")] == "Column Description"
        assert (entry.relation.name, "notes") not in truth
    for code_list in spec.code_lists or []:
        assert truth[(code_list.relation.name, "display")] == code_list.roles.display


# --- case extraction --------------------------------------------------------------


def test_measurement_set_is_the_four_named_fixtures_and_no_held_out() -> None:
    fixtures = script().MEASUREMENT_FIXTURES

    assert {f.name for f in fixtures} == MEASURED
    for fixture in fixtures:
        for path in (Path(fixture.path), Path(fixture.spec)):
            assert path.exists(), path
            assert not set(path.resolve().parts) & set(HELD_OUT), path


def test_a8_cases_carry_the_spec_ground_truth(a8_cases: tuple[list[Any], Any]) -> None:
    found, embedder = a8_cases
    counted = {(c.role, c.expected) for c in found if not c.skipped}

    assert embedder.calls >= 1
    assert {c.fixture for c in found} == {"a8"}
    assert {c.relation for c in found} == {A8_RELATION}
    assert counted == A8_TRUTH
    assert {c.role for c in found if c.skipped} == {"code", "display"}
    assert all(len(c.members) >= 2 for c in found)


def test_a8_margins_are_the_embedding_rank_over_the_eligible_columns(
    a8_cases: tuple[list[Any], Any], env: Env
) -> None:
    found, _ = a8_cases
    (profile,) = env.a8.source.profiles()
    entries = {entry.name: entry for entry in lexicon.ROLES}

    for found_case in found:
        columns = [c for c in profile.columns if c.name in found_case.members]
        ranked = embedding_rank(LetterEmbedder(), entries[found_case.role], columns)
        assert found_case.top == ranked.rank[0][0], found_case
        assert found_case.margin == pytest.approx(ranked.margin), found_case
        assert math.isfinite(found_case.margin)


def test_margin_cases_skip_single_members_and_per_property_references(env: Env) -> None:
    (profile,) = env.a8.source.profiles()
    entries = [
        RoleEligibility(relation=A8_RELATION, role="name", members=("label", "description")),
        RoleEligibility(relation=A8_RELATION, role="code", members=("code",)),
        RoleEligibility(
            relation=A8_RELATION, role="domain", subject="label", members=("label", "none")
        ),
    ]
    truth = {(A8_RELATION, "name"): "label", (A8_RELATION, "code"): "code"}

    found = utils().margin_cases("a8", entries, [profile], truth, LetterEmbedder())

    assert [(c.role, c.expected, c.skipped) for c in found] == [("name", "label", False)]


def test_margin_cases_record_entries_without_ground_truth_as_skipped(env: Env) -> None:
    (profile,) = env.a8.source.profiles()
    entry = RoleEligibility(relation=A8_RELATION, role="display", members=("label", "description"))

    (found,) = utils().margin_cases("a8", [entry], [profile], {}, LetterEmbedder())

    assert (found.expected, found.skipped) == (None, True)
    assert found.top in entry.members


# --- held-out refusal ---------------------------------------------------------------


@pytest.mark.parametrize("held_out", HELD_OUT)
def test_refuse_heldout_names_the_path(held_out: str) -> None:
    path = FIXTURES / held_out / "never_read.csv"

    with pytest.raises(utils().HeldOutSourceError, match=held_out):
        utils().refuse_heldout(path)
    assert issubclass(utils().HeldOutSourceError, ValueError)


def test_refuse_heldout_resolves_relative_detours() -> None:
    detour = FIXTURES / "tabular" / ".." / "heldout2" / "never_read.csv"

    with pytest.raises(utils().HeldOutSourceError, match="heldout2"):
        utils().refuse_heldout(detour)


def test_refuse_heldout_passes_measurement_fixtures() -> None:
    for path in (A8_CSV, A8_SPEC, FIXTURES / "acris"):
        assert utils().refuse_heldout(path) == path.resolve()


@pytest.mark.parametrize("field", ["path", "spec"])
def test_measure_fixture_refuses_a_held_out_source_before_any_read(
    field: str, tmp_path: Path
) -> None:
    paths = {"path": A8_CSV, "spec": A8_SPEC}
    paths[field] = FIXTURES / "heldout" / "never_read.xlsx"
    fixture = script().MeasurementFixture(name="held", **paths)
    embedder = LetterEmbedder()

    with pytest.raises(utils().HeldOutSourceError, match="heldout"):
        script().measure_fixture(fixture, embedder, tmp_path)
    assert embedder.calls == 0
    assert not any(tmp_path.iterdir())


# --- the command ----------------------------------------------------------------------


def test_main_prints_every_margin_and_the_floor(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    from sema import cli_factories

    built: list[Any] = []

    def fake_embedder(config: Any) -> LetterEmbedder:
        built.append(config)
        return LetterEmbedder()

    monkeypatch.setattr(cli_factories, "_get_embedder", fake_embedder)

    code = script().main(["--fixture", "a8", "--workdir", str(tmp_path)])
    out = capsys.readouterr().out
    (fixture,) = [f for f in script().MEASUREMENT_FIXTURES if f.name == "a8"]
    found = script().measure_fixture(fixture, LetterEmbedder(), tmp_path / "again")

    assert code == 0
    assert len(built) == 1
    for found_case in found:
        line = next(
            row for row in out.splitlines()
            if found_case.role in row and f"{found_case.margin:.4f}" in row
        )
        assert found_case.relation in line and found_case.top in line
    assert f"EMBEDDING_MARGIN_FLOOR = {utils().select_floor(found):.2f}" in out

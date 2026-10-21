"""Shape template base: profiles, signatures, slot bindings, the closed template set."""

from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path

import pytest

from sema.intake.landing import LOCATOR_COLUMN
from sema.intake.loaders.tabular import land_tabular
from sema.ingest.duckdb_staging import Staging
from tests.unit.intake.shapes.shapes_env import (
    MULTI_RELATION_IDS,
    TEMPLATE_IDS,
    Source,
    base,
    shapes,
    staging_source,
    write_workbook,
)

pytestmark = pytest.mark.unit

FLOOR_NAME = re.compile(r"_(FLOOR|BOUNDS|CAP)$")
A2_CONCEPTS = "codes_a2__Concepts"
A2_VALUES = "codes_a2__Values"


def _column(profile: object, name: str) -> object:
    return next(c for c in profile.columns if c.name == name)  # type: ignore[attr-defined]


def test_template_set_is_closed_and_complete() -> None:
    ids = [template.ID for template in shapes().TEMPLATES]

    assert sorted(ids) == sorted(TEMPLATE_IDS)
    assert len(ids) == len(set(ids)) == 10


def test_multi_relation_subset_names_the_claiming_templates() -> None:
    assert set(shapes().MULTI_RELATION) == MULTI_RELATION_IDS


def test_every_template_exposes_candidates_and_verify() -> None:
    for template in shapes().TEMPLATES:
        assert callable(template.candidates), template.ID
        assert callable(template.verify), template.ID
        assert shapes(template.ID).ID == template.ID


def test_shared_floors_are_in_range() -> None:
    module = base()
    low, high = module.LABEL_MEAN_LENGTH_BOUNDS

    assert 0.0 < module.ID_DISTINCT_RATIO_FLOOR <= 1.0
    assert 0.0 < module.JOIN_RESOLVE_FLOOR <= 1.0
    assert 0 < low < high


def test_shared_floors_hold_on_the_fixtures_they_were_measured_from(a2: Source) -> None:
    module = base()
    concepts, values = a2.profile(A2_CONCEPTS), a2.profile(A2_VALUES)
    low, high = module.LABEL_MEAN_LENGTH_BOUNDS

    assert _column(concepts, "concept_code").distinct_ratio >= module.ID_DISTINCT_RATIO_FLOOR
    assert _column(values, "concept_code").distinct_ratio < module.ID_DISTINCT_RATIO_FLOOR
    assert low <= _column(concepts, "concept_name").mean_length <= high
    joined = a2.tools.check_join(values.ref, "concept_code", concepts.ref, "concept_code")
    assert joined.resolve_a_to_b >= module.JOIN_RESOLVE_FLOOR


def test_every_floor_constant_carries_its_measured_value_beside_it() -> None:
    modules = [base()] + [shapes(template_id) for template_id in TEMPLATE_IDS]
    bare = []
    for module in modules:
        lines = inspect.getsource(module).splitlines()
        for node in ast.parse("\n".join(lines)).body:
            targets = node.targets if isinstance(node, ast.Assign) else (
                [node.target] if isinstance(node, ast.AnnAssign) else []
            )
            for target in targets:
                if isinstance(target, ast.Name) and FLOOR_NAME.search(target.id):
                    nearby = lines[max(0, node.lineno - 2) : node.end_lineno]
                    if not any("#" in line for line in nearby):
                        bare.append(f"{module.__name__}.{target.id}")
    assert bare == []


def test_one_profile_per_landed_relation_with_ref_stem_and_rows(a2: Source) -> None:
    profiles = a2.profiles()

    assert [p.landed.name for p in profiles] == [r.name for r in a2.land.relations]
    concepts = a2.profile(A2_CONCEPTS)
    assert (concepts.ref.provider, concepts.ref.name) == ("staging", "land_a2." + A2_CONCEPTS)
    assert concepts.row_count == 5
    assert {p.stem for p in profiles} == {"codes_a2"}
    assert not any(p.headerless for p in profiles)


def test_column_profiles_carry_the_aggregates_and_exclude_the_locator(a2: Source) -> None:
    concepts = a2.profile(A2_CONCEPTS)
    code = _column(concepts, "concept_code")

    assert LOCATOR_COLUMN not in {c.name for c in concepts.columns}
    assert (code.type, code.null_rate, code.distinct_ratio) == ("VARCHAR", 0.0, 1.0)
    assert code.is_unique and not code.is_constant
    assert code.mean_length == pytest.approx(4.2)
    assert _column(concepts, "data_type").is_constant


def test_signature_is_the_sorted_column_type_pairs(acris: Source) -> None:
    dropdowns, infos = acris.sheet("Dropdown_Options"), acris.sheet("Dataset_Info")

    for profile in dropdowns + infos:
        assert profile.signature == tuple(sorted((c.name, c.type) for c in profile.columns))
    assert len({p.signature for p in dropdowns}) == 1
    assert len({p.signature for p in infos}) == len(infos)


def test_stem_is_the_file_part_of_a_sheet_relation(acris: Source) -> None:
    stems = {p.stem for p in acris.profiles()}

    assert len(stems) == 20
    for profile in acris.sheet("Column_Info"):
        assert profile.landed.name == f"{profile.stem}__Column_Info"
    assert acris.profile("Country_Codes").stem == "Country_Codes"


def test_headerless_comes_from_the_land_result(tmp_path: Path) -> None:
    sheets: dict[str, list[list[object]]] = {
        "Codes": [["code", "label"], ["A", "Alpha"], ["B", "Beta"]],
        "Figures": [[1, 2, 3], [4, 5, 6]],
    }
    path = write_workbook(tmp_path / "book.xlsx", sheets)
    with Staging(str(tmp_path / "staging.duckdb"), schemas=()) as staging:
        land = land_tabular(path, staging, schema="land_b")
        source = staging_source(staging, land, tmp_path / "ledger" / "run")

        assert source.profile("book__Figures").headerless
        assert not source.profile("book__Codes").headerless


def test_id_and_label_signatures_leave_a_constant_column_out(acris: Source) -> None:
    module = base()
    country = acris.profile("Country_Codes")

    assert set(module.id_columns(country)) == {"COUNTRY CODE", "DESCRIPTION"}
    assert set(module.label_columns(country)) == {"DESCRIPTION"}


def test_slot_binding_round_trips_through_json() -> None:
    module = base()
    original = module.SlotBinding(
        template="concept_list",
        section=None,
        slots={"relation": "s.r"},
        evidence={"rows": 3.0, "note": "x"},
        relations=("s.r",),
    )

    assert module.SlotBinding.model_validate_json(original.model_dump_json()) == original
    assert module.SlotBinding.model_validate(original.model_dump()) == original


def test_template_protocol_is_exported() -> None:
    assert base().Template is not None

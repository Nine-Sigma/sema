"""Concept source importer against ARRAY columns returned as numpy arrays.

The Databricks connector returns ARRAY<string> values as numpy.ndarray, not
lists; they must import exactly as the same values given as Python lists.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from sema.binding.importer_utils import expand_list
from tests.unit.binding.fixtures.concept_library import (
    MULTI_CODES,
    RACE_ID,
    SPEC_PATH,
    QueryFn,
    build_library,
    load_dataset_map,
    provider_for,
    query_fn,
    with_array_codes,
)

pytestmark = pytest.mark.unit

ToArray = Callable[[list[Any]], Any]

MIXED_ITEMS = ["Race", None, "", "  Race {SNOMED: 103579009} ", "White, other"]
EXPANDED = ["Race", "Race {SNOMED: 103579009}", "White, other"]
ARRAY_SHAPES: dict[str, ToArray] = {
    "object_dtype": lambda items: np.array(items, dtype=object),
    "string_dtype": lambda items: np.array(items, dtype=str),
}


def _as_arrays(run: QueryFn, to_array: ToArray) -> QueryFn:
    def query(sql: str) -> list[tuple[Any, ...]]:
        return [
            tuple(to_array(v) if isinstance(v, list) else v for v in row)
            for row in run(sql)
        ]

    return query


def _import(to_array: ToArray | None = None) -> Any:
    from sema.binding.importer import import_concept_source
    from sema.binding.spec import load_concept_source_spec

    con = build_library()
    with_array_codes(con)
    run = query_fn(con)
    query = _as_arrays(run, to_array) if to_array else run
    try:
        return import_concept_source(
            load_concept_source_spec(SPEC_PATH),
            provider_for(con, query),
            dataset_map=load_dataset_map(),
        )
    finally:
        con.close()


def _snapshot_hash(result: Any, path: Path) -> str:
    from sema.targets.adapters.manifest import ManifestTargetAdapter
    from sema.targets.hashing import SnapshotHasher
    from sema.targets.normalizer import TargetModelNormalizer

    path.write_text(yaml.safe_dump(result.manifest, sort_keys=False))
    return SnapshotHasher.hash(
        TargetModelNormalizer.normalize(ManifestTargetAdapter(path))
    )


def _term_synonyms(result: Any, code: str) -> list[str]:
    term = next(t for t in result.manifest["terms"] if t["code"] == code)
    synonyms: list[str] = term["synonyms"]
    return synonyms


def _concept_synonyms(result: Any, concept_id: str) -> list[str]:
    properties = result.manifest["entities"][0]["properties"]
    prop = next(p for p in properties if p["concept_ref"]["id"] == concept_id)
    synonyms: list[str] = prop["synonyms"]
    return synonyms


def test_object_array_expands_like_the_same_list() -> None:
    array = np.array(MIXED_ITEMS, dtype=object)

    assert expand_list(array) == expand_list(MIXED_ITEMS) == EXPANDED


def test_string_dtype_array_expands_to_separate_items_in_order() -> None:
    array = np.array(["Race", "", " Race {SNOMED: 103579009}", "White, other"])

    assert expand_list(array) == EXPANDED


@pytest.mark.parametrize("to_array", ARRAY_SHAPES.values(), ids=ARRAY_SHAPES.keys())
def test_empty_array_expands_to_nothing(to_array: ToArray) -> None:
    assert expand_list(to_array([])) == []


@pytest.mark.parametrize("to_array", ARRAY_SHAPES.values(), ids=ARRAY_SHAPES.keys())
def test_array_synonyms_and_codes_import_as_separate_items(to_array: ToArray) -> None:
    result = _import(to_array)

    assert _term_synonyms(result, "SYN_RACE_WHITE") == ["Caucasian"]
    assert _term_synonyms(result, "SYN_SEX_FEMALE") == ["F"]
    assert _concept_synonyms(result, RACE_ID) == ["Patient race"]
    mixed = next(m for m in result.authored.value_mappings if m.source_value == "mixed")
    assert mixed.codes == MULTI_CODES


@pytest.mark.parametrize("to_array", ARRAY_SHAPES.values(), ids=ARRAY_SHAPES.keys())
def test_array_rows_give_same_snapshot_hash_and_sidecar_as_lists(
    to_array: ToArray, tmp_path: Path
) -> None:
    from_lists, from_arrays = _import(), _import(to_array)

    assert from_arrays.manifest == from_lists.manifest
    assert _snapshot_hash(from_arrays, tmp_path / "arrays.yaml") == _snapshot_hash(
        from_lists, tmp_path / "lists.yaml"
    )
    assert (
        from_arrays.authored.model_dump_json() == from_lists.authored.model_dump_json()
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("coalesce(col_a, col_b)", ["coalesce(col_a, col_b)"]),
        ("race, ethnicity", ["race, ethnicity"]),
        ('["race_label", "race_cd"]', ["race_label", "race_cd"]),
        ("['coalesce(col_a, col_b)', 'col_c']", ["coalesce(col_a, col_b)", "col_c"]),
        ("[not, a list", ["[not, a list"]),
        (("race", None, " cd "), ["race", "cd"]),
        (["race", "", "cd"], ["race", "cd"]),
        (42, ["42"]),
        (None, []),
        ("  ", []),
    ],
)
def test_existing_value_shapes_expand_unchanged(
    value: Any, expected: list[str]
) -> None:
    assert expand_list(value) == expected

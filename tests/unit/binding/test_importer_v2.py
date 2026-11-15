"""Importer over version 2 specs: optional sections, hierarchy, lifecycle, lang, mappings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest
import yaml

from sema.targets.adapters.manifest_parser import parse_manifest_raw
from tests.unit.binding.fixtures.concept_library_v2 import (
    LIFECYCLE,
    MAPPING_ROLES,
    RETIRED_CONCEPT,
    RETIRED_TERM,
    build_library_v2,
    captured_log,
    provider_v2,
    spec_v2,
    value_sets,
)

pytestmark = pytest.mark.unit


def _import(spec: Any, provider: Any = None) -> Any:
    from sema.binding.importer import import_concept_source

    con = build_library_v2()
    try:
        return import_concept_source(
            spec, provider or provider_v2(con), dataset_map={}
        )
    finally:
        con.close()


def _edges(result: Any) -> set[tuple[str, str, str, bool, bool]]:
    parsed = parse_manifest_raw(yaml.safe_load(yaml.safe_dump(result.manifest)))
    return {
        (r.from_.id, r.to.id, r.type, bool(r.hierarchical), r.far_end_loaded)
        for r in parsed.relationships
    }


def _levels_spec() -> Any:
    return spec_v2(
        concepts={
            "relation": "land.level_concepts",
            "group_by": "code",
            "roles": {"concept_id": "code", "name": "label"},
        },
        hierarchy={
            "relation": "land.levels",
            "level_columns": ["l1", "l2", "l3"],
            "relationship_type": "BROADER",
        },
    )


def _paths_spec() -> Any:
    return spec_v2(
        concepts={
            "relation": "land.paths",
            "group_by": "code",
            "roles": {"concept_id": "code", "name": "label"},
        },
        hierarchy={"path_column": "code", "separator": "."},
    )


def test_minimal_code_list_yields_one_scheme_entity_and_empty_sidecar() -> None:
    result = _import(spec_v2())

    manifest = result.manifest
    assert [e["kind"] for e in manifest["entities"]] == ["CONCEPT_SCHEME"]
    assert manifest["vocabularies"] == []
    assert manifest["terms"] == []
    assert manifest["relationships"] == []
    authored = result.authored
    assert authored.column_bindings == []
    assert authored.value_mappings == []
    assert authored.mappings == []
    assert authored.retirements == []


def test_parent_column_emits_child_to_parent_edge() -> None:
    spec = spec_v2(hierarchy={"parent_column": "parent_id", "relationship_type": "BROADER"})

    assert _edges(_import(spec)) == {("RACE_DETAIL", "RACE", "BROADER", True, True)}


def test_level_columns_emit_parent_edges_and_skip_null_levels() -> None:
    assert _edges(_import(_levels_spec())) == {
        ("B", "A", "BROADER", True, True),
        ("C", "A", "BROADER", True, True),
        ("D", "C", "BROADER", True, True),
    }


def test_path_codes_skip_implicit_prefix_and_log_it() -> None:
    with captured_log() as messages:
        result = _import(_paths_spec())

    assert _edges(result) == {("1.1", "1", "BROADER", True, True)}
    assert [m for m in messages if "implicit" in m and "1.2" in m]


def test_retired_inline_term_is_omitted_counted_and_recorded() -> None:
    spec = spec_v2(value_sets=value_sets(**LIFECYCLE))

    with captured_log() as messages:
        result = _import(spec)

    codes = {t["code"] for t in result.manifest["terms"]}
    assert RETIRED_TERM not in codes
    assert {"RACE_W", "RACE_B"} <= codes
    retired = [r.model_dump() for r in result.authored.retirements if r.kind == "term"]
    assert retired == [
        {
            "kind": "term",
            "vocabulary": "VS_RACE",
            "code": RETIRED_TERM,
            "replaced_by": "RACE_W",
            "valid_from": "2000-01-01",
            "valid_to": "2023-12-31",
        }
    ]
    assert [m for m in messages if "1 retired term" in m]


def test_retired_concept_is_omitted_with_its_value_set() -> None:
    concepts = {
        "relation": "land.concepts",
        "group_by": "concept_id",
        "roles": {"concept_id": "concept_id", "name": "label", **LIFECYCLE},
    }
    result = _import(spec_v2(concepts=concepts, value_sets=value_sets()))

    entity = result.manifest["entities"][0]
    ids = {p["concept_ref"]["id"] for p in entity["properties"]}
    assert ids == {"RACE", "RACE_DETAIL"}
    assert "VS_OLD" not in {v["name"] for v in result.manifest["vocabularies"]}
    assert [r.model_dump() for r in result.authored.retirements] == [
        {
            "kind": "concept",
            "vocabulary": None,
            "code": RETIRED_CONCEPT,
            "replaced_by": "RACE",
            "valid_from": "2001-01-01",
            "valid_to": "2019-06-30",
        }
    ]


def test_status_outside_both_lists_names_relation_column_and_value() -> None:
    status = {"column": "state", "active_values": ["active"], "retired_values": ["x"]}
    spec = spec_v2(value_sets=value_sets(status=status))

    with pytest.raises(ValueError, match=r"land\.vals.*state.*deprecated"):
        _import(spec)


def test_lang_is_carried_onto_term_synonyms() -> None:
    from sema.binding.importer_terms_utils import collect_terms

    spec = spec_v2(value_sets=value_sets(lang="label_lang"))
    rows = [
        {
            "concept_id": "RACE",
            "code": "RACE_W",
            "label": "White",
            "synonym": "Caucasian",
            "label_lang": "en",
        }
    ]

    (term,) = collect_terms(spec, spec.value_sets.roles, "land.vals", "VS_RACE", rows, {})

    assert term.lang == "en"
    assert term.synonyms == ("Caucasian",)


def test_exact_mapping_becomes_external_ref_and_close_mapping_stays_whole() -> None:
    spec = spec_v2(
        value_sets=value_sets(),
        mappings={"relation": "land.sssom", "roles": MAPPING_ROLES},
    )

    result = _import(spec)

    terms = {t["code"]: t for t in result.manifest["terms"]}
    assert {"system": "omop", "id": "8527"} in terms["RACE_W"]["external_refs"]
    assert terms["RACE_B"]["external_refs"] == []
    race = next(
        p
        for p in result.manifest["entities"][0]["properties"]
        if p["concept_ref"]["id"] == "RACE"
    )
    assert {"system": "omop", "id": "4013886"} in race["external_refs"]
    close = [m for m in result.authored.mappings if m.predicate_id == "skos:closeMatch"]
    assert [m.model_dump() for m in close] == [
        {
            "subject_id": "syn:RACE_B",
            "predicate_id": "skos:closeMatch",
            "object_id": "omop:8516",
            "mapping_justification": "semapv:ManualMappingCuration",
            "subject_label": None,
            "object_label": None,
            "confidence": None,
        }
    ]
    assert len(result.authored.mappings) == 3


def test_mixed_providers_read_concepts_from_warehouse_and_mappings_from_staging(
    tmp_path: Path,
) -> None:
    from sema.ingest.duckdb_staging import Staging
    from sema.intake.land.relations import RelationProvider

    staging = Staging(str(tmp_path / "staging.duckdb"), schemas=("landed",))
    staging.write_table(
        "landed",
        "sssom",
        pa.table(
            {
                "subject_id": ["syn:RACE_W"],
                "predicate_id": ["skos:exactMatch"],
                "object_id": ["omop:8527"],
                "mapping_justification": ["semapv:ManualMappingCuration"],
            }
        ),
        {
            "subject_id": "VARCHAR",
            "predicate_id": "VARCHAR",
            "object_id": "VARCHAR",
            "mapping_justification": "VARCHAR",
        },
        {},
        None,
    )
    spec = spec_v2(
        value_sets=value_sets(),
        mappings={
            "relation": {"provider": "staging", "name": "landed.sssom"},
            "roles": MAPPING_ROLES,
        },
    )
    con = build_library_v2()
    warehouse = provider_v2(con).warehouse

    result = _import(spec, RelationProvider(warehouse=warehouse, staging=staging))

    terms = {t["code"]: t for t in result.manifest["terms"]}
    assert terms["RACE_W"]["external_refs"] == [{"system": "omop", "id": "8527"}]
    staging.close()
    con.close()


def test_shape_is_recorded_in_the_import_log() -> None:
    spec = spec_v2(shape=[{"section": "concepts", "template": "concept_list"}])

    with captured_log() as messages:
        _import(spec)

    assert [m for m in messages if "concept_list" in m]

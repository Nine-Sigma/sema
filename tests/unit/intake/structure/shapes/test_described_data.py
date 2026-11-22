"""Described data is not a dictionary (relation-structure, ADDED 2026-10-21).

Pinned rule: a relation whose landed column names equal, after the lexicon's name
normalization, the property names a bound dictionary relation lists is the data that
dictionary describes. The ``entity_model`` candidate that would use that relation as a
dictionary or header is dropped before precedence, so a lone ``concept_list`` binds by code.
The comparison reads schema-relation text only, so it runs under ``metadata_only``.

The synthetic catalog mirrors the measured A10 layout: the dictionary and its key-value header
share a stem group, and the relation they describe lands in a group of its own.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import duckdb
import pytest

from sema.intake.land.loaders.tabular import land_tabular
from sema.intake.structure.shapes.base import SlotBinding
from tests.unit.intake.structure.shapes.shapes_env import OMOP_CSV, Source, warehouse_source
from tests.unit.intake.structure_env import (
    A2_BOUND,
    A2_WORKBOOK,
    A3_DB,
    ACRIS,
    bound_by_template,
    entry_over,
    fresh_tools,
    land,
    run,
    sql_lines,
)
from tests.unit.intake.verify.verify_env import (
    STRUCTURE_CHECKS,
    assert_named,
    assert_passes,
    tools_for,
)
from tests.unit.intake.verify.verify_env import verifier as verifier_module

pytestmark = pytest.mark.unit

SCHEMA = "dd"
DICTIONARY_STEM = "catalog"
DICTIONARY = f"{DICTIONARY_STEM}__columns"
HEADER = f"{DICTIONARY_STEM}__info"
DESCRIBED = "kinds"
SAME_GROUP_DESCRIBED = f"{DICTIONARY_STEM}__data"
DATA_COLUMNS = ("kind_code", "kind_label", "kind_note")
DATA_ROWS = (
    ("K1", "Alder", "Grows by rivers"),
    ("K2", "Birch", "Pale bark"),
    ("K3", "Cedar", "Scented wood"),
    ("K4", "Dogwood", "Spring flowers"),
    ("K5", "Elm", "Tall crown"),
)
HEADER_ROWS = (("Dataset Name", "Kinds catalog"), ("Owner", "Records team"), ("Updated", "Yearly"))
CASE_ONLY = ("KIND_CODE", "Kind_Label", "kind_NOTE")
SEPARATORS = ("Kind Code", "kind-label", "KindNote")
SUPERSET = (*DATA_COLUMNS, "kind_extra")
SUBSET = DATA_COLUMNS[:2]
DROPPED_MEMBERS = {
    "concept_list(relation=dd.kinds)",
    "entity_model(dictionary=dd.kinds, layout=per_file)",
}

Build = Callable[..., Source]


def _table(name: str, columns: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    values = ", ".join("(" + ", ".join(f"'{cell}'" for cell in row) + ")" for row in rows)
    return (
        f"CREATE TABLE {SCHEMA}.{name} AS SELECT * FROM (VALUES {values})"
        f" t({', '.join(columns)})"
    )


def _catalog_sql(described: str, property_names: Sequence[str]) -> tuple[str, ...]:
    dictionary_rows = [(name, f"Describes {name}", "Free text") for name in property_names]
    return (
        _table(DICTIONARY, ("column_name", "description", "notes"), dictionary_rows),
        _table(HEADER, ("field", "value"), HEADER_ROWS),
        _table(described, DATA_COLUMNS, DATA_ROWS),
    )


@pytest.fixture
def catalog(tmp_path: Path) -> Iterator[Build]:
    connections: list[duckdb.DuckDBPyConnection] = []

    def build(property_names: Sequence[str] = DATA_COLUMNS, described: str = DESCRIBED) -> Source:
        con = duckdb.connect(":memory:")
        connections.append(con)
        con.execute(f"CREATE SCHEMA {SCHEMA}")
        for statement in _catalog_sql(described, property_names):
            con.execute(statement)
        return warehouse_source(con, SCHEMA, tmp_path / "ledger" / "landing")

    yield build
    for con in connections:
        con.close()


def _structure(source: Source, ledger: Path) -> Any:
    return run(source, fresh_tools(source, ledger))


def _qualified(name: str) -> str:
    return f"{SCHEMA}.{name}"


def _assert_dictionary_bound(result: Any) -> None:
    entry = entry_over(result, _qualified(DICTIONARY))
    assert entry.binding.template == "entity_model"
    assert entry.binding.slots == {
        "layout": "per_file",
        "dictionary": _qualified(DICTIONARY),
        "header": _qualified(HEADER),
    }


def _assert_code_bound_list(result: Any, relation: str) -> None:
    entry = entry_over(result, relation)
    assert entry.binding.template == "concept_list"
    assert (entry.source, entry.confidence) == ("code", 1.0)


def test_relation_a_bound_dictionary_describes_binds_concept_list_by_code(
    catalog: Build, tmp_path: Path
) -> None:
    result = _structure(catalog(), tmp_path / "ledger" / "structure")

    assert result.ambiguities == (), [(a.subject, a.members) for a in result.ambiguities]
    _assert_code_bound_list(result, _qualified(DESCRIBED))
    _assert_dictionary_bound(result)


def test_described_relation_in_the_dictionary_group_binds_concept_list_by_code(
    catalog: Build, tmp_path: Path
) -> None:
    source = catalog(described=SAME_GROUP_DESCRIBED)

    result = _structure(source, tmp_path / "ledger" / "structure")

    assert result.ambiguities == (), [(a.subject, a.members) for a in result.ambiguities]
    _assert_code_bound_list(result, _qualified(SAME_GROUP_DESCRIBED))
    _assert_dictionary_bound(result)


@pytest.mark.parametrize("names", [CASE_ONLY, SEPARATORS], ids=["case_only", "separators"])
def test_property_names_compare_after_name_normalization(
    catalog: Build, tmp_path: Path, names: tuple[str, ...]
) -> None:
    result = _structure(catalog(names), tmp_path / "ledger" / "structure")

    assert result.ambiguities == (), [(a.subject, a.members) for a in result.ambiguities]
    _assert_code_bound_list(result, _qualified(DESCRIBED))


@pytest.mark.parametrize("names", [SUPERSET, SUBSET], ids=["superset", "subset"])
def test_partial_overlap_drops_no_candidate(
    catalog: Build, tmp_path: Path, names: tuple[str, ...]
) -> None:
    result = _structure(catalog(names), tmp_path / "ledger" / "structure")

    (ambiguity,) = result.ambiguities
    assert ambiguity.subject == _qualified(DESCRIBED)
    assert set(ambiguity.members) == DROPPED_MEMBERS
    _assert_dictionary_bound(result)


def test_rule_reads_schema_relation_text_only(catalog: Build, tmp_path: Path) -> None:
    ledger = tmp_path / "ledger" / "structure"

    result = _structure(catalog(), ledger)

    assert result.ambiguities == ()
    value_bearing = [line for line in sql_lines(ledger) if line["value_bearing"]]
    assert all(line["schema_relation"] for line in value_bearing), value_bearing
    assert not [line for line in value_bearing if DESCRIBED in line["statement"]]


def test_structure_verifier_passes_the_described_data_output(
    catalog: Build, tmp_path: Path
) -> None:
    source = catalog()
    result = _structure(source, tmp_path / "ledger" / "structure")
    tools = tools_for(source, tmp_path / "ledger" / "verify", stage="structure")

    verdict = verifier_module("structure").verify_structure(
        result, source.land, tools, provider=source.provider
    )

    assert_passes(verdict, STRUCTURE_CHECKS)


def test_pinned_binding_reading_described_data_as_a_dictionary_fails_the_verifier(
    catalog: Build, tmp_path: Path
) -> None:
    source = catalog()
    result = _structure(source, tmp_path / "ledger" / "structure")
    listed = entry_over(result, _qualified(DESCRIBED))
    slots = {"layout": "per_file", "dictionary": _qualified(DESCRIBED)}
    pinned = listed.model_copy(
        update={
            "binding": SlotBinding(
                template="entity_model", slots=slots, relations=(_qualified(DESCRIBED),)
            ),
            "source": "human",
        }
    )
    bound = tuple(pinned if entry is listed else entry for entry in result.bound)
    tools = tools_for(source, tmp_path / "ledger" / "verify", stage="structure")

    verdict = verifier_module("structure").verify_structure(
        result.model_copy(update={"bound": bound}), source.land, tools, provider=source.provider
    )

    found = [f for f in verdict.failures if f.check == "in_eligibility_set"]
    assert [(f.relation, f.evidence["described_by"]) for f in found] == [
        (_qualified(DESCRIBED), _qualified(DICTIONARY))
    ]
    assert_named(verdict, STRUCTURE_CHECKS)


@pytest.fixture(scope="module")
def acris_result(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[Source, Any]]:
    root = tmp_path_factory.mktemp("described_acris")
    staging, source = land(root, "acris", land_tabular, ACRIS)
    yield source, _structure(source, root / "ledger" / "structure")
    staging.close()


def test_a10_described_output_passes_the_structure_verifier(
    acris_result: tuple[Source, Any], tmp_path: Path
) -> None:
    source, result = acris_result
    tools = tools_for(source, tmp_path / "ledger" / "verify", stage="structure")

    verdict = verifier_module("structure").verify_structure(
        result, source.land, tools, provider=source.provider
    )

    assert_passes(verdict, STRUCTURE_CHECKS)


def test_a2_binding_is_unchanged(tmp_path: Path) -> None:
    staging, source = land(tmp_path, "a2", land_tabular, A2_WORKBOOK)
    result = _structure(source, tmp_path / "ledger" / "structure")
    staging.close()

    assert result.ambiguities == ()
    assert bound_by_template(result) == A2_BOUND


@pytest.mark.parametrize("loader", ["semsql", "athena_bundle"])
def test_loader_fixed_binding_is_unchanged(tmp_path: Path, loader: str) -> None:
    staging, source = land(tmp_path, loader, *_loader_fixed(loader, tmp_path))
    result = _structure(source, tmp_path / "ledger" / "structure")
    staging.close()

    assert result.ambiguities == ()
    assert {entry.binding.template for entry in result.bound} == {loader}


def _loader_fixed(loader: str, tmp_path: Path) -> tuple[Any, Path]:
    if loader == "semsql":
        from sema.intake.land.loaders.semsql import land_semsql

        return land_semsql, A3_DB
    from sema.intake.land.loaders.athena import land_athena_bundle
    from tests.support.athena_bundle import make_minimal_bundle

    bundle = tmp_path / "bundle"
    make_minimal_bundle(bundle)
    return land_athena_bundle, bundle


def test_a7_csv_dictionary_of_relations_not_landed_keeps_its_ambiguity(tmp_path: Path) -> None:
    staging, source = land(tmp_path, "omop", land_tabular, OMOP_CSV)
    result = _structure(source, tmp_path / "ledger" / "structure")
    staging.close()

    assert result.bound == ()
    (ambiguity,) = result.ambiguities
    templates = sorted({member.partition("(")[0] for member in ambiguity.members})
    assert templates == ["concept_with_values_denormalized", "entity_model"]
    assert any("layout=single_relation" in member for member in ambiguity.members)

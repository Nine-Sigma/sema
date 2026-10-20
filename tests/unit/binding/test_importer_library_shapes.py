"""Concept source importer against value shapes seen in a live library.

The flag column holds '1' / '0', column lists are Python-literal strings whose
entries can be SQL expressions, mapping codes are arrays, and the relationships
relation is large enough that it must be concept-scoped in SQL.
"""

from __future__ import annotations

from typing import Any

import duckdb
import pytest
import yaml

from sema.targets.adapters.manifest_parser import parse_manifest_raw
from tests.unit.binding.fixtures.concept_library import (
    CONCEPT_IDS,
    MULTI_CODES,
    RACE_ID,
    RACE_REF,
    SPEC_PATH,
    QueryFn,
    build_library,
    load_dataset_map,
    provider_for,
    query_fn,
    with_array_codes,
)

pytestmark = pytest.mark.unit

EXPRESSION_COLUMNS = "['coalesce(col_a, col_b)', 'col_c']"
RELATIONSHIP_FROM = " FROM lib.concept_relationship"


class _RecordingQuery:
    """Wraps the DuckDB query function and keeps every statement and its rows."""

    def __init__(self, run: QueryFn) -> None:
        self._run = run
        self.calls: list[tuple[str, list[tuple[Any, ...]]]] = []

    def __call__(self, sql: str) -> list[tuple[Any, ...]]:
        rows = self._run(sql)
        self.calls.append((sql, rows))
        return rows

    def relationship_calls(self) -> list[tuple[str, list[tuple[Any, ...]]]]:
        return [(sql, rows) for sql, rows in self.calls if RELATIONSHIP_FROM in sql]


def _import_from(con: duckdb.DuckDBPyConnection, query: QueryFn | None = None) -> Any:
    from sema.binding.importer import import_concept_source
    from sema.binding.spec import load_concept_source_spec

    spec = load_concept_source_spec(SPEC_PATH)
    try:
        return import_concept_source(
            spec, provider_for(con, query), dataset_map=load_dataset_map()
        )
    finally:
        con.close()


def _with_expression_binding(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        "INSERT INTO lib.column_bindings VALUES (?,?,?,?,?)",
        ["SYN_SEX", "Synthetic Labs", "ds-0003", EXPRESSION_COLUMNS, None],
    )


def _mapping(result: Any, source_value: str) -> dict[str, Any]:
    matches = [
        m.model_dump()
        for m in result.authored.value_mappings
        if m.source_value == source_value
    ]
    assert len(matches) == 1
    found: dict[str, Any] = matches[0]
    return found


def test_string_hierarchical_flags_import_as_booleans() -> None:
    result = _import_from(build_library())
    parsed = parse_manifest_raw(yaml.safe_load(yaml.safe_dump(result.manifest)))

    flags = {
        (r.from_.id, r.to.id, r.type): r.hierarchical for r in parsed.relationships
    }
    assert flags == {
        ("SYN_ETHNICITY", RACE_ID, "Is a"): True,
        (RACE_ID, "900999", "Maps to"): False,
    }


def test_python_literal_column_list_keeps_expressions_verbatim() -> None:
    con = build_library()
    _with_expression_binding(con)

    bindings = {b.concept_ref: b for b in _import_from(con).authored.column_bindings}

    assert bindings["syn:SYN_SEX"].columns == ["coalesce(col_a, col_b)", "col_c"]


def test_array_codes_import_as_one_mapping_with_every_code() -> None:
    con = build_library()
    with_array_codes(con)

    result = _import_from(con)

    assert len(result.authored.value_mappings) == 4
    mixed = _mapping(result, "mixed")
    assert mixed.get("codes") == MULTI_CODES
    assert mixed["concept_ref"] == RACE_REF
    assert mixed["approved"] is True
    assert _mapping(result, "caucasian").get("codes") == ["SYN_RACE_WHITE"]


def test_scalar_code_imports_as_one_element_list() -> None:
    result = _import_from(build_library())

    white, pending = _mapping(result, "white"), _mapping(result, "asian american")
    assert white.get("codes") == ["SYN_RACE_WHITE"]
    assert white["approved"] is True
    assert pending.get("codes") == ["SYN_RACE_ASIAN"]
    assert pending["approved"] is False


def test_relationship_query_carries_concept_scope_predicate() -> None:
    from sema.binding.importer_utils import in_concept_scope
    from sema.binding.spec import load_concept_source_spec

    spec = load_concept_source_spec(SPEC_PATH)
    con = build_library()
    recorder = _RecordingQuery(query_fn(con))
    _import_from(con, recorder)

    calls = recorder.relationship_calls()
    assert len(calls) == 1
    sql = calls[0][0]
    roles = spec.relationships.roles
    assert in_concept_scope(roles.from_.id, spec) in sql
    assert in_concept_scope(roles.to.id, spec) in sql
    # Scope comes from identifiers and the concepts subquery, never row values.
    assert not [concept_id for concept_id in CONCEPT_IDS if concept_id in sql]


def test_relationship_with_no_loaded_endpoint_is_never_fetched() -> None:
    con = build_library()
    recorder = _RecordingQuery(query_fn(con))
    _import_from(con, recorder)

    fetched = [row for _, rows in recorder.relationship_calls() for row in rows]
    assert fetched
    assert not [row for row in fetched if "SYN_OTHER" in row or "900500" in row]

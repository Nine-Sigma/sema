"""RelationProvider: a spec relation resolves to a warehouse or staging query."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest

from sema.binding.spec import ConceptSourceSpec, RelationRef
from sema.ingest.duckdb_staging import Staging

pytestmark = pytest.mark.unit

HOSTILE = "x'); DROP TABLE t; --"
FIXED_SELECT = re.compile(r"^SELECT [\w\", ]+ FROM [\w\".]+( WHERE .+)?$", re.DOTALL)


def _staging(tmp_path: Path) -> Staging:
    staging = Staging(str(tmp_path / "staging.duckdb"), schemas=("land",))
    rows = pa.table(
        {"code": ["A", "B"], "label": ["Alpha", HOSTILE], "parent": [None, "A"]}
    )
    staging.write_table(
        "land",
        "codes",
        rows,
        {"code": "VARCHAR", "label": "VARCHAR", "parent": "VARCHAR"},
        {},
        None,
    )
    return staging


def test_staging_ref_queries_the_staging_table(tmp_path: Path) -> None:
    from sema.intake.relations import RelationProvider

    staging = _staging(tmp_path)
    provider = RelationProvider(warehouse=None, staging=staging)

    query = provider.query(RelationRef(provider="staging", name="land.codes"))

    assert query("SELECT code FROM land.codes ORDER BY code") == [("A",), ("B",)]
    staging.close()


def test_warehouse_ref_calls_the_supplied_query_function() -> None:
    from sema.intake.relations import RelationProvider

    calls: list[str] = []

    def warehouse(sql: str) -> list[tuple[Any, ...]]:
        calls.append(sql)
        return [("row",)]

    provider = RelationProvider(warehouse=warehouse, staging=None)
    query = provider.query(RelationRef(provider="warehouse", name="cat.s.t"))

    assert query("SELECT a FROM cat.s.t") == [("row",)]
    assert calls == ["SELECT a FROM cat.s.t"]


def test_require_names_provider_and_first_relation_before_any_query() -> None:
    from sema.intake.relations import MissingProviderError, RelationProvider

    calls: list[str] = []

    def warehouse(sql: str) -> list[tuple[Any, ...]]:
        calls.append(sql)
        return []

    provider = RelationProvider(warehouse=warehouse, staging=None)
    refs = [
        RelationRef(provider="warehouse", name="cat.s.t"),
        RelationRef(provider="staging", name="land.first"),
        RelationRef(provider="staging", name="land.second"),
    ]

    with pytest.raises(MissingProviderError, match=r"staging.*land\.first"):
        provider.require(refs)
    assert calls == []


def test_query_text_stays_fixed_select_forms_for_hostile_cells(tmp_path: Path) -> None:
    from sema.binding.importer import import_concept_source
    from sema.intake.relations import RelationProvider

    staging = _staging(tmp_path)
    statements: list[str] = []
    run = staging.execute

    def recording(sql: str, params: Any = None) -> Any:
        statements.append(sql)
        return run(sql, params)

    staging.execute = recording  # type: ignore[method-assign]
    spec = ConceptSourceSpec.model_validate(
        {
            "spec_version": 2,
            "id": "codes",
            "identity_system": "ex",
            "release": "1",
            "default_provider": "staging",
            "scheme": {"qualified_name": "ex.codes", "description": "Codes"},
            "concepts": {
                "relation": "land.codes",
                "group_by": "code",
                "roles": {"concept_id": "code", "name": "label"},
            },
            "hierarchy": {"parent_column": "parent"},
        }
    )

    result = import_concept_source(
        spec, RelationProvider(warehouse=None, staging=staging), dataset_map={}
    )

    assert statements
    assert all(FIXED_SELECT.match(sql) for sql in statements), statements
    assert not [sql for sql in statements if "DROP" in sql]
    names = {p["name"] for e in result.manifest["entities"] for p in e["properties"]}
    assert HOSTILE in names
    staging.close()

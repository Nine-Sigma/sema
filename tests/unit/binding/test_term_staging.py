"""Term staging: value sets above the inline cap go to staging in the store shape."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest

from sema.binding.spec import ConceptSourceSpec
from sema.ingest.duckdb_staging import Staging

pytestmark = pytest.mark.unit

NAMESPACE = "vocabulary_ex_terms"
STATUS = {"column": "state", "active_values": ["active"], "retired_values": ["deprecated"]}
LIFECYCLE = {
    "status": STATUS,
    "valid_from": "valid_start",
    "valid_to": "valid_end",
    "replaced_by": "successor",
    "lang": "label_lang",
}
COLUMNS = (
    "concept_id",
    "code",
    "label",
    "synonym",
    "label_lang",
    "state",
    "valid_start",
    "valid_end",
    "successor",
)


def _value_rows(big: int = 6) -> list[tuple[Any, ...]]:
    rows: list[tuple[Any, ...]] = [
        ("BIG", f"B{i}", f"Big {i}", f"Large {i}", "en", "active", "2000-01-01", None, None)
        for i in range(1, big)
    ]
    rows.append(
        ("BIG", "B0", "Big old", None, "en", "deprecated", "1999-01-01", "2010-12-31", "B1")
    )
    rows += [
        ("SMALL", f"S{i}", f"Small {i}", None, "en", "active", None, None, None)
        for i in range(4)
    ]
    return rows


def _staging(path: Path, rows: list[tuple[Any, ...]]) -> Staging:
    staging = Staging(str(path), schemas=("land",))
    concepts = pa.table({"concept_id": ["BIG", "SMALL"], "label": ["Big", "Small"]})
    staging.write_table(
        "land", "concepts", concepts, {"concept_id": "VARCHAR", "label": "VARCHAR"}, {}, None
    )
    values = pa.table({name: [row[i] for row in rows] for i, name in enumerate(COLUMNS)})
    staging.write_table("land", "vals", values, {c: "VARCHAR" for c in COLUMNS}, {}, None)
    return staging


def _spec(lifecycle: bool = True, cap: int = 5) -> ConceptSourceSpec:
    roles: dict[str, Any] = {
        "concept_id": "concept_id",
        "code": "code",
        "display": "label",
        "synonyms": "synonym",
    }
    if lifecycle:
        roles.update(LIFECYCLE)
    return ConceptSourceSpec.model_validate(
        {
            "spec_version": 2,
            "id": "ex-terms",
            "identity_system": "ex",
            "release": "1",
            "default_provider": "staging",
            "scheme": {"qualified_name": "ex.terms", "description": "Terms"},
            "concepts": {
                "relation": "land.concepts",
                "group_by": "concept_id",
                "roles": {"concept_id": "concept_id", "name": "label"},
            },
            "value_sets": {"relation": "land.vals", "roles": roles},
            "term_storage": {"inline_cap": cap},
        }
    )


def _import(spec: ConceptSourceSpec, staging: Staging | None, source: Staging) -> Any:
    from sema.binding.importer import import_concept_source
    from sema.intake.relations import RelationProvider

    provider = RelationProvider(warehouse=None, staging=source)
    return import_concept_source(spec, provider, dataset_map={}, staging=staging)


def _fetch(staging: Staging, sql: str) -> list[tuple[Any, ...]]:
    rows: list[tuple[Any, ...]] = staging.execute(sql).fetchall()
    return rows


def _schemas(staging: Staging) -> set[str]:
    return {r[0] for r in _fetch(staging, "SELECT schema_name FROM duckdb_schemas()")}


class _Conn:
    def __init__(self, staging: Staging) -> None:
        self._staging = staging

    def execute(self, sql: str, parameters: Any = None) -> Any:
        return self._staging.execute(sql, parameters)


@pytest.fixture
def staging(tmp_path: Path) -> Staging:
    handle = _staging(tmp_path / "staging.duckdb", _value_rows())
    yield handle
    handle.close()


def test_value_set_above_the_cap_is_external_and_staged(staging: Staging) -> None:
    assert NAMESPACE not in _schemas(staging)

    result = _import(_spec(), staging, staging)

    vocabularies = {v["name"]: v["source"] for v in result.manifest["vocabularies"]}
    assert vocabularies == {"VS_BIG": "EXTERNAL", "VS_SMALL": "INLINE"}
    assert not [t for t in result.manifest["terms"] if t["vocabulary"] == "VS_BIG"]
    count = _fetch(staging, f"SELECT count(*) FROM {NAMESPACE}.concept")
    assert count == [(6,)]
    assert NAMESPACE in _schemas(staging)


def test_value_set_at_or_below_the_cap_stays_inline(staging: Staging) -> None:
    result = _import(_spec(), staging, staging)

    small = [t for t in result.manifest["terms"] if t["vocabulary"] == "VS_SMALL"]
    assert len(small) == 4
    staged = _fetch(
        staging, f"SELECT count(*) FROM {NAMESPACE}.concept WHERE vocabulary_id = 'VS_SMALL'"
    )
    assert staged == [(0,)]


def test_missing_staging_raises_naming_vocabulary_and_cap(staging: Staging) -> None:
    with pytest.raises(ValueError, match=r"VS_BIG.*inline_cap 5"):
        _import(_spec(), None, staging)


def test_shape_equals_the_showcase_omop_shape() -> None:
    from dataclasses import asdict

    from sema.binding.term_staging import TERM_STAGING_SCHEMA
    from showcase.cbioportal_to_omop.omop_policy import OMOP_VOCAB_SCHEMA

    assert asdict(TERM_STAGING_SCHEMA) == asdict(OMOP_VOCAB_SCHEMA)


def test_store_reads_the_namespace_with_a_null_standard_flag(staging: Staging) -> None:
    from sema.binding.term_staging import TERM_STAGING_SCHEMA
    from sema.resolve.vocab_store import VocabStore

    _import(_spec(), staging, staging)
    store = VocabStore(_Conn(staging), schema=TERM_STAGING_SCHEMA, namespace=NAMESPACE)

    row = store.concept_by_code("VS_BIG", "B2")

    assert row is not None
    assert (row.id, row.name, row.vocabulary, row.standard) == ("VS_BIG:B2", "Big 2", "VS_BIG", None)
    flags = _fetch(staging, f"SELECT DISTINCT standard_concept FROM {NAMESPACE}.concept")
    assert flags == [(None,)]


def test_retired_row_carries_invalid_reason_and_dates(staging: Staging) -> None:
    _import(_spec(), staging, staging)

    rows = _fetch(
        staging,
        "SELECT invalid_reason, valid_start_date, valid_end_date"
        f" FROM {NAMESPACE}.concept WHERE concept_code = 'B0'",
    )
    assert rows == [("deprecated", date(1999, 1, 1), date(2010, 12, 31))]


def test_replaced_by_yields_one_relationship_row(staging: Staging) -> None:
    _import(_spec(), staging, staging)

    rows = _fetch(
        staging,
        f"SELECT concept_id_1, concept_id_2, relationship_id FROM {NAMESPACE}.concept_relationship",
    )
    assert rows == [("VS_BIG:B0", "VS_BIG:B1", "replaced_by")]


def test_lang_is_carried_onto_staged_synonyms(staging: Staging) -> None:
    _import(_spec(), staging, staging)

    rows = _fetch(
        staging,
        "SELECT concept_synonym_name, language"
        f" FROM {NAMESPACE}.concept_synonym WHERE concept_id = 'VS_BIG:B1'",
    )
    assert rows == [("Large 1", "en")]


def test_no_lifecycle_roles_yields_null_reason_and_open_interval(staging: Staging) -> None:
    _import(_spec(lifecycle=False), staging, staging)

    rows = _fetch(
        staging,
        "SELECT DISTINCT invalid_reason, valid_start_date, valid_end_date"
        f" FROM {NAMESPACE}.concept",
    )
    assert rows == [(None, date(1970, 1, 1), date(2099, 12, 31))]


def test_second_import_replaces_the_namespace(tmp_path: Path) -> None:
    path = tmp_path / "staging.duckdb"
    first = _staging(path, _value_rows(big=6))
    _import(_spec(), first, first)
    first.close()
    second = _staging(path, _value_rows(big=8))

    _import(_spec(), second, second)

    codes = {r[0] for r in _fetch(second, f"SELECT concept_code FROM {NAMESPACE}.concept")}
    assert codes == {f"B{i}" for i in range(8)}
    second.close()


def test_failed_synonym_write_leaves_the_previous_tables(
    staging: Staging, monkeypatch: pytest.MonkeyPatch
) -> None:
    _import(_spec(), staging, staging)
    real = staging.write_table

    def failing(schema: str, table: str, *args: Any, **kwargs: Any) -> None:
        if schema == NAMESPACE and table == "concept_synonym":
            raise RuntimeError("synonym write failed")
        real(schema, table, *args, **kwargs)

    monkeypatch.setattr(staging, "write_table", failing)
    staging.execute("UPDATE land.vals SET label = 'Changed' WHERE code = 'B2'")

    with pytest.raises(RuntimeError, match="synonym write failed"):
        _import(_spec(), staging, staging)

    names = _fetch(
        staging, f"SELECT concept_name FROM {NAMESPACE}.concept WHERE concept_code = 'B2'"
    )
    assert names == [("Big 2",)]
    assert _fetch(staging, f"SELECT count(*) FROM {NAMESPACE}.concept") == [(6,)]
    assert _fetch(staging, f"SELECT count(*) FROM {NAMESPACE}.concept_synonym") == [(5,)]

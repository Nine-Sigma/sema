"""Done-when 2 (N3): `bind_declared` binds a declared join's entities to loaded relations.

The dictionary is the declared-join fixture's, read at `read_dictionary`'s new path; the counts
run on the fixture in DuckDB, through the DuckDB connector, in a `Counts` built as the
identity step builds one. Each declared join is found by the relation its referencing entry
describes, so no test depends on how the import spells an entity name.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from sema.connectors.duckdb import DuckDBConnector
from sema.engine.identity_utils import Counts
from sema.engine.warehouse_lookup import RelationEvidence
from sema.models.config import DuckDBConfig
from tests.support.connector_fixture_declared import (
    ACCEPTED,
    DECLARED_TABLES,
    MISSPELLED_PROPERTY,
    OPEN,
    REJECTED,
    DictionaryEntry,
    entry_with,
    write_declared_database,
    write_declared_dictionary,
)
from tests.support.connector_fixture_identity import IDENTITY_CATALOG, IDENTITY_SCHEMA

pytestmark = pytest.mark.unit


@dataclass
class Bound:
    dictionary: Any = None
    bindings: Any = None
    error: str | None = None

    def built(self) -> Bound:
        assert self.error is None, self.error
        return self

    def entry(self, name: str) -> DictionaryEntry:
        (entity,) = [e for e in self.dictionary.entities if e.name == name]
        return entry_with(entity.properties)

    def join_of(self, child: str) -> list[Any]:
        """The declared joins whose referencing entry describes `child`."""
        return [j for j in self.dictionary.declared_joins
                if self.entry(j.referencing).relation == child]

    def name_of(self, relation: str) -> str:
        (entity,) = [e.name for e in self.dictionary.entities
                     if entry_with(e.properties).relation == relation]
        return str(entity)


def _bind(directory: Path, wrap: Callable[[Any], Any] = lambda evidence: evidence) -> Bound:
    from sema.engine.declared_join_utils import bind_declared
    from sema.pipeline.dictionary_import_utils import read_dictionary

    database, files = write_declared_database(directory), write_declared_dictionary(directory)
    dictionary = read_dictionary(str(files.spec), str(files.staging))
    connector = DuckDBConnector(DuckDBConfig(path=str(database)))
    try:
        columns = {
            (IDENTITY_SCHEMA, table.name): [
                c.name for c in connector.extract_columns_dto(
                    IDENTITY_CATALOG, IDENTITY_SCHEMA, table.name)]
            for table in DECLARED_TABLES
        }
        counts = Counts(wrap(RelationEvidence(connector, IDENTITY_CATALOG)))
        bindings = bind_declared(dictionary.entities, dictionary.declared_joins, columns, counts)
    finally:
        connector.close()
    return Bound(dictionary, bindings)


@pytest.fixture(scope="module")
def bound(tmp_path_factory: pytest.TempPathFactory) -> Bound:
    try:
        return _bind(tmp_path_factory.mktemp("bindings"))
    except Exception as exc:  # each test reports the missing name or the crash as its failure
        return Bound(error=f"{type(exc).__name__}: {exc}")


def _parts(item: Any) -> tuple[Any, ...]:
    """An accepted entry: the join, the referencing and referenced relations, the NULL rows."""
    if dataclasses.is_dataclass(item):
        return tuple(getattr(item, f.name) for f in dataclasses.fields(item))
    return tuple(item)


def _accepted(bound: Bound, child: str) -> list[tuple[Any, ...]]:
    joins = bound.join_of(child)
    return [_parts(item) for item in bound.bindings.accepted if _parts(item)[0] in joins]


def _relation(table: str) -> tuple[str, str]:
    return (IDENTITY_SCHEMA, table)


def test_a_two_column_join_binds_both_entities_and_is_accepted(bound: Bound) -> None:
    built = bound.built()
    ((join, referencing, referenced, _),) = _accepted(built, "crate_line")

    assert tuple(referencing) == _relation("crate_line")
    assert tuple(referenced) == _relation(ACCEPTED["crate_line"])
    assert tuple(built.bindings.bound[join.referencing]) == _relation("crate_line")
    assert tuple(built.bindings.bound[join.referenced]) == _relation("crate_head")
    assert join not in built.bindings.open and join not in built.bindings.rejected


def test_the_tie_decoy_with_the_same_column_set_is_not_bound(bound: Bound) -> None:
    built = bound.built()

    assert _relation("crate_head_spare") not in {tuple(r) for r in built.bindings.bound.values()}
    assert all(tuple(_parts(item)[2]) != _relation("crate_head_spare")
               for item in built.bindings.accepted)
    assert _accepted(built, "crate_line"), "the true referenced relation is accepted"


def test_a_referenced_relation_whose_tuples_repeat_is_rejected(bound: Bound) -> None:
    built = bound.built()
    (join,) = built.join_of("bale_line")

    assert join in built.bindings.rejected
    assert join not in built.bindings.open
    assert not _accepted(built, "bale_line")
    assert join.referenced not in built.bindings.bound


def test_two_referenced_candidates_that_both_pass_are_open(bound: Bound) -> None:
    built = bound.built()
    (join,) = built.join_of(OPEN[0])

    assert join in built.bindings.open
    assert join not in built.bindings.rejected
    assert not _accepted(built, OPEN[0])
    assert join.referencing not in built.bindings.bound
    assert join.referenced not in built.bindings.bound


def test_a_null_part_on_the_referenced_side_still_holds_and_is_counted(bound: Bound) -> None:
    built = bound.built()
    ((_, _, referenced, null_rows),) = _accepted(built, "crate_line")
    seal = [t for t in DECLARED_TABLES if t.name == "crate_head"][0]
    index = seal.column_names.index("seal_day")

    assert tuple(referenced) == _relation("crate_head")
    assert null_rows == sum(row[index] is None for row in seal.rows) == 1


def test_a_misspelled_property_is_counted_and_blocks_nothing(bound: Bound) -> None:
    built = bound.built()
    name = built.name_of("crate_head")

    assert set(built.bindings.unplaced.get(name, ())) == {MISSPELLED_PROPERTY}
    assert tuple(built.bindings.bound[name]) == _relation("crate_head")
    others = {k: set(v) for k, v in built.bindings.unplaced.items() if k != name and set(v)}
    assert others == {}


def test_an_entity_two_accepted_joins_bind_to_two_relations_stays_unbound(bound: Bound) -> None:
    built = bound.built()
    name = built.name_of("pallet_east")

    for child in ("east_load", "west_load"):
        ((join, referencing, referenced, _),) = _accepted(built, child)
        assert join.referenced == name
        assert tuple(referencing) == _relation(child)
        assert tuple(referenced) == _relation(ACCEPTED[child])
        assert tuple(built.bindings.bound[join.referencing]) == _relation(child)
    assert name not in built.bindings.bound


def test_an_entity_under_nine_properties_with_one_misspelling_binds_nothing(
    bound: Bound,
) -> None:
    built = bound.built()
    name = built.name_of("cask_head")
    (join,) = built.join_of("cask_line")

    assert name not in built.bindings.bound
    assert _relation("cask_head") not in {tuple(r) for r in built.bindings.bound.values()}
    assert join in built.bindings.rejected and not _accepted(built, "cask_line")


def test_only_the_accepted_joins_bind_and_an_entity_in_no_join_is_not_bound(
    bound: Bound,
) -> None:
    built = bound.built()

    accepted = sorted(tuple(_parts(item)[1])[1] for item in built.bindings.accepted)
    assert accepted == sorted(ACCEPTED)
    assert sorted(built.entry(j.referencing).relation for j in built.bindings.rejected) == sorted(
        REJECTED)
    assert built.name_of("tray_stock") not in built.bindings.bound


class _ContainmentMissing:
    """The fixture's evidence, with no count for one relation's tuples within another's."""

    def __init__(self, evidence: RelationEvidence, relation: tuple[str, str]) -> None:
        self._evidence, self._relation = evidence, relation

    def contained(self, relation: Any, columns: Any, owner: Any, owner_columns: Any) -> Any:
        if tuple(relation) == self._relation and tuple(owner) != self._relation:
            return None
        return self._evidence.contained(relation, columns, owner, owner_columns)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._evidence, name)


def test_a_join_whose_containment_count_is_missing_is_open(tmp_path: Path) -> None:
    built = _bind(
        tmp_path, lambda evidence: _ContainmentMissing(evidence, _relation("crate_line")),
    ).built()
    (join,) = built.join_of("crate_line")

    assert join in built.bindings.open
    assert join not in built.bindings.rejected
    assert not _accepted(built, "crate_line")


class _ContainmentOnce:
    """The fixture's evidence, answering each containment of one relation within another once."""

    def __init__(self, evidence: RelationEvidence, relation: tuple[str, str]) -> None:
        self._evidence, self._relation = evidence, relation
        self._asked: set[tuple[str, ...]] = set()

    def contained(self, relation: Any, columns: Any, owner: Any, owner_columns: Any) -> Any:
        if tuple(relation) == self._relation and tuple(owner) != self._relation:
            if tuple(owner) in self._asked:
                return None
            self._asked.add(tuple(owner))
        return self._evidence.contained(relation, columns, owner, owner_columns)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._evidence, name)


def test_a_join_is_decided_on_one_containment_count_per_pair(tmp_path: Path) -> None:
    built = _bind(
        tmp_path, lambda evidence: _ContainmentOnce(evidence, _relation("crate_line")),
    ).built()
    (join,) = built.join_of("crate_line")

    assert join not in built.bindings.rejected
    assert [parts[2] for parts in _accepted(built, "crate_line")] == [
        _relation(ACCEPTED["crate_line"])]

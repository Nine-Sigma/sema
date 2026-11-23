"""Done-when 1 (N3): what the build keeps from intake's import.

`read_dictionary`, at its new path, reads the declared-join fixture's concept spec over its
staging file: each entity with its property names, each declared join with both entity names
and its ordered (local, remote) column pairs, and each inline value list, which also becomes a
code list under `<entity>.<property>` with one stated domain. The bare JSON form of a spec, as an
intake run leaves it, gives the same `Dictionary` as its `concept_source` form.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from tests.support.connector_fixture_codes import DictionaryFiles
from tests.support.connector_fixture_declared import (
    CODE_LIST_ROLES,
    CRATE_ENTRY,
    DICTIONARY_ENTRIES,
    TRAY_LIST,
    declared_spec,
    entry_with,
    write_declared_dictionary,
    write_intake_run,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def files(tmp_path: Path) -> DictionaryFiles:
    return write_declared_dictionary(tmp_path)


def _read(spec: Path, staging: Path) -> Any:
    from sema.pipeline.dictionary_import_utils import read_dictionary

    dictionary = read_dictionary(str(spec), str(staging))
    assert dictionary is not None
    return dictionary


def _by_name(dictionary: Any) -> dict[str, tuple[str, ...]]:
    return {entity.name: tuple(entity.properties) for entity in dictionary.entities}


def test_each_entity_is_kept_with_its_property_names_as_written(files: DictionaryFiles) -> None:
    from sema.engine.declared_join_utils import DictionaryEntity

    dictionary = _read(files.spec, files.staging)

    assert all(isinstance(e, DictionaryEntity) for e in dictionary.entities)
    assert len(dictionary.entities) == len(DICTIONARY_ENTRIES)
    assert sorted(sorted(props) for props in _by_name(dictionary).values()) == sorted(
        sorted(e.properties) for e in DICTIONARY_ENTRIES)


def test_each_declared_join_names_both_entities_and_its_ordered_column_pairs(
    files: DictionaryFiles,
) -> None:
    from sema.engine.declared_join_utils import DeclaredJoin

    dictionary = _read(files.spec, files.staging)
    names = _by_name(dictionary)

    declaring = [e for e in DICTIONARY_ENTRIES if e.joins_to]
    assert len(dictionary.declared_joins) == len(declaring)
    found = []
    for join in dictionary.declared_joins:
        assert isinstance(join, DeclaredJoin)
        referencing, referenced = entry_with(names[join.referencing]), entry_with(
            names[join.referenced])
        assert referencing.joins_to == referenced.entity
        assert tuple(tuple(pair) for pair in join.pairs) == referencing.pairs
        found.append(referencing.relation)
    assert sorted(found) == sorted(e.relation for e in declaring)


def test_each_inline_list_is_kept_with_its_entity_property_and_codes(
    files: DictionaryFiles,
) -> None:
    from sema.engine.declared_join_utils import InlineList

    dictionary = _read(files.spec, files.staging)
    names = _by_name(dictionary)

    lists = {(item.property): item for item in dictionary.inline_lists}
    assert all(isinstance(item, InlineList) for item in lists.values())
    assert set(lists) == set(CRATE_ENTRY.lists)
    for prop, item in lists.items():
        assert entry_with(names[item.entity]) is CRATE_ENTRY
        assert dict(item.codes) == CRATE_ENTRY.lists[prop]


def test_each_inline_list_is_a_code_list_with_one_stated_domain(files: DictionaryFiles) -> None:
    from sema.engine.coded_column_utils import StatedDomain

    dictionary = _read(files.spec, files.staging)

    for item in dictionary.inline_lists:
        name = f"{item.entity}.{item.property}"
        assert dictionary.code_lists[name] == dict(item.codes)
        assert [d for d in dictionary.domains if d.vocabulary == name] == [
            StatedDomain(item.entity, item.property, name)]


def test_equal_lists_on_two_properties_stay_two_lists(files: DictionaryFiles) -> None:
    dictionary = _read(files.spec, files.staging)

    equal = [item for item in dictionary.inline_lists
             if dict(item.codes) == CRATE_ENTRY.lists["lid_kind"]]
    assert sorted(item.property for item in equal) == ["lid_kind", "wrap_kind"]
    assert len({f"{item.entity}.{item.property}" for item in equal} & set(
        dictionary.code_lists)) == 2


def test_the_existing_fields_still_hold_the_code_list_and_its_roles(
    files: DictionaryFiles,
) -> None:
    dictionary = _read(files.spec, files.staging)

    assert dictionary.roles == CODE_LIST_ROLES
    assert dictionary.code_lists["tray_kinds"] == TRAY_LIST
    assert [d.vocabulary for d in dictionary.domains if d.property == "tray_kind"] == [
        "tray_kinds"]


def test_the_bare_json_form_gives_the_same_dictionary(files: DictionaryFiles,
                                                      tmp_path: Path) -> None:
    run = write_intake_run(tmp_path / "runs", stage="import", status="approved")

    bare = _read(run / "artifacts" / "spec.json", run / "staging.duckdb")

    assert bare == _read(files.spec, files.staging)
    assert bare.declared_joins and bare.inline_lists and bare.entities


def test_a_spec_with_no_entities_gives_three_empty_fields(files: DictionaryFiles,
                                                          tmp_path: Path) -> None:
    spec: dict[str, Any] = declared_spec()
    del spec["concept_source"]["entity_model"]
    path = tmp_path / "code_lists_only.yaml"
    path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")

    dictionary = _read(path, files.staging)

    assert list(dictionary.entities) == []
    assert list(dictionary.declared_joins) == []
    assert list(dictionary.inline_lists) == []
    assert dictionary.code_lists == {"tray_kinds": TRAY_LIST}

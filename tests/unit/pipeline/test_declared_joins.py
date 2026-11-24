"""Done-when 3 (N4): the assertions `run_declared_joins` returns for the declared joins.

The step runs over the declared-join fixture in DuckDB, through the real connector, with the
fixture's dictionary; only the graph is a stand-in (`StepGraph`). No model is called: the
model factory fails the test if the step asks for a client. Each expected payload is the one
`normalize_foreign_key` writes for a constraint on the same column pairs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from sema.models.assertions import Assertion, AssertionPredicate
from sema.models.extraction import ExtractedForeignKey
from sema.pipeline.normalizer import AssertionNormalizer
from tests.support.connector_fixture_declared import (
    ACCEPTED,
    DICTIONARY_ENTRIES,
    OPEN,
    REJECTED,
    prepare_declared_step,
)
from tests.support.connector_fixture_identity import IDENTITY_CATALOG, IDENTITY_SCHEMA
from tests.support.connector_fixture_identity_runs import (
    STEP_RUN_ID,
    StepRun,
    fixture_connector,
    run_step,
)

pytestmark = pytest.mark.unit

SOURCE = "dictionary_declared"


def _no_model(*_: Any, **__: Any) -> Any:
    pytest.fail("the declared-join step asked for a model client")


def _returned(prepared: StepRun, into: dict[str, Any]) -> None:
    from sema.pipeline.declared_joins import run_declared_joins

    into.update(run_declared_joins(prepared.graph, prepared.connector,  # type: ignore[arg-type]
                                   prepared.config, [IDENTITY_SCHEMA], STEP_RUN_ID))


def _run(directory: Path, **config: Any) -> tuple[StepRun, dict[str, Any]]:
    prepared, returned = prepare_declared_step(directory), dict[str, Any]()
    prepared.config = prepared.config.model_copy(update=config)
    with patch("sema.pipeline.build.LLMClientFactory.create", _no_model):
        run_step(prepared, lambda step: _returned(step, returned))
    return prepared, returned


@pytest.fixture(scope="module")
def declared(tmp_path_factory: pytest.TempPathFactory) -> tuple[StepRun, dict[str, Any]]:
    return _run(tmp_path_factory.mktemp("declared_joins"))


def _expected(prepared: StepRun, child: str) -> dict[str, Any]:
    """The payload `normalize_foreign_key` writes for the child's declared pairs."""
    (entry,) = [e for e in DICTIONARY_ENTRIES if e.relation == child]
    reader = fixture_connector(Path(prepared.config.duckdb.path))
    try:
        _, platform, workspace = reader.get_datasource_ref()
    finally:
        reader.close()
    key = ExtractedForeignKey(
        "declared", IDENTITY_CATALOG, IDENTITY_SCHEMA, child, IDENTITY_CATALOG,
        IDENTITY_SCHEMA, ACCEPTED[child], entry.pairs,
    )
    (assertion,) = AssertionNormalizer(workspace, STEP_RUN_ID, platform).normalize_foreign_key(key)
    return {**assertion.payload, "subject_ref": assertion.subject_ref,
            "object_ref": assertion.object_ref}


def _assertions(declared: tuple[StepRun, dict[str, Any]]) -> list[Assertion]:
    prepared, returned = declared
    assert prepared.error is None, prepared.error
    return [a for found in returned.values() for a in found]


def _referencing(assertion: Assertion) -> str:
    return str(assertion.payload["join_predicates"][0]["right_table"])


def test_one_joins_to_assertion_per_accepted_join_keyed_by_its_schema(
    declared: tuple[StepRun, dict[str, Any]],
) -> None:
    found = _assertions(declared)

    assert set(declared[1]) == {IDENTITY_SCHEMA}
    assert sorted(_referencing(a) for a in found) == sorted(ACCEPTED)
    assert all(a.predicate == AssertionPredicate.JOINS_TO for a in found)


def test_each_assertion_has_one_predicate_per_pair_and_the_declared_source(
    declared: tuple[StepRun, dict[str, Any]],
) -> None:
    prepared = declared[0]
    for assertion in _assertions(declared):
        expected = _expected(prepared, _referencing(assertion))
        assert assertion.source == SOURCE
        assert assertion.payload["join_predicates"] == expected["join_predicates"]
        assert len(assertion.payload["join_predicates"]) == 2
        for field in ("from_ref", "to_ref", "hop_count"):
            assert assertion.payload[field] == expected[field], field
        assert (assertion.subject_ref, assertion.object_ref) == (
            expected["subject_ref"], expected["object_ref"])
        assert "cardinality" not in assertion.payload
        assert assertion.confidence == 0.95


def test_no_assertion_for_a_rejected_or_open_join_or_onto_the_tie_decoy(
    declared: tuple[StepRun, dict[str, Any]],
) -> None:
    found = _assertions(declared)

    assert {_referencing(a) for a in found}.isdisjoint({*REJECTED, *OPEN})
    referenced = {str(a.payload["join_predicates"][0]["left_table"]) for a in found}
    assert referenced == set(ACCEPTED.values())


def test_a_build_that_names_no_dictionary_gets_none(tmp_path: Path) -> None:
    prepared, returned = _run(tmp_path, concept_spec=None, concept_staging=None)

    assert prepared.error is None, prepared.error
    assert returned == {}


def test_a_build_with_join_detection_off_gets_none(tmp_path: Path) -> None:
    prepared, returned = _run(tmp_path, enable_fk_detection=False)

    assert prepared.error is None, prepared.error
    assert returned == {}


class _EveryCountHolds:
    """Counts where every relation has two rows, none NULL, and every tuple is held."""

    def __init__(self) -> None:
        self.evidence = self
        self.asked: list[tuple[tuple[str, str], tuple[str, ...]]] = []

    def rows(self, relation: Any) -> int:
        return 2

    def null_rows(self, relation: Any, columns: Any) -> int:
        return 0

    def contained(self, relation: Any, columns: Any, owner: Any, owner_columns: Any) -> Any:
        self.asked.append((tuple(relation), tuple(columns)))
        return (2, 2)


def test_a_column_spelled_two_ways_is_written_as_the_one_checked() -> None:
    from sema.engine.declared_join_utils import DeclaredJoin, DictionaryEntity, bind_declared
    from sema.pipeline.declared_joins import _foreign_key

    child, parent = ("land", "owner_line"), ("land", "owner")
    columns = {child: ["owner id", "owner_id", "line_no"], parent: ["id", "name"]}
    join = DeclaredJoin("Owner Line", "Owner", (("owner_id", "id"),))
    counts = _EveryCountHolds()

    bindings = bind_declared(
        [DictionaryEntity("Owner Line", ("owner_id", "line_no")),
         DictionaryEntity("Owner", ("id", "name"))],
        [join], columns, counts,  # type: ignore[arg-type]
    )
    key = _foreign_key("cat", join, (child, columns[child]), (parent, columns[parent]))

    assert [j for j, *_ in bindings.accepted] == [join]
    (checked,) = {names for relation, names in counts.asked if relation == child}
    assert tuple(local for local, _ in key.column_pairs) == checked

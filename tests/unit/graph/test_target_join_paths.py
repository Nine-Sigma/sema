"""Unit tests for manifest join paths: predicate shape, sources, skip counts."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from sema.consumers.nl2sql.prompting_utils import _format_join_predicates
from sema.graph.target_join_paths import (
    DECLARED_JOIN_CONFIDENCE,
    materialize_target_join_paths,
)
from sema.graph.target_retrieval_loader import (
    EntityTableBinding,
    PhysicalSchemaBinding,
)
from sema.models.context import JoinPath
from sema.models.planner._enums import PrimaryKeyStrategy, TargetArtifactKind
from sema.models.planner.target_model import ForeignKeyObligation
from sema.models.target.completeness import (
    SemanticCompleteness,
    SemanticCompletenessAnnotations,
)
from sema.models.target.declared import DeclaredJoinDecl
from sema.models.target.descriptor import TargetModelDescriptor
from sema.models.target.entity import TargetEntityDecl
from sema.models.target.normalized import NormalizedTargetModel
from sema.models.target.obligation import TargetObligationDecl
from sema.models.target.refs import TargetEntityRef
from sema.pipeline import context_utils
from sema.pipeline.context_utils import _parse_join_predicates

pytestmark = pytest.mark.unit

_CATALOG = "cat"
_SCHEMA = "sch"
_PARENT, _CHILD, _OTHER = "ns.parent", "ns.child", "ns.other"
_TABLES = {_PARENT: "parent_t", _CHILD: "child_t", _OTHER: "other_t"}
_KEYS = [("doc_id", "doc_id"), ("as_of", "as_of_date")]


def _ref(table: str) -> str:
    return f"{_CATALOG}.{_SCHEMA}.{table}"


def _entity(
    name: str, joins: list[DeclaredJoinDecl] | None = None
) -> TargetEntityDecl:
    ref = TargetEntityRef(
        target_model_id="m", qualified_name=name,
        kind=TargetArtifactKind.TABLE_ROW,
    )
    return TargetEntityDecl(ref=ref, declared_joins=joins or [])


def _declared(referenced: str, keys: list[tuple[str, str]]) -> DeclaredJoinDecl:
    return DeclaredJoinDecl(
        referenced_entity=referenced, join_keys=keys, evidence="declared"
    )


def _obligation(entity: str, referenced: str) -> TargetObligationDecl:
    return TargetObligationDecl(
        target_entity=entity, required_fields=["doc_id"],
        primary_key=PrimaryKeyStrategy.NATURAL_KEY,
        foreign_keys=[
            ForeignKeyObligation(referenced_entity=referenced, join_keys=_KEYS)
        ],
    )


def _model(
    entities: list[TargetEntityDecl],
    obligations: list[TargetObligationDecl] | None = None,
) -> NormalizedTargetModel:
    return NormalizedTargetModel(
        descriptor=TargetModelDescriptor(
            target_model_id="m", target_model_version="1", display_name="M",
            completeness=SemanticCompletenessAnnotations(
                structure=SemanticCompleteness.COMPLETE,
                obligations=SemanticCompleteness.NONE,
                vocabulary_bindings=SemanticCompleteness.NONE,
                semantic_aliases=SemanticCompleteness.NONE,
                terms=SemanticCompleteness.NONE,
            ),
        ),
        entities=entities,
        obligations=obligations or [],
    )


def _binding(*names: str) -> PhysicalSchemaBinding:
    return PhysicalSchemaBinding(
        catalog=_CATALOG, schema=_SCHEMA,
        entity_tables=tuple(EntityTableBinding(n, _TABLES[n]) for n in names),
    )


def _upserts(loader: MagicMock) -> list[dict[str, Any]]:
    calls = []
    for call in loader.upsert_join_path.call_args_list:
        args = dict(zip(("name", "join_predicates"), call.args))
        calls.append({**args, **call.kwargs})
    return calls


def _uses_by_path(loader: MagicMock) -> dict[str, set[str]]:
    uses: dict[str, set[str]] = {}
    for call in loader.add_join_path_uses.call_args_list:
        name = call.kwargs.get("join_path_name", call.args[0] if call.args else None)
        ref = call.kwargs.get("table_ref", call.args[1] if len(call.args) > 1 else None)
        uses.setdefault(name, set()).add(ref)
    return uses


def _fk_model() -> NormalizedTargetModel:
    return _model(
        [_entity(_PARENT), _entity(_CHILD)], [_obligation(_CHILD, _PARENT)]
    )


def _declared_model() -> NormalizedTargetModel:
    return _model([_entity(_PARENT), _entity(_CHILD, [_declared(_PARENT, _KEYS)])])


def _expected_predicates(entity: str, referenced: str) -> list[dict[str, str]]:
    return [
        {
            "left_table": _ref(_TABLES[entity]), "left_column": local,
            "right_table": _ref(_TABLES[referenced]), "right_column": remote,
            "operator": "=",
        }
        for local, remote in _KEYS
    ]


def test_declared_join_confidence_is_the_policy_constant() -> None:
    assert DECLARED_JOIN_CONFIDENCE == 0.8
    assert DECLARED_JOIN_CONFIDENCE < 1.0
    assert DECLARED_JOIN_CONFIDENCE > context_utils._SEMANTIC_CONFIDENCE_THRESHOLD
    assert DECLARED_JOIN_CONFIDENCE > context_utils._STRUCTURAL_CONFIDENCE_THRESHOLD


def test_obligation_foreign_key_writes_manifest_path() -> None:
    loader = MagicMock()
    result = materialize_target_join_paths(
        loader, _fk_model(), _binding(_PARENT, _CHILD)
    )
    [upsert] = _upserts(loader)
    assert upsert["source"] == "manifest"
    assert upsert["confidence"] == 1.0
    assert upsert["source_schema"] == _SCHEMA
    assert (result.written, result.skipped) == (1, 0)


def test_declared_join_writes_manifest_declared_path() -> None:
    loader = MagicMock()
    result = materialize_target_join_paths(
        loader, _declared_model(), _binding(_PARENT, _CHILD)
    )
    [upsert] = _upserts(loader)
    assert upsert["source"] == "manifest_declared"
    assert upsert["confidence"] == DECLARED_JOIN_CONFIDENCE
    assert upsert["source_schema"] == _SCHEMA
    assert (result.written, result.skipped) == (1, 0)


def test_one_path_per_join_each_using_both_tables() -> None:
    model = _model(
        [
            _entity(_PARENT),
            _entity(_CHILD, [_declared(_OTHER, [("other_id", "other_id")])]),
            _entity(_OTHER, [_declared(_PARENT, _KEYS)]),
        ],
        [_obligation(_CHILD, _PARENT)],
    )
    loader = MagicMock()
    result = materialize_target_join_paths(
        loader, model, _binding(_PARENT, _CHILD, _OTHER)
    )
    upserts = _upserts(loader)
    names = [u["name"] for u in upserts]
    assert len(names) == len(set(names)) == 3
    assert sorted(u["source"] for u in upserts) == [
        "manifest", "manifest_declared", "manifest_declared",
    ]
    uses = _uses_by_path(loader)
    assert all(len(uses.get(n, set())) == 2 for n in names)
    assert result.written == 3


def test_two_declared_joins_between_one_pair_get_distinct_paths() -> None:
    joins = [_declared(_PARENT, [("doc_id", "doc_id")]),
             _declared(_PARENT, [("alt_id", "alt_id")])]
    model = _model([_entity(_PARENT), _entity(_CHILD, joins)])
    loader = MagicMock()
    result = materialize_target_join_paths(loader, model, _binding(_PARENT, _CHILD))
    names = [u["name"] for u in _upserts(loader)]
    assert len(names) == len(set(names)) == 2
    assert result.written == 2


@pytest.mark.parametrize("model", [_fk_model(), _declared_model()],
                         ids=["obligation", "declared"])
def test_predicates_round_trip_through_retrieval_parse(
    model: NormalizedTargetModel,
) -> None:
    loader = MagicMock()
    materialize_target_join_paths(loader, model, _binding(_PARENT, _CHILD))
    [upsert] = _upserts(loader)
    assert upsert["join_predicates"] == _expected_predicates(_CHILD, _PARENT)
    parsed = _parse_join_predicates(upsert["join_predicates"])
    assert len(parsed) == len(_KEYS)
    for predicate in parsed:
        assert predicate.left_table and predicate.left_column
        assert predicate.right_table and predicate.right_column
        assert predicate.operator == "="


@pytest.mark.parametrize("model", [_fk_model(), _declared_model()],
                         ids=["obligation", "declared"])
def test_nl2sql_join_text_names_both_columns(model: NormalizedTargetModel) -> None:
    loader = MagicMock()
    materialize_target_join_paths(loader, model, _binding(_PARENT, _CHILD))
    [upsert] = _upserts(loader)
    path = JoinPath(
        from_table=_TABLES[_CHILD], to_table=_TABLES[_PARENT],
        join_predicates=_parse_join_predicates(upsert["join_predicates"]),
        confidence=upsert["confidence"],
    )
    text = _format_join_predicates(path)
    child, parent = _ref(_TABLES[_CHILD]), _ref(_TABLES[_PARENT])
    assert text == (
        f"{child}.doc_id = {parent}.doc_id AND "
        f"{child}.as_of = {parent}.as_of_date"
    )


def test_entity_without_binding_is_skipped_and_counted() -> None:
    joins = [_declared(_PARENT, _KEYS), _declared(_OTHER, [("other_id", "other_id")])]
    model = _model([_entity(_PARENT), _entity(_OTHER), _entity(_CHILD, joins)])
    loader = MagicMock()
    result = materialize_target_join_paths(loader, model, _binding(_PARENT, _OTHER))
    loader.upsert_join_path.assert_not_called()
    loader.add_join_path_uses.assert_not_called()
    assert (result.written, result.skipped) == (0, 2)


def test_referenced_entity_without_binding_is_skipped_and_counted() -> None:
    model = _model(
        [_entity(_PARENT), _entity(_CHILD, [_declared(_OTHER, _KEYS)])],
        [_obligation(_CHILD, _PARENT)],
    )
    loader = MagicMock()
    result = materialize_target_join_paths(loader, model, _binding(_PARENT, _CHILD))
    assert [u["source"] for u in _upserts(loader)] == ["manifest"]
    assert (result.written, result.skipped) == (1, 1)


def test_empty_binding_writes_nothing_and_counts_every_join() -> None:
    model = _model(
        [_entity(_PARENT), _entity(_CHILD, [_declared(_PARENT, _KEYS)])],
        [_obligation(_CHILD, _PARENT)],
    )
    loader = MagicMock()
    result = materialize_target_join_paths(loader, model, _binding())
    loader.upsert_join_path.assert_not_called()
    assert (result.written, result.skipped) == (0, 2)

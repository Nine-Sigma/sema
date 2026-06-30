"""Shared fixtures for `sema.targets` unit tests."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import pytest

from sema.models.planner._enums import (
    PrimaryKeyStrategy,
    TargetArtifactKind,
)
from sema.models.target.completeness import (
    SemanticCompleteness,
    SemanticCompletenessAnnotations,
)
from sema.models.target.context_card import TargetContextCard
from sema.models.target.descriptor import TargetModelDescriptor
from sema.models.target.endpoints import EdgeEndpointDecl, EdgeEndpointsDecl
from sema.models.target.entity import TargetEntityDecl
from sema.models.target.obligation import TargetObligationDecl
from sema.models.target.properties import TargetPropertyDecl
from sema.models.target.refs import (
    TargetEntityRef,
    TargetPropertyRef,
    VocabularyRef,
    VocabularySource,
)
from sema.models.target.term import TargetTermDecl
from sema.models.target.vocab_binding import VocabularyBindingDecl
from sema.targets import register_target_adapter
from sema.targets.registry import _clear_for_tests


def _make_descriptor(
    target_model_id: str = "fake-target",
    completeness: SemanticCompletenessAnnotations | None = None,
) -> TargetModelDescriptor:
    return TargetModelDescriptor(
        target_model_id=target_model_id,
        target_model_version="1.0.0",
        display_name="Fake Target",
        completeness=completeness or default_completeness(),
    )


def default_completeness() -> SemanticCompletenessAnnotations:
    return SemanticCompletenessAnnotations(
        structure=SemanticCompleteness.COMPLETE,
        obligations=SemanticCompleteness.COMPLETE,
        vocabulary_bindings=SemanticCompleteness.PARTIAL,
        semantic_aliases=SemanticCompleteness.PARTIAL,
        terms=SemanticCompleteness.EXTERNAL,
    )


def _make_entity_ref(
    target_model_id: str = "fake-target",
    qualified_name: str = "fake.person",
    kind: TargetArtifactKind = TargetArtifactKind.TABLE_ROW,
) -> TargetEntityRef:
    return TargetEntityRef(
        target_model_id=target_model_id, qualified_name=qualified_name, kind=kind
    )


class FakeAdapter:
    """Minimal conforming adapter used across protocol/registry tests."""

    def __init__(self, target_model_id: str = "fake-target") -> None:
        self._descriptor = _make_descriptor(target_model_id)
        self._entity_ref = _make_entity_ref(target_model_id=target_model_id)
        self._entity_decl = TargetEntityDecl(
            ref=self._entity_ref,
            properties=[TargetPropertyDecl(name="person_id", type="string", nullable=False)],
        )
        self._obligation = TargetObligationDecl(
            target_entity=self._entity_ref.qualified_name,
            required_fields=["person_id"],
            primary_key=PrimaryKeyStrategy.NATURAL_KEY,
        )

    def describe(self) -> TargetModelDescriptor:
        return self._descriptor

    def discover_entities(self) -> Iterable[TargetEntityRef]:
        return [self._entity_ref]

    def load_entity(self, ref: TargetEntityRef) -> TargetEntityDecl:
        return self._entity_decl

    def load_obligation(self, ref: TargetEntityRef) -> TargetObligationDecl:
        return self._obligation

    def load_vocabulary_bindings(
        self, ref: TargetPropertyRef
    ) -> Iterable[VocabularyBindingDecl]:
        return []

    def load_context_card(self, ref: TargetEntityRef) -> TargetContextCard:
        return TargetContextCard(
            entity_ref=ref,
            card_version="1.0.0",
            description=f"Fake card for {ref.qualified_name}",
        )

    def iter_terms(self, vocabulary_ref: VocabularyRef) -> Iterator[TargetTermDecl]:
        raise NotImplementedError("EXTERNAL terms; adapter does not inline")


class ScriptedAdapter:
    """Adapter built from explicit DTO collections, for normalizer/loader tests."""

    def __init__(
        self,
        descriptor: TargetModelDescriptor,
        entities: list[TargetEntityDecl],
        obligations: list[TargetObligationDecl],
        bindings: list[VocabularyBindingDecl] | None = None,
        terms: list[TargetTermDecl] | None = None,
        cards: list[TargetContextCard] | None = None,
    ) -> None:
        self._descriptor = descriptor
        self._entities = list(entities)
        self._entities_by_ref = {e.ref: e for e in entities}
        self._obligations = {o.target_entity: o for o in obligations}
        self._bindings = list(bindings or [])
        self._terms = list(terms or [])
        self._cards = list(cards or [])
        self._cards_by_ref = {c.entity_ref: c for c in self._cards}

    def describe(self) -> TargetModelDescriptor:
        return self._descriptor

    def discover_entities(self) -> Iterable[TargetEntityRef]:
        return [e.ref for e in self._entities]

    def load_entity(self, ref: TargetEntityRef) -> TargetEntityDecl:
        return self._entities_by_ref[ref]

    def load_obligation(self, ref: TargetEntityRef) -> TargetObligationDecl:
        return self._obligations[ref.qualified_name]

    def load_vocabulary_bindings(
        self, ref: TargetPropertyRef
    ) -> Iterable[VocabularyBindingDecl]:
        return [
            b
            for b in self._bindings
            if b.entity_ref == ref.entity_ref and b.property_name == ref.property_name
        ]

    def load_context_card(self, ref: TargetEntityRef) -> TargetContextCard:
        if ref in self._cards_by_ref:
            return self._cards_by_ref[ref]
        return TargetContextCard(
            entity_ref=ref,
            card_version="0.0.0+synthesized",
            description=f"Auto-generated card for {ref.qualified_name}.",
        )

    def iter_terms(self, vocabulary_ref: VocabularyRef) -> Iterator[TargetTermDecl]:
        matching = [t for t in self._terms if t.vocabulary.name == vocabulary_ref.name]
        if not matching:
            raise NotImplementedError(
                f"adapter does not inline terms for vocabulary {vocabulary_ref.name!r}"
            )
        return iter(matching)


def make_table_row_entity(
    target_model_id: str = "fake-target",
    qualified_name: str = "fake.person",
    properties: list[TargetPropertyDecl] | None = None,
    completeness: SemanticCompletenessAnnotations | None = None,
) -> TargetEntityDecl:
    ref = _make_entity_ref(
        target_model_id=target_model_id,
        qualified_name=qualified_name,
        kind=TargetArtifactKind.TABLE_ROW,
    )
    return TargetEntityDecl(
        ref=ref,
        properties=properties or [TargetPropertyDecl(name="person_id", type="string", nullable=False)],
        completeness=completeness,
    )


def make_graph_node_entity(
    target_model_id: str = "fake-target",
    qualified_name: str = "fake.LLC",
    properties: list[TargetPropertyDecl] | None = None,
) -> TargetEntityDecl:
    ref = _make_entity_ref(
        target_model_id=target_model_id,
        qualified_name=qualified_name,
        kind=TargetArtifactKind.GRAPH_NODE,
    )
    return TargetEntityDecl(
        ref=ref,
        properties=properties or [TargetPropertyDecl(name="name", type="string", nullable=False)],
    )


def make_graph_edge_entity(
    target_model_id: str,
    qualified_name: str,
    subject_target: TargetEntityRef,
    object_target: TargetEntityRef,
    columnar_properties: list[TargetPropertyDecl] | None = None,
) -> TargetEntityDecl:
    ref = _make_entity_ref(
        target_model_id=target_model_id,
        qualified_name=qualified_name,
        kind=TargetArtifactKind.GRAPH_EDGE,
    )
    endpoints = EdgeEndpointsDecl(
        subject=EdgeEndpointDecl(role="subject", target_entity=subject_target),
        object=EdgeEndpointDecl(role="object", target_entity=object_target),
    )
    return TargetEntityDecl(
        ref=ref,
        properties=columnar_properties or [],
        endpoints=endpoints,
    )


@pytest.fixture
def fake_adapter_cls() -> type[FakeAdapter]:
    return FakeAdapter


@pytest.fixture(autouse=True)
def _isolate_registry() -> Iterator[None]:
    _clear_for_tests()
    yield
    _clear_for_tests()


def make_descriptor(
    target_model_id: str = "fake-target",
    completeness: SemanticCompletenessAnnotations | None = None,
) -> TargetModelDescriptor:
    return _make_descriptor(target_model_id, completeness=completeness)


def make_entity_ref(
    target_model_id: str = "fake-target",
    qualified_name: str = "fake.person",
    kind: TargetArtifactKind = TargetArtifactKind.TABLE_ROW,
) -> TargetEntityRef:
    return _make_entity_ref(target_model_id, qualified_name, kind)


def make_obligation(
    target_entity: str = "fake.person",
    required_fields: list[str] | None = None,
    pk: PrimaryKeyStrategy = PrimaryKeyStrategy.NATURAL_KEY,
) -> TargetObligationDecl:
    return TargetObligationDecl(
        target_entity=target_entity,
        required_fields=required_fields or ["person_id"],
        primary_key=pk,
    )


__all__ = [
    "FakeAdapter",
    "ScriptedAdapter",
    "make_descriptor",
    "make_entity_ref",
    "make_obligation",
    "make_table_row_entity",
    "make_graph_node_entity",
    "make_graph_edge_entity",
    "default_completeness",
    "register_target_adapter",
    "VocabularyRef",
    "VocabularySource",
]

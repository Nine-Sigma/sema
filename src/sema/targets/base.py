"""TargetOntologyAdapter protocol surface."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Protocol, runtime_checkable

from sema.models.target.context_card import TargetContextCard
from sema.models.target.descriptor import TargetModelDescriptor
from sema.models.target.entity import TargetEntityDecl
from sema.models.target.obligation import TargetObligationDecl
from sema.models.target.refs import TargetEntityRef, TargetPropertyRef, VocabularyRef
from sema.models.target.term import TargetTermDecl
from sema.models.target.vocab_binding import VocabularyBindingDecl


@runtime_checkable
class TargetOntologyAdapter(Protocol):
    """Declarative loader for a target ontology.

    Adapters emit DTOs; they MUST NOT call the matching engine, planner,
    or constraint layer, MUST NOT call any LLM, and MUST NOT mutate the
    graph. Snapshot hashing is owned by `SnapshotHasher`; adapters MUST
    NOT compute or return snapshot hashes.
    """

    def describe(self) -> TargetModelDescriptor: ...

    def discover_entities(self) -> Iterable[TargetEntityRef]: ...

    def load_entity(self, ref: TargetEntityRef) -> TargetEntityDecl: ...

    def load_obligation(self, ref: TargetEntityRef) -> TargetObligationDecl: ...

    def load_vocabulary_bindings(
        self, ref: TargetPropertyRef
    ) -> Iterable[VocabularyBindingDecl]: ...

    def load_context_card(self, ref: TargetEntityRef) -> TargetContextCard: ...

    def iter_terms(self, vocabulary_ref: VocabularyRef) -> Iterator[TargetTermDecl]: ...


class TargetOntologyAdapterMixin:
    """Default-implementation mixin for adapters that do not inline terms."""

    def iter_terms(self, vocabulary_ref: VocabularyRef) -> Iterator[TargetTermDecl]:
        raise NotImplementedError("EXTERNAL terms; adapter does not inline")


REQUIRED_METHODS: tuple[str, ...] = (
    "describe",
    "discover_entities",
    "load_entity",
    "load_obligation",
    "load_vocabulary_bindings",
    "load_context_card",
)

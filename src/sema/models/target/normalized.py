"""NormalizedTargetModel — sorted, cross-resolved adapter output."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from sema.models.target.context_card import TargetContextCard
from sema.models.target.descriptor import TargetModelDescriptor
from sema.models.target.entity import TargetEntityDecl
from sema.models.target.obligation import TargetObligationDecl
from sema.models.target.refs import VocabularyRef
from sema.models.target.term import TargetTermDecl
from sema.models.target.vocab_binding import VocabularyBindingDecl


class NormalizedTargetModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    descriptor: TargetModelDescriptor
    entities: list[TargetEntityDecl] = Field(default_factory=list)
    obligations: list[TargetObligationDecl] = Field(default_factory=list)
    vocabularies: list[VocabularyRef] = Field(default_factory=list)
    vocabulary_bindings: list[VocabularyBindingDecl] = Field(default_factory=list)
    terms: list[TargetTermDecl] = Field(default_factory=list)
    context_cards: list[TargetContextCard] = Field(default_factory=list)

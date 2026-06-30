"""Vocabulary-agnostic resolver policy (US-004).

A :class:`ResolverPolicy` captures the SOURCE-side resolution rules — which
source vocabulary a code is matched in, which relationship standardizes to the
target, and which flag marks a standard concept — while reading the TARGET-side
obligation (domain, ``require_standard``, ``allow_zero_default``) from the
loaded :class:`~sema.models.target.vocab_binding.VocabularyBindingDecl`. The
binding's fields are read, never duplicated, so source-vs-target conflation
(R9) is structurally impossible: the policy names the SOURCE vocabulary while
the binding's singular ``vocabulary`` is the TARGET.

This module is generic: no OMOP/OncoTree literal appears here. The concrete
instance lives in :mod:`sema.resolve.policies.omop`, the only place those
literals may appear.
"""

from __future__ import annotations

from dataclasses import dataclass

from sema.models.target.vocab_binding import VocabularyBindingDecl


@dataclass(frozen=True)
class Candidate:
    """A target-vocabulary candidate row, reduced to policy-relevant fields.

    ``standard_value`` is the row's standard-flag value (compared against the
    policy's ``standard_flag``); ``is_invalid`` is True when the row carries any
    invalid reason; ``domain`` is the row's domain (gated against the binding's
    target domain).
    """

    standard_value: str | None
    is_invalid: bool
    domain: str | None


@dataclass(frozen=True)
class ResolverPolicy:
    """Source-side resolution rules bound to a target obligation.

    Source-side literals (``source_vocabulary``, ``maps_to_relationship``,
    ``standard_flag``) are policy data; target-side obligations are read from
    ``binding`` so they are never duplicated.
    """

    source_vocabulary: str
    maps_to_relationship: str
    standard_flag: str
    binding: VocabularyBindingDecl

    @property
    def target_domain(self) -> str | None:
        return self.binding.domain

    @property
    def require_standard(self) -> bool:
        return self.binding.require_standard

    @property
    def allow_zero_default(self) -> bool:
        return self.binding.allow_zero_default

    def candidate_lookup(self, source_code: str) -> tuple[str, str]:
        """Key for matching a source code at candidate-generation time (R9).

        Returns ``(source_vocabulary, source_code)`` — the code is matched in
        the SOURCE vocabulary, never the binding's target vocabulary.
        """
        return self.source_vocabulary, source_code

    def is_standard(self, candidate: Candidate) -> bool:
        return candidate.standard_value == self.standard_flag

    def is_valid(self, candidate: Candidate) -> bool:
        return not candidate.is_invalid

    def in_target_domain(self, candidate: Candidate) -> bool:
        return self.target_domain is None or candidate.domain == self.target_domain

    def accepts(self, candidate: Candidate) -> bool:
        """Validity predicate: valid, standard (if required), in target domain."""
        if not self.is_valid(candidate):
            return False
        if self.require_standard and not self.is_standard(candidate):
            return False
        return self.in_target_domain(candidate)

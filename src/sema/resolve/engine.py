"""Slice-0 vocabulary resolver orchestration (the §4 algorithm).

A thin orchestrator: candidate generation (step 1a), standardization (step 2),
the domain gate (step 3), and single-model disambiguation of the ambiguous tail
(step 4) compose into one per-code decision. The code-bearing hot path is pure
SQL — it invokes no model and never touches embeddings or vector recall.

Every field-level detail (the resolved target, the store row, the emitted
assertion) lives in :mod:`sema.resolve.engine_utils`, so this R29-scanned core
module names no domain literal and the resolver stays vocabulary-agnostic.
:class:`VocabularyResolver` is the SOLE writer of value-mapping-store rows.
"""

from __future__ import annotations

from collections.abc import Iterable

from sema.models.planner.mapping_plan import MappingAssertion
from sema.resolve.candidates import generate_candidates
from sema.resolve.disambiguate import Disambiguator, pick_single_survivor
from sema.resolve.domain_gate import apply_domain_gate
from sema.resolve.engine_utils import (
    CodeResolution,
    ResolveContext,
    build_value_mapping,
    build_vocab_lookup_assertion,
    classify_resolution,
    dedupe_concepts,
    ResolveTrace,
)
from sema.resolve.policy import ResolverPolicy
from sema.resolve.standardize import standardize
from sema.resolve.value_mapping_store import ValueMappingStore
from sema.resolve.value_mapping_store_utils import ValueMapping
from sema.resolve.vocab_store import VocabStore
from sema.resolve.vocab_store_utils import ConceptRow


class VocabularyResolver:
    """Deterministic source-code → target-concept resolver (§4 hot path)."""

    def __init__(
        self,
        store: VocabStore,
        policy: ResolverPolicy,
        *,
        disambiguator: Disambiguator | None = None,
    ) -> None:
        self._store = store
        self._policy = policy
        self._disambiguate = disambiguator or pick_single_survivor

    def resolve(self, source_code: str) -> CodeResolution:
        """Resolve one distinct source code to a Zone-classified decision."""
        candidates = generate_candidates(self._store, self._policy, source_code)
        standardized = self._standardized(candidates)
        survivors = apply_domain_gate(self._policy, standardized)
        trace = ResolveTrace(
            n_candidates=len(candidates), n_standardized=len(standardized)
        )
        return classify_resolution(
            source_code, survivors, self._disambiguate, trace, self._policy
        )

    def _standardized(self, candidates: list[ConceptRow]) -> list[ConceptRow]:
        """Standard/valid targets (step 2), deduped, BEFORE the domain gate."""
        standardized: list[ConceptRow] = []
        for candidate in candidates:
            standardized.extend(standardize(self._store, self._policy, candidate))
        return dedupe_concepts(standardized)

    def to_value_mapping(
        self, resolution: CodeResolution, context: ResolveContext
    ) -> ValueMapping:
        return build_value_mapping(resolution, self._policy, context)

    def to_assertion(
        self, resolution: CodeResolution, context: ResolveContext
    ) -> MappingAssertion:
        return build_vocab_lookup_assertion(resolution, self._policy, context)

    def resolve_and_store(
        self,
        source_codes: Iterable[str],
        store: ValueMappingStore,
        context: ResolveContext,
    ) -> list[ValueMapping]:
        """Resolve each code and persist one row per §1.5(a) grain (sole writer)."""
        mappings = [
            self.to_value_mapping(self.resolve(code), context)
            for code in source_codes
        ]
        store.upsert(mappings)
        return mappings

"""Arm S context provider: normal SEMA retrieval over the scoped graph.

`RetrievalEngine.retrieve` then `prune_to_sco` select the physical assets, joins
and governed values; nothing is read from the candidates before pruning, so the
visibility policy (rejected and superseded never shown) applies. Required evidence
is never truncated: an over-budget context is returned whole with an overflow
finding. The budget is the run's first-prompt limit at the conservative chars-per-token
ratio, less the harness prompt frame and the question.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Protocol

from sema.models.config import HierarchyAnchorConfig
from sema.models.context import SemanticCandidateSet
from sema.pipeline.context import prune_to_sco
from showcase.ontology_query_benchmark.profile import Limits
from showcase.ontology_query_benchmark.sema_bindings import (
    BindingResolution,
    ProductSpec,
    Sidecar,
    SpecPinError,
    SuppliedBinding,
)
from showcase.ontology_query_benchmark.sema_build import BuildManifest
from showcase.ontology_query_benchmark.sema_build_utils import Finding, LibraryFacts
from showcase.ontology_query_benchmark.sema_context_utils import Rendering, render


class ManifestMismatchError(RuntimeError):
    """The built graph was not made from these sidecar and facts inputs."""


@dataclass(frozen=True)
class SContext:
    text: str
    identity: str
    findings: tuple[Finding, ...]


class Retriever(Protocol):
    def retrieve(self, question: str, top_k: int = 10) -> SemanticCandidateSet: ...


def conventions_sha256(conventions: tuple[str, ...]) -> str:
    return hashlib.sha256("\n".join(conventions).encode()).hexdigest()


def load_conventions(sidecar: Sidecar) -> tuple[str, ...]:
    """The conventions every arm receives, read from the pinned harness file."""
    data = sidecar.conventions_path.read_bytes()
    if hashlib.sha256(data).hexdigest() != sidecar.conventions_file_sha256:
        raise SpecPinError(f"conventions file {sidecar.conventions_path} changed")
    return tuple(json.loads(data)["conventions"])


def context_char_budget(limits: Limits, sidecar: Sidecar, question: str) -> int:
    whole = math.floor(limits.first_prompt_ceiling * limits.conservative_chars_per_token)
    return whole - sidecar.prompt_frame_chars - len(question)


# Bounds the variable-length PARENT_OF walk whatever depth a slice reports.
ANCHOR_DEPTH_CEILING = 20


def anchor_bounds(manifest: BuildManifest) -> HierarchyAnchorConfig | None:
    """Walk as deep as the built slice; without a hierarchy keep the core default."""
    depth = manifest.counts.get("hierarchy_depth")
    if not depth:
        return None
    return HierarchyAnchorConfig(max_depth=min(depth, ANCHOR_DEPTH_CEILING))


def check_manifest(manifest: BuildManifest, sidecar: Sidecar, facts: LibraryFacts) -> None:
    if (manifest.sidecar_sha256, manifest.facts_sha256) != (sidecar.sha256, facts.sha256):
        raise ManifestMismatchError(
            "graph manifest was built from different sidecar or library facts"
        )


class SContextProvider:
    def __init__(
        self,
        retriever: Retriever,
        sidecar: Sidecar,
        products: tuple[ProductSpec, ...],
        resolution: BindingResolution,
        facts: LibraryFacts,
        *,
        graph_identity: str,
        conventions: tuple[str, ...],
        limits: Limits,
        top_k: int = 10,
        hierarchy_relations: dict[str, bool] | None = None,
    ) -> None:
        self._retriever = retriever
        self._sidecar = sidecar
        self._products = products
        self._resolution = resolution
        self._facts = facts
        self._graph_identity = graph_identity
        self._conventions = conventions
        self._limits = limits
        self._top_k = top_k
        self._hierarchy_relations = hierarchy_relations

    @property
    def facts(self) -> LibraryFacts:
        return self._facts

    def approved_in(self, tables: tuple[str, ...]) -> tuple[SuppliedBinding, ...]:
        return tuple(b for b in self._resolution.approved() if b.table in tables)

    def __call__(self, question: str) -> SContext:
        return self.explain(question)[0]

    def explain(self, question: str) -> tuple[SContext, Rendering]:
        candidates = self._retriever.retrieve(question, top_k=self._top_k)
        rendering = render(
            Rendering(prune_to_sco(candidates)), self._sidecar,
            self._products, self._resolution, self._facts,
            readable=self._hierarchy_relations,
        )
        header = (
            f"SEMA CONTEXT graph={self._graph_identity[:12]} "
            f"bindings={self._sidecar.version} library={self._facts.release}"
        )
        text = f"{header}\n{rendering.text}"
        findings = [*rendering.findings, *self._checks(text, question)]
        identity = hashlib.sha256(f"{text}\n{self._graph_identity}".encode()).hexdigest()
        return SContext(text, identity, tuple(findings)), rendering

    def _checks(self, text: str, question: str) -> list[Finding]:
        findings = []
        if conventions_sha256(self._conventions) != self._sidecar.conventions_sha256:
            findings.append(
                Finding(
                    "conventions differ from the pinned workload conventions "
                    f"(sha256 {self._sidecar.conventions_sha256[:12]})"
                )
            )
        budget = context_char_budget(self._limits, self._sidecar, question)
        if len(text) > budget:
            findings.append(
                Finding(
                    f"overflow: {len(text)} context chars exceed the {budget}-char budget of the "
                    f"{self._limits.first_prompt_ceiling}-token first-prompt limit"
                )
            )
        return findings

"""Arm S context provider: normal SEMA retrieval over the scoped graph.

`RetrievalEngine.retrieve` then `prune_to_sco` select the physical assets, joins
and governed values; nothing is read from the candidates before pruning, so the
visibility policy (rejected and superseded never shown) applies. Required evidence
is never truncated: an over-budget context is returned whole with an overflow
finding.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Protocol

from sema.models.context import SemanticCandidateSet
from sema.pipeline.context import prune_to_sco
from showcase.ontology_query_benchmark.sema_bindings import (
    BindingResolution,
    ProductSpec,
    Sidecar,
    SpecPinError,
    SuppliedBinding,
)
from showcase.ontology_query_benchmark.sema_build import BuildManifest
from showcase.ontology_query_benchmark.sema_build_utils import LibraryFacts
from showcase.ontology_query_benchmark.sema_context_utils import Rendering, render


FIRST_PROMPT_TOKEN_LIMIT = 13_934
CHARS_PER_TOKEN = 2.08
HARNESS_OVERHEAD_CHARS = 2_900


class ManifestMismatchError(RuntimeError):
    """The built graph was not made from these sidecar and facts inputs."""


@dataclass(frozen=True)
class SContext:
    text: str
    identity: str
    findings: tuple[str, ...]


class Retriever(Protocol):
    def retrieve(self, question: str, top_k: int = 10) -> SemanticCandidateSet: ...


def conventions_sha256(conventions: tuple[str, ...]) -> str:
    return hashlib.sha256("\n".join(conventions).encode()).hexdigest()


def load_conventions(sidecar: Sidecar) -> tuple[str, ...]:
    """The W4S conventions every arm receives, read from the pinned harness file."""
    data = sidecar.conventions_path.read_bytes()
    if hashlib.sha256(data).hexdigest() != sidecar.conventions_file_sha256:
        raise SpecPinError(f"conventions file {sidecar.conventions_path} changed")
    return tuple(json.loads(data)["conventions"])


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
        top_k: int = 10,
    ) -> None:
        self._retriever = retriever
        self._sidecar = sidecar
        self._products = products
        self._resolution = resolution
        self._facts = facts
        self._graph_identity = graph_identity
        self._conventions = conventions
        self._top_k = top_k

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
        )
        header = (
            f"SEMA CONTEXT graph={self._graph_identity[:12]} "
            f"bindings={self._sidecar.version} library={self._facts.release}"
        )
        text = f"{header}\n{rendering.text}"
        findings = [*rendering.findings, *self._checks(text)]
        identity = hashlib.sha256(f"{text}\n{self._graph_identity}".encode()).hexdigest()
        return SContext(text, identity, tuple(findings)), rendering

    def _checks(self, text: str) -> list[str]:
        findings = []
        if conventions_sha256(self._conventions) != self._sidecar.conventions_sha256:
            findings.append(
                "conventions differ from the pinned workload conventions "
                f"(sha256 {self._sidecar.conventions_sha256[:12]})"
            )
        if len(text) > self._sidecar.context_char_budget:
            findings.append(
                f"context overflow: {len(text)} chars exceed the "
                f"{self._sidecar.context_char_budget}-char context budget"
            )
        return findings

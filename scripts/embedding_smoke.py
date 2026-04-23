"""Measurement B — embedding smoke for OpenRouter (ada-002) vs Databricks (bge-large-en).

Prints per-provider output dimension and cosine-similarity table for a fixed
set of domain-term pairs from `showcase/cbioportal_to_omop/slices/dev_slice_poc.yaml`.

Run with each provider sourced in env:

    source .env.openrouter-baseline
    uv run python scripts/embedding_smoke.py --label openrouter

    source .env.databricks-candidate
    uv run python scripts/embedding_smoke.py --label databricks-mosaic
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass


DOMAIN_TERMS: list[str] = [
    "patient",
    "sample",
    "mutation",
    "structural variant",
    "copy number alteration",
    "gene panel matrix",
    "clinical supplemental hypoxia",
    "resource definition",
    "resource patient",
    "timeline sample acquisition",
    "timeline status",
    "timeline treatment",
    "overall survival months",
    "disease-free survival status",
    "tumor mutational burden",
    "microsatellite instability",
    "icd10 diagnosis code",
    "cancer stage",
    "drug regimen",
    "variant classification",
]

# Pairs that should have high similarity (related) and low similarity (unrelated).
RELATED_PAIRS: list[tuple[str, str]] = [
    ("patient", "resource patient"),
    ("mutation", "variant classification"),
    ("structural variant", "mutation"),
    ("copy number alteration", "mutation"),
    ("tumor mutational burden", "mutation"),
    ("microsatellite instability", "mutation"),
    ("overall survival months", "disease-free survival status"),
    ("timeline sample acquisition", "sample"),
    ("timeline treatment", "drug regimen"),
    ("icd10 diagnosis code", "cancer stage"),
]


@dataclass
class SmokeReport:
    label: str
    dim: int
    pair_scores: dict[tuple[str, str], float]

    def render(self) -> str:
        lines = [
            f"label: {self.label}",
            f"dim: {self.dim}",
            "pair cosine similarities:",
        ]
        for (a, b), score in self.pair_scores.items():
            lines.append(f"  {a!r} vs {b!r}: {score:.4f}")
        return "\n".join(lines)


def _cosine(u: list[float], v: list[float]) -> float:
    dot = sum(a * b for a, b in zip(u, v))
    norm_u = math.sqrt(sum(a * a for a in u))
    norm_v = math.sqrt(sum(b * b for b in v))
    if norm_u == 0 or norm_v == 0:
        return 0.0
    return dot / (norm_u * norm_v)


def _embed_all(embedder, terms: list[str]) -> dict[str, list[float]]:
    if hasattr(embedder, "embed_documents"):
        vecs = embedder.embed_documents(terms)
    elif hasattr(embedder, "encode"):
        vecs = embedder.encode(terms)
    else:
        raise SystemExit("Embedder exposes no known batch method.")
    return dict(zip(terms, (list(v) for v in vecs)))


def _build_embedder():
    from sema.cli_factories import _get_embedder
    from sema.models.config import EmbeddingConfig
    return _get_embedder(EmbeddingConfig())


def run(label: str) -> SmokeReport:
    embedder = _build_embedder()
    vecs = _embed_all(embedder, DOMAIN_TERMS)
    sample_dim = len(next(iter(vecs.values())))
    pair_scores: dict[tuple[str, str], float] = {}
    for a, b in RELATED_PAIRS:
        pair_scores[(a, b)] = _cosine(vecs[a], vecs[b])
    return SmokeReport(label=label, dim=sample_dim, pair_scores=pair_scores)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True, help="report label")
    args = parser.parse_args(argv)
    report = run(args.label)
    print(report.render())
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

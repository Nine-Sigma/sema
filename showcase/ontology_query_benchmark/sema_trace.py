"""Trace one question through arm S: library identity -> binding -> asset -> context.

``open_live_provider`` is the runner-facing factory; ``main`` prints the trace for
one question against the isolated Neo4j graph.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from showcase.ontology_query_benchmark.profile import Profile, load_profile
from showcase.ontology_query_benchmark.sema_bindings import (
    load_products,
    load_sidecar,
    resolve_bindings,
)
from showcase.ontology_query_benchmark.sema_build import read_catalog, read_manifest
from showcase.ontology_query_benchmark.sema_build_utils import import_library_facts
from showcase.ontology_query_benchmark.sema_context import (
    SContext,
    SContextProvider,
    check_manifest,
    load_conventions,
)
from showcase.ontology_query_benchmark.sema_context_utils import Rendering

def trace(provider: SContextProvider, question: str) -> list[dict[str, Any]]:
    return trace_rows(provider, *provider.explain(question))


def trace_rows(
    provider: SContextProvider, context: SContext, rendering: Rendering
) -> list[dict[str, Any]]:
    retrieved = {(a.table, c) for a in rendering.sco.physical_assets for c in a.columns}
    rows = []
    for b in provider.approved_in(rendering.tables):
        element = provider.facts.elements.get(b.de_id or "")
        rows.append({
            "library": {"de_id": b.de_id, "library_id": b.library_id,
                        "label": element.label if element else None},
            "binding": {"status": b.status.value, "authority": b.authority,
                        "spec_path": str(b.spec.path), "spec_sha256": b.spec.sha256,
                        "strategy": b.strategy},
            "physical": {"relation": b.relation, "column": b.column,
                         "type": b.physical_type, "companions": list(b.companions)},
            "context": {"retrieved": (b.table, b.column) in retrieved,
                        "rendered": f"    {b.column} <- " in context.text,
                        "values_listed": rendering.listed.get((b.table, b.column), 0)},
        })
    return rows


@contextmanager
def open_live_provider(
    profile: Profile,
    neo4j_uri: str,
    manifest_path: Path,
    projection: Path,
    receipt: Path,
    *,
    concept_spec: Path | None = None,
) -> Iterator[SContextProvider]:
    from sema.pipeline.retrieval import RetrievalEngine, RetrievalScope
    from showcase.ontology_query_benchmark import sema_live

    sidecar = load_sidecar(profile.sema.sidecar, profile.harness.root)
    facts = import_library_facts(concept_spec or profile.sema.concept_spec, projection, receipt)
    manifest = read_manifest(manifest_path)
    check_manifest(manifest, sidecar, facts)
    workspace = sema_live.pinned_workspace(profile.routing)
    products = load_products(sidecar)
    catalog = read_catalog(sema_live.UnityCatalogSource(workspace), sidecar)
    with sema_live.neo4j_driver(neo4j_uri) as driver:
        engine = RetrievalEngine(
            driver, sema_live.embedder(workspace, profile.routing),
            embedder_model_name=manifest.embedder_model,
            scope=RetrievalScope(model_role=sidecar.model_role),
        )
        yield SContextProvider(
            engine, sidecar, products, resolve_bindings(sidecar, products, catalog), facts,
            graph_identity=manifest.graph_identity, conventions=load_conventions(sidecar),
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Trace one question through arm S.")
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--neo4j-uri", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--projection", type=Path, required=True)
    parser.add_argument("--projection-receipt", type=Path, required=True)
    parser.add_argument("--concept-spec", type=Path, help="default: the profile's concept spec")
    parser.add_argument("--question", required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    with open_live_provider(
        load_profile(args.profile), args.neo4j_uri, args.manifest, args.projection,
        args.projection_receipt, concept_spec=args.concept_spec,
    ) as p:
        context, rendering = p.explain(args.question)
        report = {"question": args.question, "identity": context.identity,
                  "findings": list(context.findings),
                  "trace": trace_rows(p, context, rendering),
                  "context_text": context.text}
    payload = json.dumps(report, indent=1)
    if args.out:
        args.out.write_text(payload)
    print(payload)


if __name__ == "__main__":
    main()

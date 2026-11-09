"""Trace one question through arm S: library identity -> binding -> asset -> context.

``open_live_provider`` is the runner-facing factory; ``main`` prints the trace for
one question (milestone A uses Q01) against the isolated Neo4j graph.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from showcase.partner_timbr_benchmark.sema_bindings import (
    SIDECAR_PATH,
    load_products,
    load_sidecar,
    resolve_bindings,
)
from showcase.partner_timbr_benchmark.sema_build import read_catalog, read_manifest
from showcase.partner_timbr_benchmark.sema_build_utils import (
    CONCEPT_SPEC_PATH,
    import_library_facts,
)
from showcase.partner_timbr_benchmark.sema_context import (
    SContext,
    SContextProvider,
    check_manifest,
    load_conventions,
)
from showcase.partner_timbr_benchmark.sema_context_utils import Rendering

Q01 = "How many living patients are female (administrative gender)?"


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
    neo4j_uri: str,
    manifest_path: Path,
    projection: Path,
    receipt: Path,
    *,
    concept_spec: Path = CONCEPT_SPEC_PATH,
    sidecar_path: Path = SIDECAR_PATH,
) -> Iterator[SContextProvider]:
    from sema.pipeline.retrieval import RetrievalEngine, RetrievalScope
    from showcase.partner_timbr_benchmark import sema_live

    sidecar = load_sidecar(sidecar_path)
    facts = import_library_facts(concept_spec, projection, receipt)
    manifest = read_manifest(manifest_path)
    check_manifest(manifest, sidecar, facts)
    workspace = sema_live.prt_test_workspace()
    products = load_products(sidecar)
    catalog = read_catalog(sema_live.UnityCatalogSource(workspace), sidecar)
    with sema_live.neo4j_driver(neo4j_uri) as driver:
        engine = RetrievalEngine(
            driver, sema_live.embedder(workspace),
            embedder_model_name=manifest.embedder_model,
            scope=RetrievalScope(model_role=sidecar.model_role),
        )
        yield SContextProvider(
            engine, sidecar, products, resolve_bindings(sidecar, products, catalog), facts,
            graph_identity=manifest.graph_identity, conventions=load_conventions(sidecar),
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Trace one question through arm S.")
    parser.add_argument("--neo4j-uri", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--projection", type=Path, required=True)
    parser.add_argument("--projection-receipt", type=Path, required=True)
    parser.add_argument("--concept-spec", type=Path, default=CONCEPT_SPEC_PATH)
    parser.add_argument("--sidecar", type=Path, default=SIDECAR_PATH)
    parser.add_argument("--question", default=Q01)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    with open_live_provider(
        args.neo4j_uri, args.manifest, args.projection, args.projection_receipt,
        concept_spec=args.concept_spec, sidecar_path=args.sidecar,
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

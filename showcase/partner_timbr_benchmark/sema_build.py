"""Build arm S's isolated, scoped retrieval graph from supplied CPD bindings.

The projection reuses SEMA's target-retrieval path (`build_target_retrieval_graph`)
and `EmbeddingEngine`; this module only adapts the CPD specs and library facts
into those inputs and records a manifest that pins every input by hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from sema.engine.embeddings import EmbeddingEngine
from sema.graph.loader import GraphLoader
from sema.graph.physical_catalog import PhysicalCatalogSource
from sema.graph.target_graph_build import build_target_retrieval_graph
from showcase.partner_timbr_benchmark.sema_bindings import (
    SIDECAR_PATH,
    BindingResolution,
    ProductSpec,
    Sidecar,
    load_products,
    load_sidecar,
    resolve_bindings,
)
from showcase.partner_timbr_benchmark.sema_build_utils import (
    FactsConceptSource,
    LibraryFacts,
    concept_fields,
    load_library_facts,
    property_text,
    target_model,
)

BUILD_VERSION = "sema-build-1"
EMBEDDER_MODEL = "databricks-bge-large-en"


@dataclass(frozen=True)
class BuildManifest:
    graph_identity: str
    model_role: str
    sidecar_version: str
    sidecar_sha256: str
    spec_sha256: tuple[str, ...]
    facts_sha256: str
    facts_release: str
    embedder_model: str
    dimensions: int
    counts: dict[str, int] = field(default_factory=dict)
    findings: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


def build_scoped_graph(
    loader: GraphLoader,
    catalog_source: PhysicalCatalogSource,
    sidecar: Sidecar,
    facts: LibraryFacts,
    embedder: Any,
    *,
    dimensions: int,
    embedder_model: str = EMBEDDER_MODEL,
) -> BuildManifest:
    products = load_products(sidecar)
    resolution = resolve_bindings(sidecar, products, read_catalog(catalog_source, sidecar))
    model, binding = target_model(sidecar, products, resolution)
    fields, unloaded, notes = concept_fields(resolution.approved(), facts)
    build_target_retrieval_graph(
        loader, model, binding, catalog_source,
        concepts=FactsConceptSource(facts),
        concept_vocabulary=sidecar.concept_vocabulary,
        concept_fields=fields, value_bridges=(),
        bridge_source_schema=sidecar.schema, load_ancestors=False,
    )
    embedded = _embed(loader, embedder, dimensions, products, resolution, facts)
    counts = {
        "approved_bindings": len(resolution.approved()),
        "refused_bindings": len(resolution.bindings) - len(resolution.approved()),
        "value_sets": len(fields),
        "value_set_members": sum(len(f.concept_codes) for f in fields),
        **embedded,
    }
    return BuildManifest(
        graph_identity=_identity(sidecar, facts, dimensions, embedder_model),
        model_role=sidecar.model_role,
        sidecar_version=sidecar.version,
        sidecar_sha256=sidecar.sha256,
        spec_sha256=tuple(p.sha256 for p in sidecar.specs),
        facts_sha256=facts.sha256,
        facts_release=facts.release,
        embedder_model=embedder_model,
        dimensions=dimensions,
        counts=counts,
        findings=resolution.findings + tuple(unloaded),
        notes=tuple(notes),
    )


def read_catalog(
    catalog_source: PhysicalCatalogSource, sidecar: Sidecar
) -> dict[str, dict[str, str]]:
    return {
        pin.table: {
            c.name: c.data_type
            for c in catalog_source.columns(
                catalog=sidecar.catalog, schema=sidecar.schema, table=pin.table
            )
        }
        for pin in sidecar.specs
    }


def _embed(
    loader: GraphLoader,
    embedder: Any,
    dimensions: int,
    products: tuple[ProductSpec, ...],
    resolution: BindingResolution,
    facts: LibraryFacts,
) -> dict[str, int]:
    engine = EmbeddingEngine(model=embedder, loader=loader)
    engine.create_all_indexes(dimensions=dimensions)
    entities = [
        {"name": p.pin.table, "text": f"{p.pin.product} {_grain(p)}"} for p in products
    ]
    engine.embed_and_store("Entity", "name", entities, lambda e: str(e["text"]))
    props = [
        {"name": col, "entity_name": b.table, "text": property_text(b, col, facts)}
        for b in resolution.approved()
        for col in (b.column, *b.companions, *b.catalog_companions)
    ]
    engine.embed_and_store("Property", "name", props, lambda p: str(p["text"]))
    return {"embedded_entities": len(entities), "embedded_properties": len(props)}


def _grain(product: ProductSpec) -> str:
    return "; ".join(str(e.get("grain", "")) for e in product.raw.get("entities", []))


def _identity(
    sidecar: Sidecar, facts: LibraryFacts, dimensions: int, embedder_model: str
) -> str:
    pinned = {
        "build": BUILD_VERSION, "sidecar": sidecar.sha256,
        "specs": [p.sha256 for p in sidecar.specs], "facts": facts.sha256,
        "model_role": sidecar.model_role, "dimensions": dimensions,
        "embedder": embedder_model,
    }
    return hashlib.sha256(json.dumps(pinned, sort_keys=True).encode()).hexdigest()


def write_manifest(manifest: BuildManifest, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(manifest), indent=1, sort_keys=True))


def read_manifest(path: Path) -> BuildManifest:
    raw = json.loads(path.read_text())
    for key in ("spec_sha256", "findings", "notes"):
        raw[key] = tuple(raw[key])
    return BuildManifest(**raw)


def main(argv: list[str] | None = None) -> None:
    from showcase.partner_timbr_benchmark import sema_live

    args = _parser().parse_args(argv)
    workspace = sema_live.prt_test_workspace()
    with sema_live.neo4j_driver(args.neo4j_uri) as driver:
        loader = GraphLoader(driver)
        manifest = build_scoped_graph(
            loader, sema_live.UnityCatalogSource(workspace),
            load_sidecar(args.sidecar), load_library_facts(args.facts),
            sema_live.embedder(workspace), dimensions=args.dimensions,
        )
    write_manifest(manifest, args.manifest)
    print(json.dumps({"graph_identity": manifest.graph_identity, **manifest.counts}))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build arm S scoped graph.")
    parser.add_argument("--neo4j-uri", required=True)
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, default=SIDECAR_PATH)
    parser.add_argument("--dimensions", type=int, default=1024)
    return parser


if __name__ == "__main__":
    main()

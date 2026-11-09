"""Build arm S's isolated, scoped retrieval graph from supplied product-spec bindings.

The projection reuses SEMA's target-retrieval path (`build_target_retrieval_graph`)
and `EmbeddingEngine`; this module only adapts the product specs and library facts
into those inputs and records a manifest that pins every input by hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from sema.engine.embeddings import EmbeddingEngine
from sema.graph.loader import GraphLoader
from sema.graph.physical_catalog import PhysicalCatalogSource
from sema.graph.target_graph_build import build_target_retrieval_graph
from showcase.ontology_query_benchmark.profile import Attribution, Profile, load_profile
from showcase.ontology_query_benchmark.routing import require_clean_process
from showcase.ontology_query_benchmark.sema_bindings import (
    BindingResolution,
    InputFile,
    ProductSpec,
    Sidecar,
    load_products,
    load_sidecar,
    resolve_bindings,
)
from showcase.ontology_query_benchmark.sema_build_utils import (
    FactsConceptSource,
    LibraryFacts,
    absent_elements,
    concept_fields,
    import_library_facts,
    property_text,
    target_model,
)
from showcase.ontology_query_benchmark.sema_hierarchy import (
    check_relations,
    release_findings,
    term_embeddings,
    write_hierarchy,
)

BUILD_VERSION = "sema-build-1"


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
    inputs: tuple[InputFile, ...] = ()
    hierarchy_relations: dict[str, bool] | None = None


def build_scoped_graph(
    loader: GraphLoader,
    catalog_source: PhysicalCatalogSource,
    sidecar: Sidecar,
    facts: LibraryFacts,
    embedder: Any,
    *,
    dimensions: int,
    embedder_model: str,
    relation_reader: Callable[[str], object] | None = None,
) -> BuildManifest:
    products = load_products(sidecar)
    resolution = resolve_bindings(sidecar, products, read_catalog(catalog_source, sidecar))
    model, binding = target_model(sidecar, products, resolution)
    fields = concept_fields(sidecar, resolution.approved(), facts)
    build_target_retrieval_graph(
        loader, model, binding, catalog_source,
        concepts=FactsConceptSource(facts),
        concept_vocabulary=sidecar.concept_vocabulary,
        concept_fields=fields, value_bridges=(),
        bridge_source_schema=sidecar.schema, load_ancestors=False,
    )
    written, readable, unreadable = _hierarchy(loader, facts, sidecar, relation_reader, embedder)
    embedded = _embed(loader, embedder, dimensions, sidecar, products, resolution, facts)
    counts = {
        "approved_bindings": len(resolution.approved()),
        "refused_bindings": len(resolution.bindings) - len(resolution.approved()),
        "value_sets": len(fields),
        "value_set_members": sum(len(f.concept_codes) for f in fields),
        **embedded,
        **written,
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
        findings=resolution.findings
        + tuple(f.message for f in absent_elements(resolution.approved(), facts))
        + unreadable,
        inputs=sidecar.input_files() + facts.inputs,
        hierarchy_relations=readable,
    )


def _hierarchy(
    loader: GraphLoader, facts: LibraryFacts, sidecar: Sidecar,
    reader: Callable[[str], object] | None, embedder: Any,
) -> tuple[dict[str, int], dict[str, bool] | None, tuple[str, ...]]:  # fmt: skip
    if facts.hierarchy is None:
        return {}, None, ()
    vocabulary, codes = sidecar.concept_vocabulary, set(facts.concepts)
    counts = write_hierarchy(loader, facts.hierarchy, member_vocabulary=vocabulary,
                             library_codes=codes, source_schema=sidecar.schema)  # fmt: skip
    embedded, capped = term_embeddings(facts.hierarchy, loader.query_nodes_by_label("Term"),
                                       member_vocabulary=vocabulary, library_codes=codes)  # fmt: skip
    engine = EmbeddingEngine(model=embedder, loader=loader)
    engine.embed_and_store("Term", "code", embedded, lambda t: str(t["label"]))
    relations, findings = check_relations(facts.hierarchy.spec, reader)
    return ({**counts, "embedded_terms": len(embedded)}, relations,
            (*release_findings(facts.hierarchy), *findings, *capped))  # fmt: skip


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
    sidecar: Sidecar,
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
        {"name": col, "entity_name": b.table, "text": property_text(sidecar, b, col, facts)}
        for b in resolution.approved()
        for col in (b.column, *b.companion_columns, *b.catalog_companions)
    ]
    engine.embed_and_store("Property", "name", props, lambda p: str(p["text"]))
    return {"embedded_entities": len(entities), "embedded_properties": len(props)}


def _grain(product: ProductSpec) -> str:
    return "; ".join(e.grain for e in product.doc.entities)


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
    raw = asdict(manifest)
    if raw["hierarchy_relations"] is None:
        del raw["hierarchy_relations"]
    path.write_text(json.dumps(raw, indent=1, sort_keys=True))


def read_manifest(path: Path) -> BuildManifest:
    raw = json.loads(path.read_text())
    for key in ("spec_sha256", "findings"):
        raw[key] = tuple(raw[key])
    raw["inputs"] = tuple(
        InputFile(**{**i, "attribution": Attribution(i["attribution"])})
        for i in raw.get("inputs", ())
    )
    return BuildManifest(**raw)


def main(argv: list[str] | None = None) -> None:
    from showcase.ontology_query_benchmark import sema_live

    parser = _parser()
    args = parser.parse_args(argv)
    if args.concept_spec and not args.concept_spec_attribution:
        parser.error("--concept-spec needs --concept-spec-attribution")
    profile = load_profile(args.profile)
    routing = profile.routing
    require_clean_process(os.environ, sys.modules, routing)
    concept_spec = args.concept_spec or profile.sema.concept_spec
    attribution = args.concept_spec_attribution or profile.sema.concept_spec_attribution
    workspace = sema_live.pinned_workspace(routing)
    hierarchy = profile.extract.concept_hierarchy
    with sema_live.neo4j_driver(args.neo4j_uri) as driver, _relation_reader(profile) as reader:
        loader = GraphLoader(driver)
        manifest = build_scoped_graph(
            loader, sema_live.UnityCatalogSource(workspace),
            load_sidecar(profile.sema.sidecar, profile.harness.root),
            import_library_facts(concept_spec, args.projection, args.projection_receipt,
                                 spec_attribution=attribution, hierarchy=hierarchy),
            sema_live.embedder(workspace, routing), dimensions=routing.embedding_dimensions,
            embedder_model=routing.embedding_endpoint, relation_reader=reader,
        )
    write_manifest(manifest, args.manifest)
    print(json.dumps({"graph_identity": manifest.graph_identity, **manifest.counts}))


@contextmanager
def _relation_reader(profile: Profile) -> Iterator[Callable[[str], object] | None]:
    """The extract's warehouse connection, opened only when a hierarchy is declared."""
    if profile.extract.concept_hierarchy is None:
        yield None
        return
    from showcase.ontology_query_benchmark.intake_extract import _connect

    cursor = _connect(profile.routing)
    try:
        yield lambda sql: (cursor.execute(sql), cursor.fetchall())
    finally:
        cursor.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build arm S scoped graph.")
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--neo4j-uri", required=True)
    parser.add_argument("--projection", type=Path, required=True)
    parser.add_argument("--projection-receipt", type=Path, required=True)
    parser.add_argument("--concept-spec", type=Path, help="default: the profile's concept spec")
    parser.add_argument("--concept-spec-attribution", type=Attribution, choices=list(Attribution))
    parser.add_argument("--manifest", type=Path, required=True)
    return parser


if __name__ == "__main__":
    main()

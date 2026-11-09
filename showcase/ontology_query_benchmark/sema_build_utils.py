"""Library facts and the product-spec-to-target-model adapter behind ``sema_build``.

Library facts come from SEMA's concept-source importer run over the scoped
workload projection, never from a benchmark-only reader.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sema.binding.importer import import_concept_source
from sema.binding.spec import load_concept_source_spec
from sema.graph.concept_source import ConceptAncestor, ConceptDetail
from sema.graph.target_graph_build import ConceptFieldSpec
from sema.graph.target_retrieval_loader import EntityTableBinding, PhysicalSchemaBinding
from sema.ingest.duckdb_staging import Staging
from sema.intake.relations import RelationProvider
from sema.models.planner._enums import TargetArtifactKind
from sema.models.target.completeness import (
    SemanticCompleteness,
    SemanticCompletenessAnnotations,
)
from sema.models.target.declared import DeclaredJoinDecl
from sema.models.target.descriptor import TargetModelDescriptor
from sema.models.target.entity import TargetEntityDecl
from sema.models.target.normalized import NormalizedTargetModel
from sema.models.target.properties import TargetPropertyDecl
from sema.models.target.refs import TargetEntityRef
from showcase.ontology_query_benchmark.product_spec import CompanionRole
from showcase.ontology_query_benchmark.profile import Attribution
from showcase.ontology_query_benchmark.sema_bindings import (
    BindingResolution,
    InputFile,
    ProductSpec,
    Sidecar,
    SpecPinError,
    SuppliedBinding,
)

COMPANION_TEXT = {
    CompanionRole.LABEL: "label text",
    CompanionRole.MAPPING_TYPE: "mapping type",
}


@dataclass(frozen=True)
class ElementFacts:
    label: str
    definition: str
    members: tuple[tuple[str, str], ...]

    @property
    def pv_count(self) -> int:
        return len(self.members)


@dataclass(frozen=True)
class LibraryFacts:
    release: str
    sha256: str
    elements: Mapping[str, ElementFacts]
    concepts: Mapping[str, ConceptDetail]
    inputs: tuple[InputFile, ...]


def import_library_facts(
    spec_path: Path, projection: Path, receipt: Path, *, spec_attribution: Attribution
) -> LibraryFacts:
    """Run the concept-source importer over a receipted projection copy."""
    digest, assistance = _check_receipt(projection, receipt)
    spec_bytes = spec_path.read_bytes()
    spec = load_concept_source_spec(spec_path)
    with tempfile.TemporaryDirectory() as scratch:
        copy = Path(scratch) / projection.name
        shutil.copyfile(projection, copy)
        staging = Staging(str(copy))
        try:
            imported = import_concept_source(
                spec, RelationProvider(warehouse=None, staging=staging), dataset_map={}
            )
        finally:
            staging.close()
    identity = hashlib.sha256(spec_bytes + digest.encode()).hexdigest()
    inputs = (
        InputFile("concept_spec", str(spec_path), hashlib.sha256(spec_bytes).hexdigest(),
                  spec_attribution),
        InputFile("projection", str(projection), digest, assistance),
    )
    return _facts(f"{spec.id}@{spec.release}", identity, imported.manifest, inputs)


def _check_receipt(projection: Path, receipt: Path) -> tuple[str, Attribution]:
    raw = json.loads(receipt.read_text())
    digest = hashlib.sha256(projection.read_bytes()).hexdigest()
    if not raw.get("complete"):
        raise SpecPinError(f"projection receipt {receipt} marks the extract incomplete")
    if raw["duckdb"]["sha256"] != digest:
        raise SpecPinError(f"projection {projection} does not match receipt {receipt}")
    kind = (raw.get("assistance") or {}).get("kind")
    if kind not in set(Attribution):
        raise SpecPinError(f"projection receipt {receipt} states no assistance kind: {kind!r}")
    return digest, Attribution(kind)


def _facts(
    release: str, identity: str, manifest: Mapping[str, Any], inputs: tuple[InputFile, ...]
) -> LibraryFacts:
    members: dict[str, list[tuple[str, str]]] = {}
    concepts: dict[str, ConceptDetail] = {}
    for term in manifest["terms"]:
        members.setdefault(term["vocabulary"], []).append((term["code"], term["display"]))
        concepts[term["code"]] = ConceptDetail(
            term["code"], term["display"], tuple(term.get("synonyms") or ())
        )
    elements = {
        prop["concept_ref"]["id"]: ElementFacts(
            label=prop["name"],
            definition=prop.get("definition") or "",
            members=tuple(members.get(_vocabulary(prop), ())),
        )
        for entity in manifest["entities"] for prop in entity.get("properties", [])
        if prop.get("concept_ref")
    }
    return LibraryFacts(release, identity, elements, concepts, inputs)


def _vocabulary(prop: Mapping[str, Any]) -> str:
    return str((prop.get("vocabulary_binding") or {}).get("vocabulary", ""))


class FactsConceptSource:
    """`ConceptSource` over the imported library terms: names and synonyms, no ancestry."""

    def __init__(self, facts: LibraryFacts) -> None:
        self._concepts = facts.concepts

    def name_synonyms(self, code: str) -> ConceptDetail | None:
        return self._concepts.get(code)

    def ancestors(self, code: str) -> list[ConceptAncestor]:
        return []


@dataclass(frozen=True)
class Finding:
    """A context defect; one naming a column concerns only SQL that reads that column."""

    message: str
    relation: str | None = None
    column: str | None = None


def absent_elements(
    bindings: tuple[SuppliedBinding, ...], facts: LibraryFacts
) -> list[Finding]:
    return [
        Finding(
            f"library element absent: {b.de_id} ({b.relation}.{b.column}) is not in "
            f"library {facts.release}",
            b.relation,
            b.column,
        )
        for b in bindings
        if b.de_id is not None and b.de_id not in facts.elements
    ]


def qualified_name(sidecar: Sidecar, product: str) -> str:
    return f"{sidecar.target_namespace}.{product}"


def target_model(
    sidecar: Sidecar,
    products: tuple[ProductSpec, ...],
    resolution: BindingResolution,
    catalog: Mapping[str, Mapping[str, str]],
) -> tuple[NormalizedTargetModel, PhysicalSchemaBinding]:
    entities = [
        TargetEntityDecl(
            ref=TargetEntityRef(
                target_model_id=sidecar.target_model_id,
                qualified_name=qualified_name(sidecar, p.pin.product),
                kind=TargetArtifactKind.TABLE_ROW,
            ),
            properties=_properties(resolution.approved(), p, catalog[p.pin.table]),
            declared_joins=_declared_joins(sidecar, p),
        )
        for p in products
    ]
    model = NormalizedTargetModel(descriptor=_descriptor(sidecar), entities=entities)
    binding = PhysicalSchemaBinding(
        catalog=sidecar.catalog,
        schema=sidecar.schema,
        entity_tables=tuple(
            EntityTableBinding(qualified_name(sidecar, p.pin.product), p.pin.table)
            for p in products
        ),
        model_role=sidecar.model_role,
    )
    return model, binding


def _properties(
    approved: tuple[SuppliedBinding, ...],
    product: ProductSpec,
    catalog_columns: Mapping[str, str],
) -> list[TargetPropertyDecl]:
    props: list[TargetPropertyDecl] = []
    for b in approved:
        if b.table != product.pin.table:
            continue
        bound = b.de_id if b.governed else None
        props.append(TargetPropertyDecl(
            name=b.column, type=b.physical_type, nullable=True, vocabulary_binding=bound
        ))
        props.extend(
            TargetPropertyDecl(name=c, type="STRING", nullable=True)
            for c in (*b.companion_columns, *b.catalog_companions)
        )
    props.extend(
        TargetPropertyDecl(name=c, type=t, nullable=True)
        for c, t in unbound_columns(approved, product, catalog_columns).items()
    )
    denied = set(product.doc.exposure.denied_columns)
    return [
        p.model_copy(update={"exposure": "denied"}) if p.name in denied else p for p in props
    ]


def unbound_columns(
    approved: tuple[SuppliedBinding, ...],
    product: ProductSpec,
    catalog_columns: Mapping[str, str],
) -> dict[str, str]:
    """Catalog columns no approved binding covers, with their physical type."""
    bound = {c for b in approved if b.table == product.pin.table for c in b.columns}
    return {c: t for c, t in catalog_columns.items() if c not in bound}


def _declared_joins(sidecar: Sidecar, product: ProductSpec) -> list[DeclaredJoinDecl]:
    return [
        DeclaredJoinDecl(
            referenced_entity=qualified_name(sidecar, rel.target_product_id),
            join_keys=list(zip(rel.source_columns, rel.target_columns)),
            evidence="declared",
        )
        for rel in product.doc.relationships
    ]


def _descriptor(sidecar: Sidecar) -> TargetModelDescriptor:
    return TargetModelDescriptor(
        target_model_id=sidecar.target_model_id,
        target_model_version=sidecar.version,
        display_name=sidecar.target_display_name,
        completeness=SemanticCompletenessAnnotations(
            structure=SemanticCompleteness.COMPLETE,
            obligations=SemanticCompleteness.NONE,
            vocabulary_bindings=SemanticCompleteness.NONE,
            semantic_aliases=SemanticCompleteness.NONE,
            terms=SemanticCompleteness.NONE,
        ),
    )


def concept_fields(
    sidecar: Sidecar, approved: tuple[SuppliedBinding, ...], facts: LibraryFacts
) -> list[ConceptFieldSpec]:
    """Value sets to materialize: every imported member of each pass-through element."""
    fields: list[ConceptFieldSpec] = []
    for b in approved:
        element = facts.elements.get(b.de_id or "")
        if b.governed and element is not None:
            codes = tuple(code for code, _ in element.members)
            fields.append(ConceptFieldSpec(qualified_name(sidecar, b.product), b.column, codes))
    return fields


def property_text(
    sidecar: Sidecar, binding: SuppliedBinding | None, column: str, facts: LibraryFacts
) -> str:
    if binding is None:
        return column.replace("-", " ").replace("_", " ")
    element = facts.elements.get(binding.de_id or "")
    base = element.label if element else binding.library_id.removesuffix(sidecar.library_id_suffix)
    base = base.replace("-", " ").replace("_", " ")
    role = next((r for r, c in binding.companions if c == column), None)
    if role is not None:
        return f"{base} {COMPANION_TEXT[role]}"
    if column in binding.catalog_companions:
        return f"{base} vocabulary code"
    return f"{base}: {element.definition}" if element and element.definition else base

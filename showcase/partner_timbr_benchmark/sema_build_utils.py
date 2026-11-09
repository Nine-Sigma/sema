"""Library facts and the CPD-to-target-model adapter behind ``sema_build``."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from sema.graph.concept_source import ConceptAncestor, ConceptDetail
from sema.graph.target_graph_build import ConceptFieldSpec
from sema.graph.target_retrieval_loader import EntityTableBinding, PhysicalSchemaBinding
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
from showcase.partner_timbr_benchmark.sema_bindings import (
    BindingResolution,
    ProductSpec,
    Sidecar,
    SuppliedBinding,
)

_MODEL_ID = "partner-bench-s"


@dataclass(frozen=True)
class ElementFacts:
    label: str
    definition: str
    pv_count: int
    members: tuple[tuple[str, str], ...] | None


@dataclass(frozen=True)
class LibraryFacts:
    release: str
    source_relation: str
    sha256: str
    elements: Mapping[str, ElementFacts]


def load_library_facts(path: Path) -> LibraryFacts:
    data = path.read_bytes()
    raw = json.loads(data)
    elements = {
        de_id: ElementFacts(
            label=e["label"] or de_id,
            definition=e["definition"] or "",
            pv_count=int(e["pv_count"]),
            members=None if e["members"] is None
            else tuple((code, label) for code, label in e["members"]),
        )
        for de_id, e in raw["data_elements"].items()
    }
    return LibraryFacts(
        raw["release"], raw["source_relation"], hashlib.sha256(data).hexdigest(), elements
    )


class FactsConceptSource:
    """`ConceptSource` over exported library facts: labels only, no ancestry."""

    def __init__(self, facts: LibraryFacts) -> None:
        self._labels = {
            code: label
            for e in facts.elements.values() for code, label in (e.members or ())
        }

    def name_synonyms(self, code: str) -> ConceptDetail | None:
        label = self._labels.get(code)
        return ConceptDetail(code, label) if label is not None else None

    def ancestors(self, code: str) -> list[ConceptAncestor]:
        return []


def qualified_name(product: str) -> str:
    return f"partner_bench.{product}"


def target_model(
    sidecar: Sidecar,
    products: tuple[ProductSpec, ...],
    resolution: BindingResolution,
) -> tuple[NormalizedTargetModel, PhysicalSchemaBinding]:
    entities = [
        TargetEntityDecl(
            ref=TargetEntityRef(
                target_model_id=_MODEL_ID,
                qualified_name=qualified_name(p.pin.product),
                kind=TargetArtifactKind.TABLE_ROW,
            ),
            properties=_properties(resolution.approved(), p.pin.table),
            declared_joins=_declared_joins(p),
        )
        for p in products
    ]
    model = NormalizedTargetModel(descriptor=_descriptor(sidecar), entities=entities)
    binding = PhysicalSchemaBinding(
        catalog=sidecar.catalog,
        schema=sidecar.schema,
        entity_tables=tuple(
            EntityTableBinding(qualified_name(p.pin.product), p.pin.table)
            for p in products
        ),
        model_role=sidecar.model_role,
    )
    return model, binding


def _properties(
    approved: tuple[SuppliedBinding, ...], table: str
) -> list[TargetPropertyDecl]:
    props: list[TargetPropertyDecl] = []
    for b in approved:
        if b.table != table:
            continue
        bound = b.de_id if b.strategy == "pass_through" else None
        props.append(TargetPropertyDecl(
            name=b.column, type=b.physical_type, nullable=True, vocabulary_binding=bound
        ))
        props.extend(
            TargetPropertyDecl(name=c, type="STRING", nullable=True)
            for c in b.companions
        )
    return props


def _declared_joins(product: ProductSpec) -> list[DeclaredJoinDecl]:
    return [
        DeclaredJoinDecl(
            referenced_entity=qualified_name(rel["target_product_id"]),
            join_keys=list(zip(rel["source_columns"], rel["target_columns"])),
            evidence="declared",
        )
        for rel in product.raw.get("relationships", [])
    ]


def _descriptor(sidecar: Sidecar) -> TargetModelDescriptor:
    return TargetModelDescriptor(
        target_model_id=_MODEL_ID,
        target_model_version=sidecar.version,
        display_name="partner benchmark arm S scoped products",
        completeness=SemanticCompletenessAnnotations(
            structure=SemanticCompleteness.COMPLETE,
            obligations=SemanticCompleteness.NONE,
            vocabulary_bindings=SemanticCompleteness.NONE,
            semantic_aliases=SemanticCompleteness.NONE,
            terms=SemanticCompleteness.NONE,
        ),
    )


def concept_fields(
    approved: tuple[SuppliedBinding, ...], facts: LibraryFacts
) -> tuple[list[ConceptFieldSpec], list[str], list[str]]:
    """Value sets to materialize, plus unloaded findings and reference-only notes."""
    fields: list[ConceptFieldSpec] = []
    findings: list[str] = []
    notes: list[str] = []
    for b in approved:
        if b.strategy != "pass_through" or b.de_id is None:
            continue
        element = facts.elements.get(b.de_id)
        if element is None:
            findings.append(
                f"value set unloaded: {b.de_id} ({b.relation}.{b.column}) "
                f"is absent from library facts {facts.release}"
            )
        elif element.members is None:
            notes.append(
                f"value set reference-only: {b.de_id} has {element.pv_count} members "
                "above the listing limit; context points at the membership relation"
            )
        else:
            codes = tuple(code for code, _ in element.members)
            fields.append(ConceptFieldSpec(qualified_name(b.product), b.column, codes))
    return fields, findings, notes


def property_text(binding: SuppliedBinding, column: str, facts: LibraryFacts) -> str:
    element = facts.elements.get(binding.de_id or "")
    base = element.label if element else binding.library_id.removesuffix("-v1")
    base = base.replace("-", " ").replace("_", " ")
    if column.endswith("_label"):
        return f"{base} label text"
    if column.endswith("_mapping_type"):
        return f"{base} mapping type"
    return f"{base}: {element.definition}" if element and element.definition else base

"""ManifestTargetAdapter — first concrete adapter, target-model-agnostic."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

from sema.models.target.completeness import SemanticCompletenessAnnotations
from sema.models.target.context_card import TargetContextCard
from sema.models.target.descriptor import TargetModelDescriptor
from sema.models.target.entity import TargetEntityDecl
from sema.models.target.obligation import TargetObligationDecl
from sema.models.target.refs import TargetEntityRef, TargetPropertyRef, VocabularyRef
from sema.models.target.term import TargetTermDecl
from sema.models.target.vocab_binding import VocabularyBindingDecl
from sema.targets.adapters.manifest_models import (
    ManifestEntity,
    ParsedManifest,
)
from sema.targets.adapters.manifest_parser import parse_manifest_file
from sema.targets.adapters.manifest_utils import (
    descriptor_completeness,
    filter_inline_term_vocabularies,
    inline_term_vocab_names,
    supplied_card,
    synthesized_card,
    to_inline_terms,
    to_obligation,
    to_target_entity,
    to_target_entity_ref,
    to_vocabulary_bindings,
)
from sema.targets.registry import register_target_adapter

MANIFEST_ADAPTER_ID = "manifest"
MANIFEST_REGISTRY_TARGET_MODEL_ID = "manifest"


class ManifestTargetAdapter:
    """Loads target ontology declarations from a YAML/JSON manifest file."""

    def __init__(self, manifest_path: Path) -> None:
        self._manifest_path = Path(manifest_path)
        self._parsed: ParsedManifest = parse_manifest_file(self._manifest_path)
        self._target_model_id = self._parsed.descriptor.target_model_id
        self._entities_by_qname: dict[str, ManifestEntity] = {
            e.qualified_name: e for e in self._parsed.entities
        }

    @property
    def parsed_manifest(self) -> ParsedManifest:
        return self._parsed

    def describe(self) -> TargetModelDescriptor:
        completeness = self._descriptor_completeness()
        return TargetModelDescriptor(
            target_model_id=self._parsed.descriptor.target_model_id,
            target_model_version=self._parsed.descriptor.target_model_version,
            display_name=self._parsed.descriptor.display_name,
            owner=self._parsed.descriptor.owner,
            vocabulary_release=self._parsed.descriptor.vocabulary_release,
            completeness=completeness,
        )

    def discover_entities(self) -> Iterable[TargetEntityRef]:
        refs = [
            to_target_entity_ref(e, self._target_model_id)
            for e in self._parsed.entities
        ]
        return sorted(refs, key=lambda r: r.qualified_name)

    def load_entity(self, ref: TargetEntityRef) -> TargetEntityDecl:
        manifest_entity = self._lookup_entity(ref)
        return to_target_entity(
            self._parsed,
            manifest_entity,
            self._target_model_id,
            self._descriptor_completeness(),
            inline_term_vocab_names(self._parsed),
        )

    def load_obligation(self, ref: TargetEntityRef) -> TargetObligationDecl:
        manifest_entity = self._lookup_entity(ref)
        return to_obligation(manifest_entity)

    def load_vocabulary_bindings(
        self, ref: TargetPropertyRef
    ) -> Iterable[VocabularyBindingDecl]:
        manifest_entity = self._entities_by_qname.get(ref.entity_ref.qualified_name)
        if manifest_entity is None:
            return []
        for prop in manifest_entity.properties:
            if prop.name == ref.property_name:
                return to_vocabulary_bindings(
                    self._parsed, manifest_entity, prop, self._target_model_id
                )
        return []

    def load_context_card(self, ref: TargetEntityRef) -> TargetContextCard:
        manifest_entity = self._lookup_entity(ref)
        if manifest_entity.context_card is not None:
            return supplied_card(manifest_entity, self._target_model_id)
        return synthesized_card(manifest_entity, self._target_model_id)

    def iter_terms(self, vocabulary_ref: VocabularyRef) -> Iterator[TargetTermDecl]:
        if not filter_inline_term_vocabularies(self._parsed, vocabulary_ref.name):
            raise NotImplementedError(
                f"manifest adapter does not inline terms for vocabulary "
                f"{vocabulary_ref.name!r}; treated as EXTERNAL"
            )
        return iter(to_inline_terms(self._parsed, vocabulary_ref.name))

    def _descriptor_completeness(self) -> SemanticCompletenessAnnotations:
        return descriptor_completeness(self._parsed)

    def _lookup_entity(self, ref: TargetEntityRef) -> ManifestEntity:
        manifest_entity = self._entities_by_qname.get(ref.qualified_name)
        if manifest_entity is None:
            raise KeyError(
                f"manifest does not declare entity {ref.qualified_name!r}; "
                f"declared={sorted(self._entities_by_qname)}"
            )
        return manifest_entity


def register_manifest_adapter() -> None:
    """Register `ManifestTargetAdapter` under the manifest sentinel.

    The manifest adapter is target-model-agnostic: any manifest file's
    descriptor supplies the actual `target_model_id` at runtime. The
    registry entry uses a sentinel `target_model_id` so a single class
    serves every manifest. Idempotent under the registry's overlap rule
    because the supported_versions wildcard re-registration would
    overlap; this helper short-circuits if already registered.
    """
    from sema.targets.registry import _REGISTRY

    key = (MANIFEST_ADAPTER_ID, MANIFEST_REGISTRY_TARGET_MODEL_ID)
    if any(r.cls is ManifestTargetAdapter for r in _REGISTRY._by_key.get(key, ())):
        return
    register_target_adapter(
        adapter_id=MANIFEST_ADAPTER_ID,
        target_model_id=MANIFEST_REGISTRY_TARGET_MODEL_ID,
        supported_versions="",
        wildcard_target_model_id=True,
    )(ManifestTargetAdapter)


register_manifest_adapter()


__all__ = [
    "ManifestTargetAdapter",
    "register_manifest_adapter",
    "MANIFEST_ADAPTER_ID",
    "MANIFEST_REGISTRY_TARGET_MODEL_ID",
]

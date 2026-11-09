"""The product-spec format: this package's generic input contract for supplied bindings.

A source supplies one JSON document per data product. Arm S pins each document by sha256
in its sidecar; the library extract reads the element ids the documents declare. Every
module reads a document only through ``parse_product_spec``. The keys read are:

- ``product_id``: the product's id; other products' relationships name it.
- ``entities[]``: ``grain`` (prose) and ``key_columns``.
- ``fields[]``: ``column_id`` (the library's id for the column), ``column_name`` (the
  physical column), ``physical_type`` and ``strategy``; optionally ``de_id`` (the library
  element id), ``concept_name``, ``label_column`` and ``mapping_type_column``. Strategy
  ``pass_through`` marks a governed value column: it holds a library member id, and its
  declared companions hold that member's label and mapping type.
- ``relationships[]``: ``target_product_id``, ``source_columns``, ``target_columns`` and
  ``cardinality``.

Other keys are ignored. A document that breaks this shape is refused at load.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError

GOVERNED_STRATEGY = "pass_through"


class ProductSpecError(ValueError):
    """A product-spec document does not follow the declared format."""


class CompanionRole(StrEnum):
    LABEL = "label"
    MAPPING_TYPE = "mapping_type"


class _Doc(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")


class SpecField(_Doc):
    column_id: str
    column_name: str
    physical_type: str
    strategy: str
    de_id: str | None = None
    concept_name: str | None = None
    label_column: str | None = None
    mapping_type_column: str | None = None

    @property
    def governed(self) -> bool:
        return self.strategy == GOVERNED_STRATEGY

    @property
    def companions(self) -> tuple[tuple[CompanionRole, str], ...]:
        """Companion columns by declared role; only a governed column has companions."""
        if not self.governed:
            return ()
        declared = (
            (CompanionRole.LABEL, self.label_column),
            (CompanionRole.MAPPING_TYPE, self.mapping_type_column),
        )
        return tuple((role, column) for role, column in declared if column)


class SpecEntity(_Doc):
    grain: str = ""
    key_columns: tuple[str, ...] = ()


class SpecRelationship(_Doc):
    target_product_id: str
    source_columns: tuple[str, ...]
    target_columns: tuple[str, ...]
    cardinality: str


class ProductSpecDoc(_Doc):
    product_id: str
    entities: tuple[SpecEntity, ...] = ()
    fields: tuple[SpecField, ...]
    relationships: tuple[SpecRelationship, ...] = ()


def parse_product_spec(data: bytes, source: Path) -> ProductSpecDoc:
    try:
        return ProductSpecDoc.model_validate_json(data)
    except ValidationError as exc:
        raise ProductSpecError(
            f"product spec {source} does not follow the declared format: {exc}"
        ) from exc


def load_product_spec(path: Path) -> ProductSpecDoc:
    return parse_product_spec(path.read_bytes(), path)

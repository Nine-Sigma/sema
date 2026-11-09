"""Supplied governed bindings for arm S, read from the sidecar's pinned product specs.

The specs follow the package's product-spec format (``product_spec``). They are
operator-supplied (assisted) inputs, never discovery output. Each
binding keeps its library identity, spec provenance and authority. A binding is
executable only when the sidecar approval policy admits its strategy and the
catalog holds its column and companions with a matching type; anything else is
refused and reported as a finding.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from showcase.ontology_query_benchmark.profile import Attribution
from showcase.ontology_query_benchmark.product_spec import (
    CompanionRole,
    ProductSpecDoc,
    SpecField,
    parse_product_spec,
)


class SpecPinError(RuntimeError):
    """A supplied product spec does not match its pinned sha256."""


class BindingStatus(StrEnum):
    APPROVED = "approved_supplied"
    REFUSED = "refused"


@dataclass(frozen=True)
class InputFile:
    role: str
    path: str
    sha256: str
    attribution: Attribution


@dataclass(frozen=True)
class SpecPin:
    product: str
    table: str
    path: Path
    sha256: str
    attribution: Attribution


@dataclass(frozen=True)
class Sidecar:
    path: Path
    version: str
    sha256: str
    attribution: Attribution
    authority: str
    executable_strategies: frozenset[str]
    model_role: str
    concept_vocabulary: str
    catalog: str
    schema: str
    specs: tuple[SpecPin, ...]
    conventions_path: Path
    conventions_file_sha256: str
    conventions_sha256: str
    spec_name: str
    target_model_id: str
    target_namespace: str
    target_display_name: str
    grain_product: str
    grain_columns: tuple[str, ...]
    surrogate_row_id: str
    grain_note: str
    rules_scope: str
    membership_relation: str
    membership_sql: str
    catalog_companion_suffix: str
    library_id_suffix: str
    binding_note: str
    hierarchy_heading: str
    hierarchy_reference: tuple[str, ...]
    prompt_frame_chars: int
    value_list_limit: int

    def relation(self, table: str) -> str:
        return f"{self.catalog}.{self.schema}.{table}"

    def input_files(self) -> tuple[InputFile, ...]:
        own = InputFile("sidecar", str(self.path), self.sha256, self.attribution)
        return own, *(
            InputFile(f"product_spec:{p.product}", str(p.path), p.sha256, p.attribution)
            for p in self.specs
        )


@dataclass(frozen=True)
class ProductSpec:
    pin: SpecPin
    doc: ProductSpecDoc


@dataclass(frozen=True)
class SuppliedBinding:
    library_id: str
    de_id: str | None
    concept_name: str | None
    product: str
    table: str
    relation: str
    column: str
    physical_type: str
    strategy: str
    governed: bool
    companions: tuple[tuple[CompanionRole, str], ...]
    catalog_companions: tuple[str, ...]
    spec: SpecPin
    authority: str
    status: BindingStatus
    reason: str

    @property
    def companion_columns(self) -> tuple[str, ...]:
        return tuple(column for _, column in self.companions)

    @property
    def columns(self) -> tuple[str, ...]:
        return (self.column, *self.companion_columns, *self.catalog_companions)


@dataclass(frozen=True)
class BindingResolution:
    bindings: tuple[SuppliedBinding, ...]
    findings: tuple[str, ...]

    def approved(self) -> tuple[SuppliedBinding, ...]:
        return tuple(b for b in self.bindings if b.status is BindingStatus.APPROVED)

    def by_column(self, table: str, column: str) -> SuppliedBinding | None:
        return next(
            (b for b in self.bindings if (b.table, b.column) == (table, column)), None
        )


def load_sidecar(path: Path, harness_root: Path) -> Sidecar:
    """Parse the sidecar; spec and conventions paths are relative to ``harness_root``."""
    data = path.read_bytes()
    raw = json.loads(data)
    root, grain, target = harness_root, raw["grain"], raw["target_model"]
    return Sidecar(
        path=path,
        version=raw["sidecar_version"],
        sha256=hashlib.sha256(data).hexdigest(),
        attribution=Attribution(raw["attribution"]),
        authority=raw["authority"],
        executable_strategies=frozenset(raw["approval_policy"]["executable_strategies"]),
        model_role=raw["model_role"],
        concept_vocabulary=raw["concept_vocabulary"],
        catalog=raw["physical"]["catalog"],
        schema=raw["physical"]["schema"],
        specs=tuple(
            SpecPin(s["product"], s["table"], root / s["path"], s["sha256"],
                    Attribution(s["attribution"]))
            for s in raw["specs"]
        ),
        conventions_path=root / raw["conventions"]["path"],
        conventions_file_sha256=raw["conventions"]["file_sha256"],
        conventions_sha256=raw["conventions"]["conventions_sha256"],
        spec_name=raw["spec_name"],
        target_model_id=target["id"],
        target_namespace=target["namespace"],
        target_display_name=target["display_name"],
        grain_product=grain["product"],
        grain_columns=tuple(grain["columns"]),
        surrogate_row_id=grain["surrogate_row_id"],
        grain_note=grain["note"],
        rules_scope=raw["rules_scope"],
        membership_relation=raw["membership_relation"],
        membership_sql=raw["membership_sql"],
        catalog_companion_suffix=raw["catalog_companion_suffix"],
        library_id_suffix=raw["library_id_suffix"],
        binding_note=raw["binding_note"],
        hierarchy_heading=raw["hierarchy_heading"],
        hierarchy_reference=tuple(raw["hierarchy_reference"]),
        prompt_frame_chars=int(raw["prompt_frame_chars"]),
        value_list_limit=int(raw["value_list_limit"]),
    )


def load_products(sidecar: Sidecar) -> tuple[ProductSpec, ...]:
    products = []
    for pin in sidecar.specs:
        data = pin.path.read_bytes()
        actual = hashlib.sha256(data).hexdigest()
        if actual != pin.sha256:
            raise SpecPinError(
                f"{pin.product} spec {pin.path} sha256 {actual} != pinned {pin.sha256}"
            )
        products.append(ProductSpec(pin, parse_product_spec(data, pin.path)))
    return tuple(products)


def resolve_bindings(
    sidecar: Sidecar,
    products: tuple[ProductSpec, ...],
    catalog: Mapping[str, Mapping[str, str]],
) -> BindingResolution:
    """Classify every spec field against the approval policy and the catalog.

    ``catalog`` maps table name to ``{column: data_type}`` for the sidecar schema.
    """
    bindings: list[SuppliedBinding] = []
    findings: list[str] = []
    for product in products:
        columns = catalog.get(product.pin.table, {})
        counts = Counter(f.column_name for f in product.doc.fields)
        for field in product.doc.fields:
            reason = _refusal(field, columns, sidecar, counts)
            binding = _binding(sidecar, product.pin, field, reason, columns)
            bindings.append(binding)
            if reason:
                findings.append(
                    f"binding refused: {binding.relation}.{binding.column} "
                    f"({binding.library_id}): {reason}"
                )
    return BindingResolution(tuple(bindings), tuple(findings))


def _refusal(
    field: SpecField,
    columns: Mapping[str, str],
    sidecar: Sidecar,
    counts: Counter[str],
) -> str:
    column = field.column_name
    if counts[column] > 1:
        return f"ambiguous: {counts[column]} spec fields bind this column"
    if field.strategy not in sidecar.executable_strategies:
        return f"strategy {field.strategy!r} is not approved by the policy"
    for name in (column, *(c for _, c in field.companions)):
        if name not in columns:
            return f"no physical column {name!r} in the catalog"
    actual = columns[column].lower()
    if actual != field.physical_type.lower():
        return f"type mismatch: spec {field.physical_type} vs catalog {actual}"
    return ""


def _binding(
    sidecar: Sidecar,
    pin: SpecPin,
    field: SpecField,
    reason: str,
    columns: Mapping[str, str],
) -> SuppliedBinding:
    code_column = f"{field.column_name}{sidecar.catalog_companion_suffix}"
    return SuppliedBinding(
        library_id=field.column_id,
        de_id=field.de_id,
        concept_name=field.concept_name,
        product=pin.product,
        table=pin.table,
        relation=sidecar.relation(pin.table),
        column=field.column_name,
        physical_type=field.physical_type,
        strategy=field.strategy,
        governed=field.governed,
        companions=field.companions,
        catalog_companions=(code_column,) if code_column in columns else (),
        spec=pin,
        authority=sidecar.authority,
        status=BindingStatus.REFUSED if reason else BindingStatus.APPROVED,
        reason=reason,
    )

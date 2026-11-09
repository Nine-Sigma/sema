"""Section renderers for arm S context.

Physical assets and joins reuse SEMA's nl2sql section renderers. The added
sections render supplied binding evidence, the SCO's governed values (complete
lists only, otherwise an executable membership reference), grain and the supplied
hierarchy relation facts. Business rules are referenced by hash, never restated.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass, field

from sema.consumers.nl2sql.prompting_utils import _add_join_section, _add_tables_and_columns
from sema.models.context import SemanticContextObject
from showcase.ontology_query_benchmark.product_spec import SpecRelationship
from showcase.ontology_query_benchmark.sema_bindings import (
    BindingResolution,
    ProductSpec,
    Sidecar,
    SuppliedBinding,
)
from showcase.ontology_query_benchmark.sema_build_utils import (
    Finding,
    LibraryFacts,
    absent_elements,
)

TableColumn = tuple[str, str]


@dataclass
class Rendering:
    sco: SemanticContextObject
    parts: list[str] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    listed: dict[TableColumn, int] = field(default_factory=dict)

    @property
    def tables(self) -> tuple[str, ...]:
        return tuple(a.table for a in self.sco.physical_assets)

    @property
    def text(self) -> str:
        return "\n".join(self.parts).strip() + "\n"


def render(
    rendering: Rendering,
    sidecar: Sidecar,
    products: tuple[ProductSpec, ...],
    resolution: BindingResolution,
    facts: LibraryFacts,
) -> Rendering:
    denied = {(p.pin.table, c) for p in products for c in p.doc.exposure.denied_columns}
    rendering.sco = _without_denied(rendering.sco, denied)
    if not rendering.tables:
        rendering.findings.append(Finding("retrieval returned no physical asset for the question"))
    selected = [
        _without_denied_companions(b, denied) for b in resolution.approved()
        if b.table in rendering.tables and (b.table, b.column) not in denied
    ]
    rendering.findings.extend(absent_elements(tuple(selected), facts))
    _add_tables_and_columns(rendering.parts, rendering.sco)
    _add_unbound(rendering.parts, rendering.sco, selected)
    _add_join_section(rendering.parts, rendering.sco)
    _add_bindings(rendering.parts, selected, sidecar, facts)
    _add_values(rendering, selected, _governed_values(rendering.sco), sidecar, facts)
    _add_grain(rendering.parts, rendering.tables, products, sidecar)
    _add_rules(rendering.parts, sidecar)
    return rendering


def _without_denied(
    sco: SemanticContextObject, denied: set[TableColumn]
) -> SemanticContextObject:
    """Apply the product exposure policy; the graph holds denied columns, S never shows them."""
    assets = [
        a.model_copy(update={"columns": [c for c in a.columns if (a.table, c) not in denied]})
        for a in sco.physical_assets
    ]
    values = [g for g in sco.governed_values if (g.table, g.column) not in denied]
    return sco.model_copy(update={"physical_assets": assets, "governed_values": values})


def _without_denied_companions(b: SuppliedBinding, denied: set[TableColumn]) -> SuppliedBinding:
    return dataclasses.replace(
        b,
        companions=tuple((r, c) for r, c in b.companions if (b.table, c) not in denied),
        catalog_companions=tuple(c for c in b.catalog_companions if (b.table, c) not in denied),
    )


def _add_unbound(
    parts: list[str], sco: SemanticContextObject, selected: list[SuppliedBinding]
) -> None:
    bound = {(b.table, c) for b in selected for c in b.columns}
    for a in sco.physical_assets:
        unbound = [c for c in a.columns if (a.table, c) not in bound]
        if unbound:
            parts.append(
                f"  {a.catalog}.{a.schema}.{a.table} no governed binding: {', '.join(unbound)}"
            )


def _add_bindings(
    parts: list[str],
    selected: list[SuppliedBinding],
    sidecar: Sidecar,
    facts: LibraryFacts,
) -> None:
    if not selected:
        return
    pins = ", ".join(f"{p.product} spec sha256 {p.sha256[:12]}" for p in sidecar.specs)
    parts.append(
        f"\nAPPROVED BINDINGS (supplied: {sidecar.authority}; {pins}). {sidecar.binding_note}"
    )
    for relation in dict.fromkeys(b.relation for b in selected):
        parts.append(f"  {relation}:")
        parts.extend(
            f"    {_binding_line(b, facts, sidecar)}" for b in selected if b.relation == relation
        )


def _binding_line(b: SuppliedBinding, facts: LibraryFacts, sidecar: Sidecar) -> str:
    element = facts.elements.get(b.de_id or "")
    label = element.label if element else b.library_id
    line = f"{b.column} <- {label} [{b.de_id or b.library_id}]"
    if element and element.definition:
        line += f": {element.definition}"
    if b.catalog_companions:
        companions = ", ".join(b.catalog_companions)
        line += f" Catalog companion, not in {sidecar.spec_name}: {companions}."
    return line


def _governed_values(sco: SemanticContextObject) -> dict[TableColumn, dict[str, str]]:
    values: dict[TableColumn, dict[str, str]] = {}
    for g in sco.governed_values:
        listed = values.setdefault((g.table, g.column), {})
        listed.update((v["code"], v["label"]) for v in g.values)
    return values


def _add_values(
    rendering: Rendering,
    selected: list[SuppliedBinding],
    retrieved: Mapping[TableColumn, dict[str, str]],
    sidecar: Sidecar,
    facts: LibraryFacts,
) -> None:
    governed = [b for b in selected if b.governed and b.de_id]
    if not governed:
        return
    template = sidecar.membership_sql.format(relation=sidecar.membership_relation, element_id="E")
    rendering.parts.append(
        "\nGOVERNED VALUES (complete lists only; code = label). Unlisted sets: "
        f"membership query E = {template}"
    )
    for b in governed:
        de_id = str(b.de_id)
        element = facts.elements.get(de_id)
        got = retrieved.get((b.table, b.column), {})
        reference = f"membership query {de_id}"
        if element is None:
            rendering.parts.append(f"  {b.column}: no governed value set loaded ({de_id})")
        elif element.pv_count > sidecar.value_list_limit:
            rendering.parts.append(
                f"  {b.column}: {element.pv_count} members, above the listing limit "
                f"({reference})"
            )
        elif set(got) == {c for c, _ in element.members}:
            listing = "; ".join(f"{code} = {got[code]}" for code, _ in element.members)
            rendering.parts.append(f"  {b.column}: {listing}")
            rendering.listed[(b.table, b.column)] = len(element.members)
        else:
            rendering.findings.append(
                Finding(
                    f"value set incomplete: retrieved {len(got)} of {element.pv_count} "
                    f"members for {de_id} ({b.column}); listing withheld",
                    b.relation,
                    b.column,
                )
            )
            rendering.parts.append(
                f"  {b.column}: {element.pv_count} members, not listed ({reference})"
            )


def _add_grain(
    parts: list[str],
    tables: tuple[str, ...],
    products: tuple[ProductSpec, ...],
    sidecar: Sidecar,
) -> None:
    chosen = [p for p in products if p.pin.table in tables]
    if not chosen:
        return
    parts.append(f"\nGRAIN AND KEYS ({sidecar.spec_name} product specs):")
    for p in chosen:
        for entity in p.doc.entities:
            keys = ", ".join(entity.key_columns)
            parts.append(f"  {sidecar.relation(p.pin.table)}: {entity.grain}; key {keys}")
        for rel in p.doc.relationships:
            pairs = " AND ".join(
                f"{p.pin.table}.{s} = {_table_of(products, rel)}.{t}"
                for s, t in zip(rel.source_columns, rel.target_columns)
            )
            parts.append(f"  join {pairs} ({rel.cardinality})")
        if p.pin.product == sidecar.grain_product:
            columns = ", ".join(sidecar.grain_columns)
            note = sidecar.grain_note.format(
                columns=columns, surrogate_row_id=sidecar.surrogate_row_id
            )
            parts.append(f"  {note}")


def _table_of(products: tuple[ProductSpec, ...], rel: SpecRelationship) -> str:
    return next(
        (p.pin.table for p in products if p.pin.product == rel.target_product_id),
        rel.target_product_id,
    )


def _add_rules(parts: list[str], sidecar: Sidecar) -> None:
    parts.append(
        "\nBUSINESS RULES: the workload conventions supplied with this request "
        f"(sha256 {sidecar.conventions_sha256[:12]}) govern {sidecar.rules_scope}; "
        "they are not restated here."
    )
    parts.append(f"\n{sidecar.hierarchy_heading} (supplied):")
    parts.extend(f"  - {line}" for line in sidecar.hierarchy_reference)

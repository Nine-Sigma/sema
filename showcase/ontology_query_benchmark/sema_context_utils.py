"""Section renderers for arm S context.

Physical assets and joins reuse SEMA's nl2sql section renderers. The added
sections render supplied binding evidence, the SCO's governed values (complete
lists only, otherwise an executable membership reference), grain and the supplied
hierarchy relation facts. Business rules are referenced by hash, never restated.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from sema.consumers.nl2sql.prompting_utils import _add_join_section, _add_tables_and_columns
from sema.models.context import SemanticContextObject
from showcase.ontology_query_benchmark.sema_bindings import (
    BindingResolution,
    ProductSpec,
    Sidecar,
    SuppliedBinding,
)
from showcase.ontology_query_benchmark.sema_build_utils import LibraryFacts, absent_elements

MEMBERSHIP_SQL = (
    "SELECT permissible_value_prt_id, permissible_value_prt_preferred_label "
    "FROM {relation} WHERE data_element_prt_id = '{de_id}'"
)

TableColumn = tuple[str, str]


@dataclass
class Rendering:
    sco: SemanticContextObject
    parts: list[str] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)
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
    if not rendering.tables:
        rendering.findings.append("retrieval returned no physical asset for the question")
    selected = [b for b in resolution.approved() if b.table in rendering.tables]
    rendering.findings.extend(absent_elements(tuple(selected), facts))
    _add_tables_and_columns(rendering.parts, _without_companions(rendering.sco, selected))
    _add_join_section(rendering.parts, rendering.sco)
    _add_bindings(rendering.parts, selected, sidecar, facts)
    _add_values(rendering, selected, _governed_values(rendering.sco), sidecar, facts)
    _add_grain(rendering.parts, rendering.tables, products, sidecar)
    _add_rules(rendering.parts, sidecar)
    return rendering


def _without_companions(
    sco: SemanticContextObject, selected: list[SuppliedBinding]
) -> SemanticContextObject:
    """Drop V_label / V_mapping_type columns; the bindings header names the pattern."""
    implied = {(b.table, c) for b in selected for c in b.companions}
    assets = [
        a.model_copy(update={"columns": [c for c in a.columns if (a.table, c) not in implied]})
        for a in sco.physical_assets
    ]
    return sco.model_copy(update={"physical_assets": assets})


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
        f"\nAPPROVED BINDINGS (supplied: {sidecar.authority}; {pins}). A governed value "
        "column V holds the permissible-value id; V_label holds its label and "
        "V_mapping_type its mapping type."
    )
    for relation in dict.fromkeys(b.relation for b in selected):
        parts.append(f"  {relation}:")
        parts.extend(
            f"    {_binding_line(b, facts)}" for b in selected if b.relation == relation
        )


def _binding_line(b: SuppliedBinding, facts: LibraryFacts) -> str:
    element = facts.elements.get(b.de_id or "")
    label = element.label if element else b.library_id
    line = f"{b.column} <- {label} [{b.de_id or b.library_id}]"
    if element and element.definition:
        line += f": {element.definition}"
    if b.catalog_companions:
        line += f" Catalog companion, not in CPD: {', '.join(b.catalog_companions)}."
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
    governed = [b for b in selected if b.strategy == "pass_through" and b.de_id]
    if not governed:
        return
    template = MEMBERSHIP_SQL.format(relation=sidecar.membership_relation, de_id="E")
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
                f"value set incomplete: retrieved {len(got)} of {element.pv_count} "
                f"members for {de_id} ({b.column}); listing withheld"
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
    parts.append("\nGRAIN AND KEYS (CPD product specs):")
    for p in chosen:
        for entity in p.raw.get("entities", []):
            keys = ", ".join(entity.get("key_columns", []))
            parts.append(f"  {sidecar.relation(p.pin.table)}: {entity.get('grain')}; key {keys}")
        for rel in p.raw.get("relationships", []):
            pairs = " AND ".join(
                f"{p.pin.table}.{s} = {_table_of(products, rel)}.{t}"
                for s, t in zip(rel["source_columns"], rel["target_columns"])
            )
            parts.append(f"  join {pairs} ({rel['cardinality']})")
        if p.pin.product == sidecar.tumor_product:
            parts.append(
                f"  Clinical tumor grain columns (supplied): {', '.join(sidecar.tumor_grain)}. "
                f"{sidecar.surrogate_row_id} is the surrogate source-row id, not the tumor grain."
            )


def _table_of(products: tuple[ProductSpec, ...], rel: Mapping[str, str]) -> str:
    return next(
        (p.pin.table for p in products if p.pin.product == rel["target_product_id"]),
        rel["target_product_id"],
    )


def _add_rules(parts: list[str], sidecar: Sidecar) -> None:
    parts.append(
        "\nBUSINESS RULES: the workload conventions supplied with this request "
        f"(sha256 {sidecar.conventions_sha256[:12]}) govern patient counting, tumor "
        "identity, null handling and attribute resolution; they are not restated here."
    )
    parts.append("\nOMOP HIERARCHY REFERENCE (supplied):")
    parts.extend(f"  - {line}" for line in sidecar.hierarchy_reference)

"""Section renderers for arm S context.

Physical assets and joins reuse SEMA's nl2sql section renderers. The added
sections render supplied binding evidence, governed values (complete lists only,
otherwise an executable membership reference), grain and the pinned hierarchy
reference. Business rules are referenced by hash, never restated.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from sema.consumers.nl2sql.prompting_utils import _add_join_section, _add_tables_and_columns
from sema.models.context import SemanticCandidateSet, SemanticContextObject
from showcase.partner_timbr_benchmark.sema_bindings import (
    BindingResolution,
    ProductSpec,
    Sidecar,
    SuppliedBinding,
)
from showcase.partner_timbr_benchmark.sema_build_utils import LibraryFacts

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
    candidates: SemanticCandidateSet,
    sidecar: Sidecar,
    products: tuple[ProductSpec, ...],
    resolution: BindingResolution,
    facts: LibraryFacts,
) -> Rendering:
    if not rendering.tables:
        rendering.findings.append("retrieval returned no physical asset for the question")
    _add_tables_and_columns(rendering.parts, rendering.sco)
    _add_join_section(rendering.parts, rendering.sco)
    selected = [b for b in resolution.approved() if b.table in rendering.tables]
    _add_bindings(rendering.parts, selected, sidecar, facts)
    _add_values(rendering, selected, _retrieved_values(candidates), sidecar, facts)
    _add_grain(rendering.parts, rendering.tables, products, sidecar)
    _add_rules(rendering.parts, sidecar)
    return rendering


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


def _retrieved_values(
    candidates: SemanticCandidateSet,
) -> dict[TableColumn, dict[str, str]]:
    values: dict[TableColumn, dict[str, str]] = {}
    for c in candidates.candidates:
        if c.get("type") == "value":
            key = (str(c.get("table", "")), str(c.get("column", "")))
            values.setdefault(key, {})[str(c.get("code", ""))] = str(c.get("label", ""))
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
    rendering.parts.append("\nGOVERNED VALUES (complete lists only; code = label):")
    for b in governed:
        de_id = str(b.de_id)
        element = facts.elements.get(de_id)
        got = retrieved.get((b.table, b.column), {})
        reference = MEMBERSHIP_SQL.format(relation=sidecar.membership_relation, de_id=de_id)
        if element is None:
            rendering.findings.append(
                f"value set unloaded: {de_id} ({b.column}) is absent from {facts.release}"
            )
            rendering.parts.append(f"  {b.column}: no governed value set loaded ({de_id})")
        elif element.members is not None and len(element.members) > sidecar.value_list_limit:
            rendering.parts.append(
                f"  {b.column}: {element.pv_count} members, above the listing limit. "
                f"Executable membership: {reference}"
            )
        elif element.members is not None and set(got) == {c for c, _ in element.members}:
            listing = "; ".join(f"{code} = {got[code]}" for code, _ in element.members)
            rendering.parts.append(f"  {b.column}: {listing}")
            rendering.listed[(b.table, b.column)] = len(element.members)
        else:
            if element.members is not None:
                rendering.findings.append(
                    f"value set incomplete: retrieved {len(got)} of {element.pv_count} "
                    f"members for {de_id} ({b.column}); listing withheld"
                )
            rendering.parts.append(
                f"  {b.column}: {element.pv_count} members, not listed. "
                f"Executable membership: {reference}"
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
    parts.append("\nOMOP HIERARCHY REFERENCE (pinned, executable):")
    parts.extend(f"  - {line}" for line in sidecar.hierarchy_reference)

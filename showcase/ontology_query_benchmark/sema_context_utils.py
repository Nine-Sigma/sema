"""Section renderers for arm S context.

Physical assets and joins reuse SEMA's nl2sql section renderers. The added
sections render supplied binding evidence, the SCO's governed values (complete
lists only, otherwise an executable membership reference), grain and the supplied
hierarchy relation facts. Business rules are referenced by hash, never restated.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Callable, Mapping
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


def nothing_denied(_name: str) -> bool:
    return False


@dataclass(frozen=True)
class Exposure:
    """Each product's denials on its own table, plus the run's name-global deny policy."""

    spec_denied: frozenset[TableColumn]
    is_denied: Callable[[str], bool]

    def hides(self, table: str, column: str) -> bool:
        return (table, column) in self.spec_denied or self.is_denied(column)


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
    is_denied: Callable[[str], bool] = nothing_denied,
) -> Rendering:
    exposure = Exposure(
        frozenset((p.pin.table, c) for p in products for c in p.doc.exposure.denied_columns),
        is_denied,
    )
    rendering.sco = _without_denied(rendering, exposure)
    if not rendering.tables:
        rendering.findings.append(Finding("retrieval returned no physical asset for the question"))
    selected = [
        _without_denied_companions(b, exposure) for b in resolution.approved()
        if b.table in rendering.tables and not exposure.hides(b.table, b.column)
    ]
    rendering.findings.extend(absent_elements(tuple(selected), facts))
    _add_tables_and_columns(rendering.parts, rendering.sco)
    _add_unbound(rendering.parts, rendering.sco, selected)
    _add_join_section(rendering.parts, rendering.sco)
    _add_bindings(rendering.parts, selected, sidecar, facts)
    _add_values(rendering, selected, _governed_values(rendering.sco), sidecar, facts)
    _add_grain(rendering, products, sidecar, exposure)
    _add_rules(rendering.parts, sidecar)
    return rendering


def _without_denied(rendering: Rendering, exposure: Exposure) -> SemanticContextObject:
    """Apply the exposure policy; the graph holds denied columns, S never shows them."""
    sco = rendering.sco
    assets = [
        a.model_copy(update={"columns": [c for c in a.columns if not exposure.hides(a.table, c)]})
        for a in sco.physical_assets
    ]
    values = [g for g in sco.governed_values if not exposure.hides(g.table, g.column)]
    joins = []
    for jp in sco.join_paths:
        names = [(p.left_table, p.left_column) for p in jp.join_predicates]
        names += [(p.right_table, p.right_column) for p in jp.join_predicates]
        names += [(jp.from_table, w) for w in re.findall(r"\w+", jp.sql_snippet or "")]
        if _kept(rendering, exposure, names, jp.from_table, "join path"):
            joins.append(jp)
    return sco.model_copy(
        update={"physical_assets": assets, "governed_values": values, "join_paths": joins}
    )


def _kept(
    rendering: Rendering, exposure: Exposure, names: list[TableColumn], relation: str, line: str
) -> bool:
    """A line naming a denied column is withheld whole, never rewritten, and leaves a finding."""
    hidden = dict.fromkeys(c for t, c in names if exposure.hides(t, c))
    rendering.findings.extend(
        Finding(f"{line} withheld: it names denied column {c}", relation, c) for c in hidden
    )
    return not hidden


def _without_denied_companions(b: SuppliedBinding, exposure: Exposure) -> SuppliedBinding:
    return dataclasses.replace(
        b,
        companions=tuple((r, c) for r, c in b.companions if not exposure.hides(b.table, c)),
        catalog_companions=tuple(
            c for c in b.catalog_companions if not exposure.hides(b.table, c)
        ),
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
    rendering: Rendering,
    products: tuple[ProductSpec, ...],
    sidecar: Sidecar,
    exposure: Exposure,
) -> None:
    chosen = [p for p in products if p.pin.table in rendering.tables]
    if not chosen:
        return
    rendering.parts.append(f"\nGRAIN AND KEYS ({sidecar.spec_name} product specs):")
    for p in chosen:
        table, relation = p.pin.table, sidecar.relation(p.pin.table)
        lines: list[tuple[str, list[TableColumn], str]] = []
        for entity in p.doc.entities:
            keys = ", ".join(entity.key_columns)
            names = [(table, k) for k in entity.key_columns]
            lines.append((f"  {relation}: {entity.grain}; key {keys}", names, "grain line"))
        for rel in p.doc.relationships:
            target = _table_of(products, rel)
            pairs = " AND ".join(
                f"{table}.{s} = {target}.{t}"
                for s, t in zip(rel.source_columns, rel.target_columns)
            )
            names = [(table, s) for s in rel.source_columns]
            names += [(target, t) for t in rel.target_columns]
            lines.append((f"  join {pairs} ({rel.cardinality})", names, "join line"))
        if p.pin.product == sidecar.grain_product:
            note = sidecar.grain_note.format(
                columns=", ".join(sidecar.grain_columns),
                surrogate_row_id=sidecar.surrogate_row_id,
            )
            names = [(table, c) for c in (*sidecar.grain_columns, sidecar.surrogate_row_id)]
            lines.append((f"  {note}", names, "grain note"))
        rendering.parts.extend(
            text for text, names, kind in lines
            if _kept(rendering, exposure, names, relation, kind)
        )


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

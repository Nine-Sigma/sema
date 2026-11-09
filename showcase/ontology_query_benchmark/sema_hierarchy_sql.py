"""One executable membership predicate per governed column a concept hierarchy reaches.

The predicate's shape is fixed: value column, then the member-to-standard bridge
(mapping types, then the declared replacement hop, standard concepts only), then the
closure at ``ancestor = {anchor_id}``. Every relation and column name is profile data.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from sema.connectors.databricks_utils import sql_string_literal
from sema.models.context import HierarchyAnchor
from showcase.ontology_query_benchmark.profile import ConceptHierarchy
from showcase.ontology_query_benchmark.sema_bindings import SuppliedBinding
from showcase.ontology_query_benchmark.sema_build_utils import Finding, LibraryFacts

ANCHOR = "{anchor_id}"


def membership_sql(spec: ConceptHierarchy, column: str, anchor: str | None = None) -> str:
    """With no ``anchor`` code the ancestor slot stays the {anchor_id} placeholder."""
    c, k, m = spec.closure, spec.concepts, spec.mapping
    joins: list[str] = []
    where: list[str] = []
    key, source = f"a.{c.descendant_column}", f"{c.relation} a"
    if m is not None:
        key, source = f"m.{m.source_column}", f"{m.relation} m"
        targets = [_typed(f"m.{m.target_column}", m.target_type)]
        if m.replacement_types:
            joins.append(f"LEFT JOIN {m.relation} r ON r.{m.source_column} = m.{m.target_column}"
                         f" AND r.{m.type_column} IN ({_quoted(m.replacement_types)})")  # fmt: skip
            targets.append(_typed(f"r.{m.target_column}", m.target_type))
        joins.append(f"JOIN {c.relation} a ON a.{c.descendant_column} IN ({', '.join(targets)})")
        where.append(f"m.{m.type_column} IN ({_quoted(m.mapping_types)})")
    if k.standard_column:
        joins.append(f"JOIN {k.relation} s ON s.{k.id_column} = a.{c.descendant_column}"
                     f" AND s.{k.standard_column} IN ({_quoted(k.standard_values)})")  # fmt: skip
    slot = ANCHOR if anchor is None else sql_string_literal(anchor)
    where.append(f"a.{c.ancestor_column} = {slot}")
    body = " ".join([f"SELECT {key} FROM {source}", *joins, "WHERE " + " AND ".join(where)])
    return f"{column} IN ({body})"


def add_hierarchy(
    parts: list[str],
    findings: list[Finding],
    selected: Sequence[SuppliedBinding],
    facts: LibraryFacts,
    readable: Mapping[str, bool] | None,
    anchors: Sequence[HierarchyAnchor] = (),
) -> None:
    hierarchy = facts.hierarchy
    if hierarchy is None:
        return
    columns = [b for b in selected if b.governed and _reached(b, facts)]
    blocked = [r for r in hierarchy.spec.relations if not (readable or {}).get(r, False)]
    if blocked:
        findings.extend(
            Finding(f"insufficient evidence: hierarchy relation {', '.join(blocked)} not "
                    f"readable at build; no hierarchy SQL for {b.table}.{b.column}",
                    b.relation, b.column)  # fmt: skip
            for b in columns
        )
        return
    if not columns:
        return
    spec = hierarchy.spec
    found = {(b.table, b.column): [a for a in anchors if (a.table, a.column) == (b.table, b.column)]
             for b in columns}  # fmt: skip
    parts.append(f"\nCONCEPT HIERARCHY ({spec.attribution.value}; each relation read at build):")
    line = f"  {_route(spec)} {_direction(spec)}"
    template = f"{line} Replace {ANCHOR} with the anchor concept id."
    parts.append(line if all(found.values()) else template)
    kind = "source member ids" if spec.mapping else "concept ids of the hierarchy"
    for b in columns:
        parts.append(f"  {b.table}.{b.column} holds {kind}:")
        parts.extend(_fragments(spec, b.column, found[(b.table, b.column)]))


def _fragments(
    spec: ConceptHierarchy, column: str, anchors: Sequence[HierarchyAnchor]
) -> list[str]:
    """One filled predicate per retrieved anchor; without one, the {anchor_id} template."""
    if not anchors:
        return [f"  WHERE {membership_sql(spec, column)}"]
    lines = []
    for a in anchors:
        vocabulary = f" in {a.vocabulary}" if a.vocabulary else ""
        lines.append(f"  anchor {a.label} = id {a.code}{vocabulary}; "
                     f"{a.member_count} members of {column} sit under it:")  # fmt: skip
        lines.append(f"  WHERE {membership_sql(spec, column, anchor=a.code)}")
    return lines


def _reached(binding: SuppliedBinding, facts: LibraryFacts) -> bool:
    element = facts.elements.get(binding.de_id or "")
    hierarchy = facts.hierarchy
    return bool(element and hierarchy and any(hierarchy.reaches(c) for c, _ in element.members))


def _route(spec: ConceptHierarchy) -> str:
    k, m = spec.concepts, spec.mapping
    route = ""
    if m is not None:
        route = (f"A member reaches a concept through {m.relation} where {m.type_column} in "
                 f"({_quoted(m.mapping_types)})")  # fmt: skip
        if m.replacement_types:
            replaced = _quoted(m.replacement_types)
            route += f", then its replacement where {m.type_column} in ({replaced})"
        route += "."
    if k.standard_column:
        route += (f" Only concepts with {k.relation}.{k.standard_column} in "
                  f"({_quoted(k.standard_values)}) count.")  # fmt: skip
    return route.strip()


def _direction(spec: ConceptHierarchy) -> str:
    c = spec.closure
    policy = (
        "self rows are included, so an anchor matches its own members"
        if c.self_rows else "self rows are absent, so an anchor's own members do not match"
    )
    return (f"In {c.relation}, {c.ancestor_column} is the broader concept and "
            f"{c.descendant_column} the narrower; {policy}.")  # fmt: skip


def _typed(column: str, target_type: str | None) -> str:
    """A text id column compared to typed concept ids; a non-id value matches nothing."""
    return f"TRY_CAST({column} AS {target_type})" if target_type else column


def _quoted(values: Iterable[str]) -> str:
    return ", ".join(sql_string_literal(v) for v in values)

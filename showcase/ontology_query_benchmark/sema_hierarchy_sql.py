"""One executable membership predicate per governed column a concept hierarchy reaches.

The predicate's shape is fixed: value column, then the member-to-standard bridge
(mapping types, then the declared replacement hop, standard concepts only, or, when a
loaded member's own id is a standard concept, that id), then the closure under the
anchor concept, selected by name plus the qualifiers given. An ancestor with no row in
the concepts relation has no name, so it alone is keyed by its id. Every relation and
column name is profile data; every string literal is quoted in the warehouse's SQL
dialect.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from sqlglot import exp

from sema.models.context import HierarchyAnchor
from sema.pipeline.dedup_utils import states_name, tokenize_query
from showcase.ontology_query_benchmark.profile import ConceptHierarchy
from showcase.ontology_query_benchmark.sema_bindings import SuppliedBinding
from showcase.ontology_query_benchmark.sema_build_utils import Finding, LibraryFacts

ANCHOR_ID = "{anchor_id}"
ANCHOR_NAME = "{anchor_name}"
ANCHOR_VOCABULARY = "{anchor_vocabulary}"
COLUMN = "{column}"
COUNTED = ("An anchor's member count is the column's values strictly below it; a value "
           "mapped to the anchor itself is not counted.")
ANCHORS_PER_COLUMN = 3


def membership_sql(
    spec: ConceptHierarchy, column: str, *, dialect: str, name: str | None = None,
    vocabulary: str | None = None, concept_class: str | None = None, own_ids: bool = False,
) -> str:
    """With no ``name`` the anchor's name and vocabulary stay placeholders; otherwise a
    qualifier left None adds no filter. ``own_ids`` adds the route for members whose own
    id is a standard concept."""
    anchor = _anchor_ids(spec, dialect, name, vocabulary, concept_class)
    return _membership(spec, column, f"IN ({anchor})", dialect, own_ids)


def id_membership_sql(
    spec: ConceptHierarchy, column: str, *, dialect: str, code: str | None = None,
    own_ids: bool = False,
) -> str:
    """The predicate for an ancestor with no concept row, keyed by its id; with no
    ``code`` the id stays the {anchor_id} placeholder."""
    slot = ANCHOR_ID if code is None else _literal(code, dialect)
    return _membership(spec, column, f"= {slot}", dialect, own_ids)


def _membership(
    spec: ConceptHierarchy, column: str, anchor: str, dialect: str, own_ids: bool,
) -> str:
    """``anchor`` is the condition on the closure's ancestor column."""
    c, k, m = spec.closure, spec.concepts, spec.mapping
    under = f"a.{c.ancestor_column} {anchor}"
    standard = (
        f" JOIN {k.relation} s ON s.{k.id_column} = a.{c.descendant_column}"
        f" AND s.{k.standard_column} IN ({_quoted(k.standard_values, dialect)})"
        if k.standard_column else ""
    )
    closure = f"SELECT a.{c.descendant_column} FROM {c.relation} a{standard} WHERE {under}"
    if m is None:
        return f"{column} IN ({closure})"
    targets = [_typed(f"m.{m.target_column}", m.target_type)]
    joins = []
    if m.replacement_types:
        replaced = _quoted(m.replacement_types, dialect)
        joins.append(f"LEFT JOIN {m.relation} r ON r.{m.source_column} = m.{m.target_column}"
                     f" AND r.{m.type_column} IN ({replaced})")  # fmt: skip
        targets.append(_typed(f"r.{m.target_column}", m.target_type))
    joins.append(f"JOIN {c.relation} a ON a.{c.descendant_column} IN ({', '.join(targets)})"
                 + standard)  # fmt: skip
    mapped = f"m.{m.type_column} IN ({_quoted(m.mapping_types, dialect)})"
    bridged = " ".join([f"SELECT m.{m.source_column} FROM {m.relation} m", *joins,
                        f"WHERE {mapped} AND {under}"])  # fmt: skip
    if not own_ids:
        return f"{column} IN ({bridged})"
    return f"({column} IN ({bridged}) OR {_typed(column, m.target_type)} IN ({closure}))"


def named_columns(spec: ConceptHierarchy) -> list[tuple[str, str]]:
    """Every (relation, column) the section's route, direction and predicates name."""
    c, k, m = spec.closure, spec.concepts, spec.mapping
    names = [(c.relation, c.ancestor_column), (c.relation, c.descendant_column)]
    anchor = (k.id_column, k.name_column, k.vocabulary_column, k.standard_column)
    names += [(k.relation, col) for col in anchor if col]
    if m is not None:
        names += [(m.relation, col) for col in (m.source_column, m.target_column, m.type_column)]
    return names


def add_hierarchy(
    parts: list[str],
    findings: list[Finding],
    selected: Sequence[SuppliedBinding],
    facts: LibraryFacts,
    readable: Mapping[str, bool] | None,
    anchors: Sequence[HierarchyAnchor] = (),
    *,
    question: str,
    dialect: str,
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
    spec, own = hierarchy.spec, bool(hierarchy.own_ids)
    found = {(b.table, b.column): ranked([a for a in anchors if (a.table, a.column)
                                          == (b.table, b.column)], question) for b in columns}
    parts.append(f"\nCONCEPT HIERARCHY ({spec.attribution.value}; each relation read at build):")
    vocabulary = (f" and {ANCHOR_VOCABULARY} with its vocabulary"
                  if spec.concepts.vocabulary_column else "")  # fmt: skip
    parts.append(f"  {_route(spec, dialect, own)} {_direction(spec)} Replace {COLUMN} with a "
                 f"value column below, {ANCHOR_NAME} with the anchor's quoted name{vocabulary}."
                 f" {COUNTED}")  # fmt: skip
    parts.append(f"  WHERE {membership_sql(spec, COLUMN, dialect=dialect, own_ids=own)}")
    names = hierarchy.names
    if any(parent not in names for parent, _ in hierarchy.edges):
        parts.append(f"  An ancestor with no row in {spec.concepts.relation} is listed by id;"
                     f" for it replace {ANCHOR_ID} with the quoted id instead:")  # fmt: skip
        parts.append(f"  WHERE {id_membership_sql(spec, COLUMN, dialect=dialect, own_ids=own)}")
    parts.extend(_example(spec, found, question, dialect, own, names))
    kind = "source member ids" if spec.mapping else "concept ids of the hierarchy"
    for b in columns:
        parts.append(f"  {b.table}.{b.column} holds {kind}:")
        parts.extend(_anchor_lines(b.column, found[(b.table, b.column)], dialect, names))


def hierarchy_asked(
    question: str, facts: LibraryFacts, anchors: Sequence[HierarchyAnchor],
) -> bool:
    """False only when the facts hold a hierarchy and the question names none of its
    concepts, neither one the slice holds nor a retrieved anchor."""
    if facts.hierarchy is None:
        return True
    concepts = [*facts.hierarchy.names.values(), *(a.label for a in anchors)]
    return any(states_name(question, concept) for concept in concepts)


def ranked(anchors: Sequence[HierarchyAnchor], question: str) -> list[HierarchyAnchor]:
    """By the share of the anchor label's words the question holds, then the score of the
    retrieval hit that found the anchor, then code."""
    asked = set(tokenize_query(question))

    def key(a: HierarchyAnchor) -> tuple[float, float, str]:
        words = set(tokenize_query(a.label))
        coverage = len(words & asked) / len(words) if words else 0.0
        return -coverage, -a.score, a.code

    return sorted(anchors, key=key)


def _example(
    spec: ConceptHierarchy, found: Mapping[tuple[str, str], list[HierarchyAnchor]],
    question: str, dialect: str, own_ids: bool, names: Mapping[str, str],
) -> list[str]:
    """The template filled for the best-ranked anchor across every column."""
    best = ranked([a for anchors in found.values() for a in anchors[:1]], question)
    if not best:
        return []
    a = best[0]
    sql = (membership_sql(spec, a.column, dialect=dialect, name=a.label,
                          vocabulary=a.vocabulary, own_ids=own_ids)
           if a.code in names
           else id_membership_sql(spec, a.column, dialect=dialect, code=a.code,
                                  own_ids=own_ids))  # fmt: skip
    return [f"  Filled for anchor {a.label} on {a.table}.{a.column}:", f"  WHERE {sql}"]


def _anchor_lines(
    column: str, anchors: Sequence[HierarchyAnchor], dialect: str, names: Mapping[str, str],
) -> list[str]:
    """The first ``ANCHORS_PER_COLUMN`` anchors, then a count of the rest."""
    lines = []
    for a in anchors[:ANCHORS_PER_COLUMN]:
        vocabulary = f", vocabulary {_literal(a.vocabulary, dialect)}" if a.vocabulary else ""
        key = f"name {_literal(a.label, dialect)}" if a.code in names else (
            f"id {_literal(a.code, dialect)}")  # fmt: skip
        lines.append(f"  anchor {key}{vocabulary}; {a.member_count} members")
    if (omitted := len(anchors) - ANCHORS_PER_COLUMN) > 0:
        lines.append(f"  {omitted} more retrieved anchors on {column} are not shown.")
    return lines


def _anchor_ids(
    spec: ConceptHierarchy, dialect: str, name: str | None, vocabulary: str | None,
    concept_class: str | None,
) -> str:
    """The anchor concept's ids by name plus every qualifier column the spec declares."""
    k = spec.concepts
    if name is None:
        named = [(k.name_column, ANCHOR_NAME), (k.vocabulary_column, ANCHOR_VOCABULARY)]
    else:
        named = [(k.name_column, _literal(name, dialect))] + [
            (col, _literal(value, dialect)) for col, value
            in ((k.vocabulary_column, vocabulary), (k.class_column, concept_class))
            if value is not None
        ]  # fmt: skip
    where = [f"n.{col} = {value}" for col, value in named if col]
    if k.standard_column:
        where.append(f"n.{k.standard_column} IN ({_quoted(k.standard_values, dialect)})")
    return f"SELECT n.{k.id_column} FROM {k.relation} n WHERE {' AND '.join(where)}"


def _reached(binding: SuppliedBinding, facts: LibraryFacts) -> bool:
    element = facts.elements.get(binding.de_id or "")
    hierarchy = facts.hierarchy
    return bool(element and hierarchy and any(hierarchy.reaches(c) for c, _ in element.members))


def _route(spec: ConceptHierarchy, dialect: str, own_ids: bool) -> str:
    k, m = spec.concepts, spec.mapping
    route = ""
    if m is not None:
        route = (f"A member reaches a concept through {m.relation} where {m.type_column} in "
                 f"({_quoted(m.mapping_types, dialect)})")  # fmt: skip
        if m.replacement_types:
            replaced = _quoted(m.replacement_types, dialect)
            route += f", then its replacement where {m.type_column} in ({replaced})"
        route += ", or directly when its own id is a concept." if own_ids else "."
    if k.standard_column:
        route += (f" Only concepts with {k.relation}.{k.standard_column} in "
                  f"({_quoted(k.standard_values, dialect)}) count.")  # fmt: skip
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


def _quoted(values: Iterable[str], dialect: str) -> str:
    return ", ".join(_literal(v, dialect) for v in values)


def _literal(value: str, dialect: str) -> str:
    return exp.Literal.string(value).sql(dialect=dialect)

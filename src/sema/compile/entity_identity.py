"""Compile an :class:`EntityIdentity` into one CTE with one row per entity.

Each attribute takes its value from the first row of a ``ROW_NUMBER`` ordering:
preferred rows that carry a value first, in their ranked order, then the rule's
ordering column with nulls last, then the lowest attribute value when the rule
orders by another column. Null ordering is a ``CASE`` term, so the SQL means the
same in every dialect. Every column is quoted, the CTE reads only the identity's
columns, and its internal column names never equal one of them.
"""

from __future__ import annotations

from dataclasses import dataclass

import sqlglot.expressions as exp

from sema.models.target.identity import (
    EntityIdentity,
    PreferredValue,
    ResolutionRule,
    ResolvedAttribute,
)

__all__ = [
    "CompositeKeyCountFinding",
    "composite_key_count_findings",
    "entity_key_name",
    "identity_cte",
    "identity_cte_sql",
    "identity_select",
    "key_clash",
]

NULL_KEY_ROW = "_null_key_row"
_DESCENDING = frozenset({ResolutionRule.MAX, ResolutionRule.MAX_BY})


def _col(name: str) -> exp.Column:
    return exp.column(name, quoted=True)


def _as(expression: exp.Expr, name: str) -> exp.Expr:
    return exp.alias_(expression, name, quoted=True)


def _flag(condition: exp.Expr, when_true: int) -> exp.Case:
    number = exp.Literal.number
    return exp.Case().when(condition, number(when_true)).else_(number(1 - when_true))


def _nulls_last(column: str, *, descending: bool) -> list[exp.Ordered]:
    return [
        exp.Ordered(this=_flag(_col(column).is_(exp.null()), 1), desc=False),
        exp.Ordered(this=_col(column), desc=descending),
    ]


def _lowest_value_breaks_ties(column: str) -> list[exp.Ordered]:
    return _nulls_last(column, descending=False)


def _preferred_first(prefer: PreferredValue, column: str) -> list[exp.Ordered]:
    unpreferred = exp.Literal.number(len(prefer.ranked))
    rank = exp.Case().when(_col(column).is_(exp.null()), unpreferred)
    for position, value in enumerate(prefer.ranked):
        rank = rank.when(_col(prefer.column).eq(exp.convert(value)), exp.Literal.number(position))
    return [exp.Ordered(this=rank.else_(unpreferred), desc=False)]


def _ordering(attribute: ResolvedAttribute) -> list[exp.Ordered]:
    descending = attribute.rule in _DESCENDING
    terms = _preferred_first(attribute.prefer, attribute.column) if attribute.prefer else []
    terms += _nulls_last(attribute.by or attribute.column, descending=descending)
    if attribute.by:
        terms += _lowest_value_breaks_ties(attribute.column)
    return terms


def _internal(name: str, identity: EntityIdentity) -> str:
    """``name``, prefixed with underscores until no column the identity reads has it."""
    while name in identity.columns:
        name = f"_{name}"
    return name


def _rank_name(attribute: ResolvedAttribute, identity: EntityIdentity) -> str:
    return _internal(f"_rank_{attribute.output_name}", identity)


def _rank(
    attribute: ResolvedAttribute, identity: EntityIdentity, partition: list[exp.Column]
) -> exp.Expr:
    window = exp.Window(
        this=exp.RowNumber(),
        partition_by=partition,
        order=exp.Order(expressions=_ordering(attribute)),
    )
    return _as(window, _rank_name(attribute, identity))


def _first_ranked(attribute: ResolvedAttribute, identity: EntityIdentity) -> exp.Expr:
    rank = _col(_rank_name(attribute, identity))
    first = exp.Case().when(rank.eq(exp.Literal.number(1)), _col(attribute.column))
    return exp.Max(this=first)


def _resolved_values(identity: EntityIdentity) -> list[exp.Expr]:
    """Each attribute's first ranked value; a label is NULL wherever its code is NULL."""
    resolved = {a.output_name: _first_ranked(a, identity) for a in identity.attributes}
    codes: dict[str, exp.Expr] = {k: _col(k) for k in identity.key_columns} | resolved
    values = []
    for attribute in identity.attributes:
        value = resolved[attribute.output_name]
        if attribute.label_of is not None:
            code_is_null = codes[attribute.label_of].copy().is_(exp.null())
            value = exp.Case().when(code_is_null, exp.null()).else_(value.copy())
        values.append(_as(value, attribute.output_name))
    return values


def _key_part(column: str) -> exp.Expr:
    """NULL as ``n``, a value as ``v<length>:<value>``: no two key tuples share a text."""
    text = exp.cast(_col(column), exp.DataType.Type.VARCHAR)
    length = exp.cast(exp.Length(this=text.copy()), exp.DataType.Type.VARCHAR)
    value = exp.Concat(expressions=[exp.Literal.string("v"), length, exp.Literal.string(":"),
                                    text])  # fmt: skip
    return exp.Case().when(_col(column).is_(exp.null()), exp.Literal.string("n")).else_(value)


def _entity_key(parts: list[str], name: str) -> exp.Expr:
    encoded = exp.Concat(expressions=[_key_part(p) for p in parts])
    return _as(exp.MD5(this=encoded), name)


def _null_key_rows_stand_alone(identity: EntityIdentity) -> bool:
    return bool(identity.nullable_keys) and not identity.null_keys_group


def _keyed_rows(relation: str, identity: EntityIdentity) -> exp.Expr:
    table = exp.to_table(relation)
    alone = _null_key_rows_stand_alone(identity)
    if not (identity.non_null_keys or alone or identity.attributes):
        return table
    columns: list[exp.Expr] = [_col(c) for c in identity.columns]
    if alone:
        any_null = exp.or_(*(_col(k).is_(exp.null()) for k in identity.nullable_keys))
        each_row = [exp.Ordered(this=_col(c)) for c in identity.columns]
        numbered = exp.Window(this=exp.RowNumber(), order=exp.Order(expressions=each_row))
        own_row = exp.Case().when(any_null, numbered)
        columns.append(_as(own_row, _internal(NULL_KEY_ROW, identity)))
    rows = exp.select(*columns).from_(table)
    if identity.non_null_keys:
        rows = rows.where(exp.and_(*(_col(k).is_(exp.null()).not_()
                                     for k in identity.non_null_keys)))  # fmt: skip
    return rows.subquery("keyed")


def identity_select(
    relation: str, identity: EntityIdentity, key: str | None = None
) -> exp.Select:
    """One row per entity of ``relation``: its keys, the ``key`` column, and attributes.

    The ``key`` column hashes the key parts with NULL as a value (and the own-row marker
    when null keys do not group), so ``COUNT(DISTINCT key)`` counts every entity.
    """
    if key and (clash := key_clash(key, identity)):
        raise ValueError(clash)
    keys = [_col(k) for k in identity.key_columns]
    alone = _null_key_rows_stand_alone(identity)
    own_row = [_internal(NULL_KEY_ROW, identity)] if alone else []
    groups = [*keys, *(_col(c) for c in own_row)]
    surrogate = [_entity_key([*identity.key_columns, *own_row], key)] if key else []
    rows = _keyed_rows(relation, identity)
    if not identity.attributes:
        return exp.select(*keys, *surrogate).from_(rows).group_by(*groups)
    ranks = [_rank(a, identity, groups) for a in identity.attributes]
    ranked = exp.select(exp.Star(), *ranks).from_(rows).subquery("ranked")
    values = _resolved_values(identity)
    return exp.select(*keys, *surrogate, *values).from_(ranked).group_by(*groups)


def entity_key_name(entity: str) -> str:
    return f"{entity}_key"


def key_clash(key: str, identity: EntityIdentity) -> str | None:
    """Why ``key`` cannot be projected: it equals a key column or an attribute name."""
    projected = [*identity.key_columns, *(a.output_name for a in identity.attributes)]
    # warehouses fold identifier case, so E_KEY and e_key are one column
    clash = next((c for c in projected if c.casefold() == key.casefold()), None)
    if clash is None:
        return None
    return f"entity key {key} equals the projected column {clash}; rename that column"


def identity_cte(
    name: str, relation: str, identity: EntityIdentity, dialect: str | None = None
) -> str:
    """``<name> AS (...)``: one entry of a ``WITH`` list, with the ``<name>_key`` column."""
    body = identity_select(relation, identity, entity_key_name(name)).sql(dialect=dialect)
    return f"{exp.to_identifier(name).sql(dialect=dialect)} AS ({body})"


def identity_cte_sql(
    name: str, relation: str, identity: EntityIdentity, dialect: str | None = None
) -> str:
    """``WITH <name> AS (...)``, ready for a ``SELECT ... FROM <name>`` to follow."""
    return f"WITH {identity_cte(name, relation, identity, dialect)}"


@dataclass(frozen=True)
class CompositeKeyCountFinding:
    """A multi-column ``COUNT(DISTINCT ...)`` that drops tuples with a NULL key part."""

    key_columns: tuple[str, ...]
    columns: tuple[str, ...]
    nullable_columns: tuple[str, ...]
    message: str


def _counted_columns(count: exp.Count) -> tuple[str, ...] | None:
    """Column names of a ``COUNT(DISTINCT ...)`` over two or more arguments, else None."""
    distinct = count.this
    if not isinstance(distinct, exp.Distinct) or len(distinct.expressions) < 2:
        return None
    return tuple(e.name for e in distinct.expressions if isinstance(e, exp.Column))


def composite_key_count_findings(
    statement: exp.Expr, identity: EntityIdentity
) -> list[CompositeKeyCountFinding]:
    """Flag each multi-column ``COUNT(DISTINCT ...)`` that counts a nullable key part.

    SQL skips every tuple holding a NULL, so such a count misses the entities whose
    nullable key part is NULL. Any counted nullable key part fires, whether or not the
    other counted columns cover the rest of the key.
    """
    findings = []
    for count in statement.find_all(exp.Count):
        columns = _counted_columns(count)
        if not columns:
            continue
        counted = {c.casefold() for c in columns}  # warehouses fold identifier case
        nullable = tuple(k for k in identity.nullable_keys if k.casefold() in counted)
        if not nullable:
            continue
        message = (
            f"COUNT(DISTINCT {', '.join(columns)}) skips every row where "
            f"{', '.join(nullable)} is NULL; count entities from the one-row-per-entity "
            f"identity over key ({', '.join(identity.key_columns)}) instead"
        )
        findings.append(
            CompositeKeyCountFinding(identity.key_columns, columns, nullable, message)
        )
    return findings

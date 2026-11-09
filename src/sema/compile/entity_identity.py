"""Compile an :class:`EntityIdentity` into one CTE with one row per entity.

Each attribute takes its value from the first row of a ``ROW_NUMBER`` ordering:
preferred rows that carry a value first, in their ranked order, then the rule's
ordering column with nulls last, then the lowest attribute value when the rule
orders by another column. Null ordering is a
``CASE`` term, so the SQL means the same in every dialect.
"""

from __future__ import annotations

import sqlglot.expressions as exp

from sema.models.target.identity import (
    EntityIdentity,
    PreferredValue,
    ResolutionRule,
    ResolvedAttribute,
)

__all__ = ["identity_cte_sql", "identity_select"]

NULL_KEY_ROW = "_null_key_row"
_DESCENDING = frozenset({ResolutionRule.MAX, ResolutionRule.MAX_BY})


def _col(name: str) -> exp.Column:
    return exp.column(name, quoted=True)


def _as(expression: exp.Expr, name: str) -> exp.Alias:
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


def _rank_name(attribute: ResolvedAttribute) -> str:
    return f"_rank_{attribute.output_name}"


def _rank(attribute: ResolvedAttribute, partition: list[exp.Column]) -> exp.Expr:
    window = exp.Window(
        this=exp.RowNumber(),
        partition_by=partition,
        order=exp.Order(expressions=_ordering(attribute)),
    )
    return _as(window, _rank_name(attribute))


def _first_ranked(attribute: ResolvedAttribute) -> exp.Expr:
    first = exp.Case().when(
        _col(_rank_name(attribute)).eq(exp.Literal.number(1)), _col(attribute.column)
    )
    return _as(exp.Max(this=first), attribute.output_name)


def _null_key_rows_stand_alone(identity: EntityIdentity) -> bool:
    return bool(identity.nullable_keys) and not identity.null_keys_group


def _keyed_rows(relation: str, identity: EntityIdentity) -> exp.Expr:
    table = exp.to_table(relation)
    alone = _null_key_rows_stand_alone(identity)
    if not identity.non_null_keys and not alone:
        return table
    columns: list[exp.Expr] = [exp.Star()]
    if alone:
        any_null = exp.or_(*(_col(k).is_(exp.null()) for k in identity.nullable_keys))
        each_row = [exp.Ordered(this=_col(c)) for c in identity.columns]
        numbered = exp.Window(this=exp.RowNumber(), order=exp.Order(expressions=each_row))
        own_row = exp.Case().when(any_null, numbered)
        columns.append(_as(own_row, NULL_KEY_ROW))
    rows = exp.select(*columns).from_(table)
    if identity.non_null_keys:
        rows = rows.where(exp.and_(*(_col(k).is_(exp.null()).not_()
                                     for k in identity.non_null_keys)))  # fmt: skip
    return rows.subquery("keyed")


def identity_select(relation: str, identity: EntityIdentity) -> exp.Select:
    """One row per entity of ``relation``: its key columns and resolved attributes."""
    keys = [_col(k) for k in identity.key_columns]
    alone = _null_key_rows_stand_alone(identity)
    groups = [*keys, _col(NULL_KEY_ROW)] if alone else keys
    rows = _keyed_rows(relation, identity)
    if not identity.attributes:
        return exp.select(*keys).from_(rows).group_by(*groups)
    ranks = [_rank(a, groups) for a in identity.attributes]
    ranked = exp.select(exp.Star(), *ranks).from_(rows).subquery("ranked")
    values = [_first_ranked(a) for a in identity.attributes]
    return exp.select(*keys, *values).from_(ranked).group_by(*groups)


def identity_cte_sql(
    name: str, relation: str, identity: EntityIdentity, dialect: str | None = None
) -> str:
    """``WITH <name> AS (...)``, ready for a ``SELECT ... FROM <name>`` to follow."""
    body = identity_select(relation, identity).sql(dialect=dialect)
    return f"WITH {exp.to_identifier(name).sql(dialect=dialect)} AS ({body})"

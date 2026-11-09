"""Compile an :class:`EntityIdentity` into one CTE with one row per entity.

Each attribute takes its value from the first row of a ``ROW_NUMBER`` ordering:
preferred rows first, then the rule's ordering column with nulls last, then the
lowest attribute value when the rule orders by another column. Null ordering is a
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
    return exp.column(name)


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


def _preferred_first(prefer: PreferredValue | None) -> list[exp.Ordered]:
    if prefer is None:
        return []
    matches = _col(prefer.column).eq(exp.convert(prefer.value))
    return [exp.Ordered(this=_flag(matches, 0), desc=False)]


def _ordering(attribute: ResolvedAttribute) -> list[exp.Ordered]:
    descending = attribute.rule in _DESCENDING
    terms = _preferred_first(attribute.prefer)
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
    return exp.alias_(window, _rank_name(attribute))


def _first_ranked(attribute: ResolvedAttribute) -> exp.Expr:
    first = exp.Case().when(
        _col(_rank_name(attribute)).eq(exp.Literal.number(1)), _col(attribute.column)
    )
    return exp.alias_(exp.Max(this=first), attribute.output_name)


def _any_key_null(identity: EntityIdentity) -> exp.Expr:
    return exp.or_(*(_col(k).is_(exp.null()) for k in identity.key_columns))


def _keyed_rows(relation: str, identity: EntityIdentity) -> exp.Expr:
    table = exp.to_table(relation)
    if identity.exclude_null_keys:
        keys_present = exp.and_(*(_col(k).is_(exp.null()).not_() for k in identity.key_columns))
        return exp.select(exp.Star()).from_(table).where(keys_present).subquery("keyed")
    if identity.null_keys_group:
        return table
    own_row = exp.Window(this=exp.RowNumber())
    each_alone = exp.alias_(exp.Case().when(_any_key_null(identity), own_row), NULL_KEY_ROW)
    return exp.select(exp.Star(), each_alone).from_(table).subquery("keyed")


def identity_select(relation: str, identity: EntityIdentity) -> exp.Select:
    """One row per entity of ``relation``: its key columns and resolved attributes."""
    keys = [_col(k) for k in identity.key_columns]
    groups = keys if identity.exclude_null_keys or identity.null_keys_group else [
        *keys, _col(NULL_KEY_ROW)
    ]
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

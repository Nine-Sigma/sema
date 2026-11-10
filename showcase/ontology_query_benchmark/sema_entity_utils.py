"""Arm S entities: a declared entity identity is a relation the model reads by name.

The S executor prepends the entity's CTE to each statement that names it, merged into
the statement's own ``WITH`` list, and rejects a multi-column key count over the
entity's source relation as a repairable error that names ``<entity>_key``. An entity
whose identity reads a denied column is never exposed, because the harness's
denied-token check sees only the model's SQL.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import sqlglot
import sqlglot.expressions as exp
from sqlglot.errors import SqlglotError

from sema.compile.entity_identity import (
    composite_key_count_findings,
    entity_key_name,
    identity_cte,
)
from sema.log import logger
from sema.models.target.identity import EntityIdentity
from showcase.ontology_query_benchmark.sema_bindings import ProductSpec, Sidecar

LEADING_WITH = re.compile(r"^\s*WITH\s+(RECURSIVE\s+)?", re.IGNORECASE)


@dataclass(frozen=True)
class ExposedEntity:
    """``cte`` is ``<name> AS (...)`` in the warehouse dialect."""

    name: str
    relation: str
    identity: EntityIdentity
    cte: str


def exposed_entities(
    products: tuple[ProductSpec, ...], sidecar: Sidecar, is_denied: Callable[[str], bool]
) -> tuple[ExposedEntity, ...]:
    """Every product's entity identity that reads no column its product or the run denies."""
    exposed = []
    for product in products:
        entity = product.doc.identity_entity
        if entity is None or entity.identity is None:
            continue
        name = entity.entity_id or product.pin.product
        spec_denied = set(product.doc.exposure.denied_columns)
        if denied := [c for c in entity.identity.columns if c in spec_denied or is_denied(c)]:
            logger.info("Entity {} withheld from arm S: it reads denied {}", name, denied)
            continue
        relation = sidecar.relation(product.pin.table)
        cte = identity_cte(name, relation, entity.identity, dialect=sidecar.sql_dialect)
        exposed.append(ExposedEntity(name, relation, entity.identity, cte))
    return tuple(exposed)


def with_entities(sql: str, ctes: list[str]) -> str:
    """``sql`` with ``ctes`` leading its ``WITH`` list, or a new one."""
    if not ctes:
        return sql
    listed = ", ".join(ctes)
    leading = LEADING_WITH.match(sql)
    if leading is None:
        return f"WITH {listed}\n{sql}"
    return f"{leading.group(0)}{listed}, {sql[leading.end():]}"


def _parsed(sql: str, dialect: str) -> exp.Expr | None:
    try:
        return sqlglot.parse_one(sql, read=dialect)
    except SqlglotError:
        return None


def _named(statement: exp.Expr | None, entities: tuple[ExposedEntity, ...]) -> list[str]:
    """CTEs of the entities ``statement`` reads by name; all of them when it does not parse."""
    if statement is None:
        return [e.cte for e in entities]
    own = {cte.alias.casefold() for cte in statement.find_all(exp.CTE)}
    read = {t.name.casefold() for t in statement.find_all(exp.Table) if not t.db}
    return [e.cte for e in entities if e.name.casefold() in read - own]


def key_count_rejections(
    statement: exp.Expr, entities: tuple[ExposedEntity, ...], dialect: str
) -> list[str]:
    """Each multi-column key count over an entity's source relation, with a repair hint.

    Only a statement reading the relation is checked, so an unrelated table that shares
    a key column name is never flagged.
    """
    read = {t.name.casefold() for t in statement.find_all(exp.Table)}
    messages = []
    for entity in entities:
        if exp.to_table(entity.relation, dialect=dialect).name.casefold() not in read:
            continue
        key = entity_key_name(entity.name)
        messages += [
            f"{finding.message}: count {key} from the {entity.name} table, as "
            f"COUNT(DISTINCT {key}) FROM {entity.name}, because a multi-column DISTINCT "
            "drops rows whose key part is NULL"
            for finding in composite_key_count_findings(statement, entity.identity)
        ]
    return messages


class EntityExecutor:
    """The warehouse executor, reading exposed entities by name; it never raises."""

    def __init__(self, harness: SimpleNamespace, inner: Any,
                 entities: tuple[ExposedEntity, ...], dialect: str) -> None:  # fmt: skip
        self.harness, self.inner = harness, inner
        self.entities, self.dialect = entities, dialect

    def execute(self, sql: str, row_limit: int) -> Any:
        started = time.perf_counter()
        statement = _parsed(sql, self.dialect)
        if statement is not None:
            if rejections := key_count_rejections(statement, self.entities, self.dialect):
                latency = time.perf_counter() - started
                message = "; ".join(rejections)
                return self.harness.models.Execution(
                    False, (), (), latency, "sql_error", message
                )
        return self.inner.execute(with_entities(sql, _named(statement, self.entities)), row_limit)

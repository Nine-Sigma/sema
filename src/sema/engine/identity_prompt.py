"""The identity proposer (contract F4): one structured call per relation names the entity
its rows describe, the columns that identify it, the column that orders its versions and the
relation's row key. The data then accepts or rejects each name; nothing else of it is used.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from sema.llm_client import IdentityProposal
from sema.log import logger

_STAGE = "identity"

_INSTRUCTIONS = """You read one relation of a warehouse and say what its rows describe.

Answer with four fields:
- entity: the name of the thing the entity_key identifies, singular.
- entity_key: the columns whose values identify one such thing. When several rows describe
  one thing (its versions, events or parts), name the columns that stay the same across
  those rows, never a surrogate column that is unique per row. When each row is its own
  thing, name the row key.
- latest_by: when the rows of one thing are its versions, the one column whose greatest
  value marks the newest version; otherwise null.
- row_key: when no row key is given below, the columns whose values identify one row;
  otherwise null.

Use only column names listed below. A column's distinct count equal to the row count marks a
column unique per row; a smaller one marks a value repeated across rows.

Example. A relation of account revisions has columns revisionid (unique per row), account,
revision and balance, and no two rows share an account and a revision. Answer: entity
Account, entity_key [account], latest_by revision, row_key [revisionid]."""


@dataclass(frozen=True)
class ColumnProfile:
    """What the proposer sees of one column: its counts on the data and its declaration."""

    name: str
    data_type: str
    comment: str | None
    distinct: int | None
    null_rows: int | None


def identity_prompt(
    relation: str, rows: int | None, columns: Sequence[ColumnProfile],
    key: Sequence[str] | None, entity: str | None, grain: str | None,
) -> str:
    """The proposer's prompt for one relation."""
    listed = "\n".join(
        f"- {c.name} ({c.data_type}); distinct {_count(c.distinct)}; "
        f"NULL in {_count(c.null_rows)} rows" + (f"; {c.comment}" if c.comment else "")
        for c in columns
    )
    return (
        f"{_INSTRUCTIONS}\n\nRelation: {relation}, {_count(rows)} rows.\n"
        f"Row key: {', '.join(key) if key else 'none verified'}.\n"
        f"Entity an earlier reading named: {entity or 'none'}; "
        f"what it said one row is: {grain or 'nothing'}.\n"
        f"Columns:\n{listed}"
    )


def _count(value: int | None) -> str:
    return "unknown" if value is None else str(value)


def propose(client: Any, prompt: str, table_ref: str) -> IdentityProposal | None:
    """One call; a model that fails or answers nothing proposes nothing, and the build goes on."""
    try:
        answer = client.invoke(prompt, IdentityProposal, table_ref=table_ref, stage_name=_STAGE)
    except Exception as exc:  # any model or transport failure leaves the relation unproposed
        logger.warning(f"No identity proposal for {table_ref}: {type(exc).__name__}: {exc}")
        return None
    return answer if isinstance(answer, IdentityProposal) else None

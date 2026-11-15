"""Guards and readers shared by the intake output-model factories."""

from __future__ import annotations

import keyword
import re
from collections.abc import Sequence
from typing import Annotated, Any, Literal

from pydantic import BaseModel, WithJsonSchema

ABSTAIN = "abstain"
NONE = "none"
MAX_SCHEMA_KEYS = 64
VERSIONED_NAME = re.compile(r"_v\d+$")

Choice = tuple[str, float, str]


def enum_type(members: Sequence[str]) -> Any:
    """A Literal that always serializes as a string enum, even with one member.

    Pydantic writes a one-member Literal as `const`, which the portability rule
    does not accept, so the schema is pinned explicitly.
    """
    unique = tuple(dict.fromkeys(members))
    if not unique:
        raise ValueError("an enum field needs at least one member")
    schema = {"type": "string", "enum": list(unique)}
    return Annotated[Literal[unique], WithJsonSchema(schema)]


def refuse_sentinel(members: Sequence[str], sentinel: str, what: str) -> None:
    if sentinel in members:
        raise ValueError(f"{what} member {sentinel!r} is a reserved sentinel")


def checked_field(prefix: str, name: str) -> str:
    if not name.isidentifier() or keyword.iskeyword(name):
        raise ValueError(f"{name!r} is not an identifier and cannot name a field")
    return f"{prefix}{name}"


def checked_name(name: str) -> str:
    if not VERSIONED_NAME.search(name):
        raise ValueError(f"output model name {name!r} must end in _v<N>")
    return name


def checked_key_count(name: str, keys: int) -> None:
    if keys > MAX_SCHEMA_KEYS:
        raise ValueError(f"{name} has {keys} keys, over the {MAX_SCHEMA_KEYS}-key cap")


def read_choices(answer: BaseModel, fields: Sequence[str]) -> list[Choice] | None:
    """Each field's choice with its confidence and reason, or None for a failed answer.

    A strict schema cannot tie the list lengths to the field count, so a length
    mismatch is read as a failed choice rather than paired by position.
    """
    confidences: list[float] = getattr(answer, "confidences")
    reasons: list[str] = getattr(answer, "reasons")
    if len(confidences) != len(fields) or len(reasons) != len(fields):
        return None
    choices = [str(getattr(answer, field)) for field in fields]
    return list(zip(choices, confidences, reasons, strict=True))

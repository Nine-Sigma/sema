from __future__ import annotations

import re
from enum import IntEnum
from typing import Final, NamedTuple
from types import MappingProxyType


class SourcePrecedence(IntEnum):
    HUMAN = 100
    ATLAN = 80
    DBT = 60
    UNITY_CATALOG = 40
    LLM_INTERPRETATION = 20
    HEURISTIC = 15
    PATTERN_MATCH = 10


_SOURCE_PRECEDENCE_MAP: Final[dict[str, int]] = {
    member.name.lower(): member.value for member in SourcePrecedence
}


def source_precedence(source: str, default: int = 0) -> int:
    return _SOURCE_PRECEDENCE_MAP.get(source, default)


REF_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^(?P<platform>[a-z]+)://(?P<workspace>[^/]+)/(?P<catalog>[^/]+)"
    r"/(?P<schema>[^/]+)/(?P<table>[^/]+)(?:/(?P<column>.+))?$"
)


class RefParts(NamedTuple):
    platform: str
    workspace: str
    catalog: str
    schema: str
    table: str
    column: str | None


def parse_ref(ref: str) -> RefParts | None:
    m = REF_PATTERN.match(ref)
    if not m:
        return None
    return RefParts(
        platform=m.group("platform"),
        workspace=m.group("workspace"),
        catalog=m.group("catalog"),
        schema=m.group("schema"),
        table=m.group("table"),
        column=m.group("column"),
    )


_UNITY_REF_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^unity://([^.]+)\.([^.]+)\.([^.]+)(?:\.(.+))?$"
)


def translate_ref(old_ref: str, workspace: str) -> str:
    m = _UNITY_REF_PATTERN.match(old_ref)
    if not m:
        return old_ref
    parts = [f"databricks://{workspace}", m.group(1), m.group(2), m.group(3)]
    if m.group(4):
        parts.append(m.group(4))
    return "/".join(parts)


# Deprecated: use parse_ref() with translate_ref() for legacy refs
UNITY_REF_PATTERN: Final[re.Pattern[str]] = _UNITY_REF_PATTERN


def parse_unity_ref(ref: str) -> tuple[str, str, str, str | None]:
    m = _UNITY_REF_PATTERN.match(ref)
    if not m:
        return "", "", ref, None
    return m.group(1), m.group(2), m.group(3), m.group(4)


def parse_unity_ref_strict(ref: str) -> tuple[str, str, str, str | None]:
    m = _UNITY_REF_PATTERN.match(ref)
    if not m:
        raise ValueError(f"Invalid ref: {ref}")
    return m.group(1), m.group(2), m.group(3), m.group(4)


GENERIC_JOIN_COLUMNS: Final[frozenset[str]] = frozenset({
    "id", "name", "type", "status", "value", "code",
    "description", "created_at", "updated_at",
})

MATCH_TYPE_BOOST: Final[MappingProxyType[str, float]] = MappingProxyType({
    "lexical_exact": 0.3,
    "vector": 0.0,
    "graph": -0.1,
})

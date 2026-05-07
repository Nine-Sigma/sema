"""Closed-set enums shared across planner capabilities."""

from __future__ import annotations

from enum import Enum


class ModelRole(str, Enum):
    SOURCE = "SOURCE"
    TARGET = "TARGET"


class TargetArtifactKind(str, Enum):
    TABLE_ROW = "TABLE_ROW"
    GRAPH_NODE = "GRAPH_NODE"
    GRAPH_EDGE = "GRAPH_EDGE"


class PrimaryKeyStrategy(str, Enum):
    DETERMINISTIC_HASH = "DETERMINISTIC_HASH"
    EXTERNAL_SEQUENCE = "EXTERNAL_SEQUENCE"
    NATURAL_KEY = "NATURAL_KEY"
    COMPOUND = "COMPOUND"


class MaterializationMode(str, Enum):
    INSERT_ONLY = "INSERT_ONLY"
    MERGE = "MERGE"
    REPLACE_PARTITION = "REPLACE_PARTITION"

"""Seam records of the toy harness, shaped like the frozen harness the runner drives."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Question:
    id: str
    question_class: str
    question: str
    result_shape: str
    gold_answer: Any
    gold_sql: str = ""
    needs: tuple[str, ...] = ()
    precision: int | None = None
    tolerance: float = 0.0
    category_columns: tuple[str, ...] = ()


@dataclass(frozen=True)
class ArmContext:
    arm: str
    dialect: str
    text: str
    tokens: int
    budget: int
    overflow: bool


@dataclass(frozen=True)
class Generation:
    text: str
    prompt_tokens: int | None
    completion_tokens: int | None
    latency_s: float


@dataclass(frozen=True)
class Execution:
    ok: bool
    columns: tuple[str, ...] = ()
    rows: tuple[tuple[Any, ...], ...] = ()
    latency_s: float = 0.0
    error_class: str | None = None
    message: str | None = None


@dataclass(frozen=True)
class ColumnPolicy:
    """Bare-name, case-insensitive column denial on any table, as the frozen harness has it."""

    denied: tuple[str, ...] = ()
    deny_patterns: tuple[str, ...] = ()
    identifiers: tuple[str, ...] = ()

    def is_denied(self, name: str) -> bool:
        key = name.lower()
        return key in {d.lower() for d in self.denied} or any(
            p.lower() in key for p in self.deny_patterns
        )

    def is_identifier(self, name: str) -> bool:
        return not self.is_denied(name) and name.lower() in {i.lower() for i in self.identifiers}


@dataclass(frozen=True)
class BenchConfig:
    relations: tuple[str, ...]
    row_concepts: tuple[str, ...]
    always_concepts: tuple[str, ...]
    policy: Any
    budget_tokens: int
    values_top_k: int
    near_miss_pct: float
    description_cap_chars: int
    conventions: tuple[str, ...]
    sql_rules: tuple[str, ...]
    sql_checks: tuple[str, ...]

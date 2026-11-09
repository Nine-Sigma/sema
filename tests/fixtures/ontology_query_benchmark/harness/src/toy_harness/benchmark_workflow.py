"""A toy question loop: one generation, at most max_repairs repairs, SELECT-only validation."""

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

FENCE = re.compile(r"```sql\s*(.*?)\s*```", re.DOTALL)


@dataclass(frozen=True)
class ArmRuntime:
    arm: str
    context: Callable[[str], Any]
    executor: Any
    policy: Any
    known_concepts: tuple[str, ...]
    row_limit: int
    max_repairs: int
    near_miss_pct: float
    conventions: tuple[str, ...] = ()
    sql_rules: tuple[str, ...] = ()
    sql_checks: tuple[str, ...] = ()


def sanitize_message(message: str) -> str:
    return message[:200]


def build_messages(context: Any, question: str, conventions: Any, sql_rules: Any) -> list[dict]:
    system = "Write one SQL query.\n" + "\n".join([*conventions, *sql_rules])
    return [{"role": "system", "content": system},
            {"role": "user", "content": f"{context.text}\n\nQuestion: {question}"}]  # fmt: skip


def run_question(question: Any, runtime: ArmRuntime, model: Any, rep: int = 1) -> dict:
    context = runtime.context(question.question)
    record: dict[str, Any] = {"question_id": question.id, "arm": runtime.arm, "rep": rep,
                              "outcome": None, "complete": False, "sql": None, "repairs": 0,
                              "latency_s": 0.0, "context_tokens": context.tokens,
                              "error_class": None, "answer": None, "attempts": []}  # fmt: skip
    if context.overflow:
        return {**record, "outcome": "context_overflow", "error_class": "context_overflow"}
    messages = build_messages(context, question.question, runtime.conventions, runtime.sql_rules)
    for attempt in range(runtime.max_repairs + 1):
        entry: dict[str, Any] = {"attempt": attempt + 1}
        record["attempts"].append(entry)
        try:
            generation = model.complete(messages)
        except Exception:  # noqa: BLE001 - a model failure is an outcome
            return {**record, "outcome": "infrastructure_failure", "error_class": "model_error"}
        found = FENCE.search(generation.text)
        entry["sql"] = record["sql"] = found.group(1) if found else ""
        if record["sql"].upper().startswith("SELECT"):
            execution = runtime.executor.execute(record["sql"], runtime.row_limit)
            return {**record, "outcome": "wrong", "complete": True,
                    "answer": [list(row) for row in execution.rows[:5]]}  # fmt: skip
        entry["error_class"] = "write_statement"
        messages = [*messages, {"role": "assistant", "content": generation.text},
                    {"role": "user", "content": "Only SELECT is allowed."}]  # fmt: skip
        record["repairs"] = min(attempt + 1, runtime.max_repairs)
    return {**record, "outcome": "validation_failure", "error_class": "write_statement"}

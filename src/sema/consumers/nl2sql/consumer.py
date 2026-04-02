"""NL2SQL consumer — generates constrained SQL from an SCO."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sema.consumers.base import (
    ConsumerDeps,
    ConsumerRequest,
    ConsumerResult,
    ContextProfile,
)
from sema.models.context import SemanticContextObject


@dataclass
class SQLPlan:
    """Result of NL2SQL plan operation."""

    sql: str
    valid: bool
    errors: list[str] = field(default_factory=list)
    attempts: int = 1


@dataclass
class SQLResult:
    """Result of NL2SQL execute operation."""

    data: dict[str, Any] = field(default_factory=dict)
    summary: str | None = None


class NL2SQLConsumer:
    """NL2SQL consumer with dual interface."""

    name: str = "nl2sql"
    capabilities: set[str] = {"plan", "explain", "execute"}

    def __init__(
        self,
        dialect: str = "databricks",
        max_prompt_chars: int = 12000,
    ) -> None:
        self.dialect = dialect
        self.max_prompt_chars = max_prompt_chars

    def context_profile(self) -> ContextProfile:
        return ContextProfile()

    def run(
        self,
        request: ConsumerRequest,
        sco: SemanticContextObject,
        deps: ConsumerDeps,
    ) -> ConsumerResult:
        if request.operation not in self.capabilities:
            raise ValueError(
                f"Unsupported operation: {request.operation!r}. "
                f"Supported: {sorted(self.capabilities)}"
            )
        sql_plan = self.plan(request, sco, deps)
        result = ConsumerResult(
            artifact=sql_plan.sql,
            valid=sql_plan.valid,
            errors=sql_plan.errors,
            attempts=sql_plan.attempts,
        )

        if request.operation == "plan":
            return result

        if not sql_plan.valid:
            return result

        if request.operation == "explain":
            explain_text = self.explain(sql_plan, deps)
            result.data = {"explain": explain_text}
            return result

        if request.operation == "execute":
            sql_result = self.execute(sql_plan, request, deps)
            result.data = sql_result.data
            result.summary = sql_result.summary
            return result

        return result

    def plan(
        self,
        request: ConsumerRequest,
        sco: SemanticContextObject,
        deps: ConsumerDeps,
    ) -> SQLPlan:
        from sema.consumers.nl2sql.prompting import build_sql_prompt
        from sema.consumers.nl2sql.validation import (
            validate_sql_against_sco,
        )

        prompt = build_sql_prompt(
            sco, request.question,
            dialect=self.dialect,
            max_chars=self.max_prompt_chars,
        )
        errors: list[str] = []
        sql = ""
        max_retries = 2

        for attempt in range(max_retries + 1):
            current_prompt = prompt
            if attempt > 0:
                feedback = (
                    "\n\nPrevious SQL had errors:\n"
                    + "\n".join(f"- {e}" for e in errors)
                    + "\n\nPlease fix and try again."
                )
                current_prompt = prompt + feedback

            try:
                response = deps.llm.invoke(current_prompt)  # type: ignore[union-attr]
                sql = _extract_sql(response)
                errors = validate_sql_against_sco(
                    sql, sco, dialect=self.dialect,
                )
                if not errors:
                    return SQLPlan(
                        sql=sql, valid=True, attempts=attempt + 1,
                    )
            except Exception as e:
                errors = [f"LLM error: {e}"]

        return SQLPlan(
            sql=sql, valid=False, errors=errors,
            attempts=max_retries + 1,
        )

    def explain(self, plan: SQLPlan, deps: ConsumerDeps) -> str:
        if deps.sql_runtime is None:
            raise ValueError("No SQL runtime configured for explain")
        return deps.sql_runtime.explain(plan.sql)  # type: ignore[no-any-return]

    def execute(
        self,
        plan: SQLPlan,
        request: ConsumerRequest,
        deps: ConsumerDeps,
    ) -> SQLResult:
        if deps.sql_runtime is None:
            raise ValueError("No SQL runtime configured for execute")
        from sema.consumers.nl2sql.synthesize import synthesize_results

        exec_data = deps.sql_runtime.execute(plan.sql)
        summary = synthesize_results(
            request.question, plan.sql, exec_data, deps.llm,
        )
        return SQLResult(data=exec_data, summary=summary)


def _extract_sql(response: Any) -> str:
    """Extract SQL string from LLM response."""
    raw = (
        response.content
        if hasattr(response, "content")
        else str(response)
    ).strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1]).strip()
    return raw

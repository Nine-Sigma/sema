"""Shared setup for test_runner.py and the tests that reuse it."""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

R01_SQL = "```sql\nSELECT count(DISTINCT rid) AS people FROM main.s.person\n```"
GOLD = 42


@dataclass(frozen=True)
class FakeFinding:
    message: str
    relation: str | None = None
    column: str | None = None


@dataclass(frozen=True)
class FakeSContext:
    text: str
    identity: str
    findings: tuple[FakeFinding, ...] = ()


class FakeModel:
    def __init__(self, harness: SimpleNamespace, replies: list[str], prompt: int | None = 900,
                 completion: int | None = 40) -> None:  # fmt: skip
        self.harness, self.replies = harness, replies
        self.prompt, self.completion = prompt, completion
        self.calls: list[list[dict[str, Any]]] = []

    def complete(self, messages: list[dict[str, Any]]) -> Any:
        self.calls.append(messages)
        text = self.replies[min(len(self.calls), len(self.replies)) - 1]
        return self.harness.models.Generation(text, self.prompt, self.completion, 0.01)


class FakeExecutor:
    def __init__(self, harness: SimpleNamespace, rows: tuple[tuple[Any, ...], ...]) -> None:
        self.harness, self.rows = harness, rows
        self.sqls: list[str] = []

    def execute(self, sql: str, row_limit: int) -> Any:
        self.sqls.append(sql)
        return self.harness.models.Execution(True, ("people",), self.rows, 0.01)


def s_provider(
    findings: tuple[FakeFinding, ...] = (), text: str = "TABLE main.s.person\n  rid string"
):
    return lambda question, is_denied: FakeSContext(text, "s-identity-1", findings)

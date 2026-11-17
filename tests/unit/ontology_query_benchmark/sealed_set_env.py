"""Shared setup for the sealed-set arm tests (unit and integration).

The neutral set lives in `tests/fixtures/sealed_set/neutral/` in the sealed file contract;
its expected answers were written by the expected-answer writer from `reference_sql_a.jsonl`
and `reference_sql_b.jsonl` on the connector fixture's DuckDB file. `ARM_REPLIES` is the
mocked model: per question, the statement each arm writes and what scoring must give it.
"""

import json
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

from showcase.ontology_query_benchmark import freeze
from showcase.ontology_query_benchmark.profile import load_profile
from tests.support.connector_fixture import write_fixture_database

FIXTURES = Path(__file__).parents[2] / "fixtures"
SEALED_SET_FIXTURE = FIXTURES / "sealed_set"
TOY_PROFILE = FIXTURES / "ontology_query_benchmark" / "profile.yaml"
S_CONTEXT_MARK = "S-STUB-CONTEXT"
# Planted in N10's text, in the mocked statement N10 gets and in the row that statement returns.
TEXT_MARKER, SQL_MARKERS, ROW_MARKER = "ZQXTEXTMARK", ("zqxsqlmark", "WORXQZ"), "ZQXROW"
MARKERS = (TEXT_MARKER, *SQL_MARKERS, ROW_MARKER)


@dataclass(frozen=True)
class Reply:
    sql: str
    strict: bool
    order_free: bool
    joins: int


def _right(sql: str, joins: int = 0) -> Reply:
    return Reply(sql, True, True, joins)


def _wrong(sql: str, joins: int = 0, order_free: bool = False) -> Reply:
    return Reply(sql, False, order_free, joins)


_N06 = ("SELECT c.region_code, {} FROM shop.customer c JOIN shop.account a "
        "ON a.customer_id = c.customer_id GROUP BY c.region_code ORDER BY c.region_code")
_LEGACY = ("SELECT o.order_id, {} AS customer_name FROM shop.order_header o LEFT JOIN "
           "shop.customer c ON c.customer_id = o.legacy_customer_id WHERE o.order_id = {}")
_LINES = ("SELECT count(*) AS order_lines FROM shop.order_line l JOIN shop.order_header o "
          "ON o.order_id = l.order_id")
_REGION_W = " JOIN shop.customer c ON c.customer_id = o.customer_id WHERE c.region_code = 'W'"
_N11 = ("SELECT DISTINCT c.customer_id FROM shop.order_line l JOIN shop.order_header o "
        "ON o.order_id = l.order_id JOIN shop.customer c ON c.customer_id = o.customer_id "
        "WHERE {}c.region_code = 'N' ORDER BY c.customer_id")
_JAN = ("SELECT count(*) AS transfers FROM shop.transfer WHERE transfer_date >= "
        "DATE '2024-{:02d}-01' AND transfer_date < DATE '2024-{:02d}-01'")
_COUNTS = "count(DISTINCT c.customer_id) AS customers, count(a.account_id) AS accounts"
_SWAPPED = "count(a.account_id) AS accounts, count(DISTINCT c.customer_id) AS customers"
_N10 = f"SELECT reverse('{SQL_MARKERS[1]}') AS {SQL_MARKERS[0]}, count(*) AS n FROM shop.supplier"

ARM_REPLIES: dict[str, dict[str, Reply]] = {
    "N01": {arm: _right("SELECT count(*) AS customers FROM shop.customer") for arm in "AS"},
    "N02": {"A": _right("SELECT count(*) FROM shop.customer WHERE region_code = 'N'"),
            "S": _wrong("SELECT count(*) FROM shop.customer WHERE region_code = 'S'")},
    "N03": {arm: _right("SELECT region_code, count(*) AS customers FROM shop.customer "
                        "GROUP BY region_code ORDER BY region_code") for arm in "AS"},
    "N04": {"A": _wrong(_JAN.format(2, 3)), "S": _right(_JAN.format(1, 2))},
    "N05": {arm: _right("SELECT sum(amount) AS total FROM shop.transfer") for arm in "AS"},
    # Swapped numeric columns: wrong under strict, right order-free.
    "N06": {"A": _right(_N06.format(_COUNTS), 1),
            "S": _wrong(_N06.format(_SWAPPED), 1, order_free=True)},
    # '' against an expected NULL: right only where scoring.json sets null_equals_empty.
    "N07": {"A": _right(_LEGACY.format("COALESCE(c.customer_name, '')", 7), 1),
            "S": _right(_LEGACY.format("c.customer_name", 7), 1)},
    "N08": {"A": _wrong(_LEGACY.format("COALESCE(c.customer_name, '')", 8), 1),
            "S": _right(_LEGACY.format("c.customer_name", 8), 1)},
    "N09": {"A": _right(_LINES + _REGION_W, 2), "S": _wrong(_LINES, 1)},
    "N10": {arm: _right(_N10) for arm in "AS"},
    # The expected result is empty; scoring.json excludes it from every rate.
    "N11": {"A": _right(_N11.format("l.product_id = 0 AND "), 2),
            "S": _wrong(_N11.format(""), 2)},
}


class ScriptedModel:
    """The mocked model client: `complete(messages) -> Generation`, one statement per
    question and arm; the arm is S when the prompt holds the stub S context's mark."""

    def __init__(self, harness: SimpleNamespace, texts: dict[str, str],
                 arm: str | None = None) -> None:  # fmt: skip
        self.harness, self.texts, self.arm = harness, texts, arm
        self.calls: list[list[dict[str, Any]]] = []

    def complete(self, messages: list[dict[str, Any]]) -> Any:
        self.calls.append(messages)
        prompt = "\n".join(str(m.get("content") or "") for m in messages)
        qid = max((q for q, text in self.texts.items() if text in prompt),
                  key=lambda q: len(self.texts[q]))  # fmt: skip
        arm = self.arm or ("S" if S_CONTEXT_MARK in prompt else "A")
        reply = f"```sql\n{ARM_REPLIES[qid][arm].sql}\n```"
        return self.harness.models.Generation(reply, 900, 40, 0.01)


@dataclass(frozen=True)
class StubSContext:
    text: str
    identity: str
    findings: tuple[Any, ...] = ()


def stub_s_provider(question: str, is_denied: Any) -> StubSContext:
    """An `SProvider`: (question, is_denied) -> SContextLike."""
    text = f"{S_CONTEXT_MARK}\n  fixcat.shop.customer: customer_id, customer_name, region_code"
    return StubSContext(text, "s-stub-identity")


def freeze_toy_harness(out_root: Path) -> Path:
    toy = load_profile(TOY_PROFILE)
    return freeze.freeze(toy.harness.root, out_root, toy.harness.closure)


def materialize(root: Path, frozen: Path) -> Path:
    """The neutral profile and set under `root`, with its DuckDB file and frozen harness."""
    shutil.copytree(SEALED_SET_FIXTURE, root)
    write_fixture_database(root)
    raw = yaml.safe_load((root / "profile.yaml").read_text())
    raw["harness"]["frozen"] = str(frozen)
    (root / "profile.yaml").write_text(yaml.safe_dump(raw))
    return root / "profile.yaml"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def question_texts(set_dir: Path) -> dict[str, str]:
    return {q["id"]: q["text"] for q in read_jsonl(set_dir / "sealed" / "questions.jsonl")}


def expected_cells(set_dir: Path, arms: tuple[str, ...]) -> dict[tuple[str, str, str], tuple]:
    """(cell, key, arm) -> (strict, order_free, n, strict_rate) from `ARM_REPLIES`, one
    attempt per question and arm; excluded questions count only in their own row."""
    scoring = json.loads((set_dir / "scoring.json").read_text())["questions"]
    excluded = {q for q, flags in scoring.items() if flags.get("excluded_from_rate")}
    tally: dict[tuple[str, str, str], list[int]] = defaultdict(lambda: [0, 0, 0])
    for q in read_jsonl(set_dir / "sealed" / "questions.jsonl"):
        for arm in arms:
            reply = ARM_REPLIES[q["id"]][arm]
            keys = [("excluded", q["id"])] if q["id"] in excluded else [
                ("class", q["question_class"]), ("hop_class", str(q["hop_class"])),
                ("hops", str(q["hops"])), ("joins", str(reply.joins)),
                ("join", "with" if reply.joins else "without")]
            for cell, key in keys:
                counts = tally[(cell, key, arm)]
                counts[0] += reply.strict
                counts[1] += reply.order_free
                counts[2] += 1
    return {k: (s, o, n, f"{s / n:.2f}") for k, (s, o, n) in tally.items()}


def report_cells(report: str) -> dict[tuple[str, str, str], tuple]:
    """The report table's rows: `| cell | key | arm | strict | order_free | n | strict_rate |`."""
    cells: dict[tuple[str, str, str], tuple] = {}
    for line in report.splitlines():
        parts = [p.strip() for p in line.strip().strip("|").split("|")]
        if len(parts) == 7 and parts[3].isdigit():
            cell, key, arm, strict, order_free, n, rate = parts
            cells[(cell, key, arm)] = (int(strict), int(order_free), int(n), rate)
    return cells

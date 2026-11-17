"""Shared setup for the sealed-set arm tests (unit and integration).

The neutral set lives in `tests/fixtures/sealed_set/neutral/` in the sealed file contract;
its expected answers were written by the expected-answer writer from `reference_sql_a.jsonl`
and `reference_sql_b.jsonl` on the connector fixture's DuckDB file. `ARM_REPLIES` is the
mocked model: per question, the statement each arm writes and what scoring must give it.
"""

import io
import json
import shutil
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import yaml

from sema.connectors.duckdb import DuckDBConnector
from sema.log import logger
from showcase.ontology_query_benchmark import freeze, runner, scorer
from showcase.ontology_query_benchmark.profile import load_profile
from tests.support.connector_fixture import write_fixture_database

FIXTURES = Path(__file__).parents[2] / "fixtures"
SEALED_SET_FIXTURE = FIXTURES / "sealed_set"
TOY_PROFILE = FIXTURES / "ontology_query_benchmark" / "profile.yaml"
S_CONTEXT_MARK = "S-STUB-CONTEXT"
CONNECTOR_READ = "execute_query_with_columns"
SINK_CANARY = "sealed-set-trace-sink-canary"
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
_N13 = ("SELECT product_id, {} FROM shop.order_line WHERE product_id < 9 "
        "GROUP BY product_id ORDER BY product_id")
# N12: S is within 1% of the expected count (tolerant, neither strict nor order-free).
# N13: eight result rows, more than a receipt's answer keeps; S swaps the numeric columns.
# N14: author B's rows differ from author A's; A writes A's statement, S writes B's.
NEAR_MISS, WIDE, AUTHOR_SPLIT = "N12", "N13", "N14"

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
    NEAR_MISS: {"A": _right("SELECT count(*) AS accounts FROM shop.account"),
                "S": _wrong("SELECT count(*) + 2 AS accounts FROM shop.account")},
    WIDE: {"A": _right(_N13.format("count(*) AS order_lines, sum(qty) AS total_qty")),
           "S": _wrong(_N13.format("sum(qty) AS total_qty, count(*) AS order_lines"),
                       order_free=True)},
    AUTHOR_SPLIT: {"A": _right("SELECT count(DISTINCT customer_id) AS customers "
                               "FROM shop.order_header"),
                   "S": _wrong("SELECT count(customer_id) AS n FROM shop.order_header")},
    # M5: answers the expected-answer writer normalizes (a non-terminating average, an
    # 11-significant-digit decimal sum, a midnight timestamp); both arms write author A's.
    "N15": {arm: _right("SELECT avg(product_id) AS mean_product_id FROM shop.order_line")
            for arm in "AS"},
    "N16": {arm: _right("SELECT sum(CAST(amount / 7 AS DECIMAL(18,5))) AS total_sevenths "
                        "FROM shop.transfer") for arm in "AS"},
    "N17": {arm: _right("SELECT min(CAST(order_date AS TIMESTAMP)) AS first_order_at "
                        "FROM shop.order_header") for arm in "AS"},
}
WRITER_RULE = ("N15", "N16", "N17")
# M1: planted in a question one arm answers wrong (N02), a flagged one (N07), a refused one
# (WIDE), and in the wrong reply's statement and the row it returns.
LEAK_TEXT, LEAK_SQL, LEAK_ROW = "ZQXLEAKTEXT", "zqxleaksql", "ZQXLEAKROW"
LEAK_MARKED = ("N02", "N07", WIDE)
LEAK_REPLY = Reply(f"SELECT '{LEAK_ROW}' AS {LEAK_SQL}", False, False, 0)


class ScriptedModel:
    """The mocked model client: `complete(messages) -> Generation`, one statement per
    question and arm; the arm is S when the prompt holds the stub S context's mark."""

    def __init__(self, harness: SimpleNamespace, texts: dict[str, str],
                 arm: str | None = None) -> None:  # fmt: skip
        self.harness, self.texts, self.arm = harness, texts, arm
        self.calls: list[list[dict[str, Any]]] = []
        self.replies: list[str] = []

    def complete(self, messages: list[dict[str, Any]]) -> Any:
        self.calls.append(messages)
        prompt = "\n".join(str(m.get("content") or "") for m in messages)
        qid = max((q for q, text in self.texts.items() if text in prompt),
                  key=lambda q: len(self.texts[q]))  # fmt: skip
        arm = self.arm or ("S" if S_CONTEXT_MARK in prompt else "A")
        self.replies.append(ARM_REPLIES[qid][arm].sql)
        reply = f"```sql\n{self.replies[-1]}\n```"
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


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def rename_question_ids(set_dir: Path) -> dict[str, str]:
    """Rewrite every question id in the set's files and `scoring.json`; return old -> new.
    New ids run in reverse, so no old id survives and none keeps its position."""
    sealed = set_dir / "sealed"
    ids = [q["id"] for q in read_jsonl(sealed / "questions.jsonl")]
    renamed = {old: f"D{len(ids) - i:02d}x" for i, old in enumerate(ids)}
    for path in sealed.glob("*.jsonl"):
        write_jsonl(path, [{**r, "id": renamed[r["id"]]} for r in read_jsonl(path)])
    scoring = json.loads((set_dir / "scoring.json").read_text())
    scoring["questions"] = {renamed[q]: flags for q, flags in scoring["questions"].items()}
    (set_dir / "scoring.json").write_text(json.dumps(scoring))
    return renamed


def truncate_expected(set_dir: Path, qid: str, keep: int) -> None:
    """Author A's record of `qid` keeps its first `keep` rows and says it was cut."""
    path = set_dir / "sealed" / "expected_answers.jsonl"
    write_jsonl(path, [{**r, "rows": r["rows"][:keep], "rows_truncated": True}
                       if (r["id"], r["author"]) == (qid, "A") else r
                       for r in read_jsonl(path)])  # fmt: skip


def mark_questions(set_dir: Path, ids: tuple[str, ...], marker: str) -> None:
    """Append `marker` to each named question's text."""
    path = set_dir / "sealed" / "questions.jsonl"
    write_jsonl(path, [{**q, "text": f"{q['text']} {marker}"} if q["id"] in ids else q
                       for q in read_jsonl(path)])  # fmt: skip


@contextmanager
def captured_logs() -> Iterator[list[str]]:
    """Every log message from TRACE up while the block runs."""
    logs: list[str] = []
    sink = logger.add(lambda message: logs.append(str(message)), level="TRACE")
    try:
        yield logs
    finally:
        logger.remove(sink)


def files_holding(dirs: tuple[Path, ...], sealed: Path, needles: tuple[str, ...]) -> list[tuple]:
    """(file, needle) for every file under `dirs`, outside `sealed`, that holds a needle."""
    files = {p for d in dirs for p in d.rglob("*") if p.is_file() and not p.is_relative_to(sealed)}
    texts = {p: p.read_bytes().decode("utf-8", "replace").lower() for p in files}
    return sorted((str(p), n) for p, text in texts.items() for n in needles if n.lower() in text)


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


@dataclass(frozen=True)
class SealedRun:
    root: Path
    set_dir: Path
    report: str
    receipts: list[dict[str, Any]]
    output: str
    logs: str
    model: ScriptedModel
    opened: list[Path]
    calls: list[str]
    scores: list[tuple[str, bool]]
    reads: list[tuple[int, str]]

    def receipt(self, slot_id: str) -> dict[str, Any]:
        return next(r for r in self.receipts if r["slot_id"] == slot_id)


def _spy(mp: pytest.MonkeyPatch, module: Any, name: str, calls: list[str]) -> None:
    original = getattr(module, name)

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        calls.append(name)
        return original(*args, **kwargs)

    mp.setattr(module, name, wrapper)


def _spy_connector(mp: pytest.MonkeyPatch, opened: list[Path]) -> None:
    original = DuckDBConnector.__init__

    def init(self: DuckDBConnector, config: Any, *args: Any, **kwargs: Any) -> None:
        opened.append(Path(config.path).resolve())
        original(self, config, *args, **kwargs)

    mp.setattr(DuckDBConnector, "__init__", init)


def _spy_scores(mp: pytest.MonkeyPatch, scores: list[tuple[str, bool]]) -> None:
    """Each `scorer.score` call: the contract's question id and whether it was order-free."""
    original = scorer.score

    def score(*args: Any, **kwargs: Any) -> Any:
        contract = args[0] if args else kwargs["contract"]
        scores.append((contract.question_id, bool(kwargs.get("order_free"))))
        return original(*args, **kwargs)

    mp.setattr(scorer, "score", score)


def _spy_reads(mp: pytest.MonkeyPatch, model: ScriptedModel, reads: list[tuple[int, str]]) -> None:
    """Each connector read, with the number of model calls made before it."""
    original = getattr(DuckDBConnector, CONNECTOR_READ, None)
    if original is None:
        return

    def read(self: DuckDBConnector, query: str, *args: Any, **kwargs: Any) -> Any:
        reads.append((len(model.calls), str(query)))
        return original(self, query, *args, **kwargs)

    mp.setattr(DuckDBConnector, CONNECTOR_READ, read)


def run_sealed(sealed_set: ModuleType, harness: SimpleNamespace, root: Path, profile_path: Path,
               arms: tuple[str, ...]) -> SealedRun:  # fmt: skip
    """One `run_sealed_set` call with the stub S provider, every surface captured."""
    set_dir = root / "neutral"
    model = ScriptedModel(harness, question_texts(SEALED_SET_FIXTURE / "neutral"))
    output, logs, opened, calls, scores, reads = io.StringIO(), [], [], [], [], []
    sink = logger.add(lambda message: logs.append(str(message)), level="TRACE")
    logger.trace(SINK_CANARY)
    try:
        with pytest.MonkeyPatch.context() as mp, redirect_stdout(output), redirect_stderr(output):
            _spy(mp, runner, "schedule", calls)
            _spy(mp, runner, "run", calls)
            _spy_connector(mp, opened)
            _spy_scores(mp, scores)
            _spy_reads(mp, model, reads)
            sealed_set.run_sealed_set(profile_path, arms, 1, model=model,
                                      s_provider=stub_s_provider, report=root / "report.md")
    finally:
        logger.remove(sink)
    receipts = [r for path in sorted((set_dir / "sealed").rglob("receipts.jsonl"))
                for r in read_jsonl(path)]  # fmt: skip
    return SealedRun(root, set_dir, (root / "report.md").read_text(), receipts,
                     output.getvalue(), "".join(logs), model, opened, calls, scores,
                     reads)  # fmt: skip

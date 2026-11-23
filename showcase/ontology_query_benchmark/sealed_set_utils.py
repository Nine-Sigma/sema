"""A sealed question set as runner inputs, and the counts-only report of its receipts.

Only these functions open `<set>/sealed/`. Questions and contracts go to the runner in
memory; the generated harness config and arm A caches are written under the run's folder
inside `sealed/`. The report prints ids, class names, counts and status words, never a
question, a statement or a row.
"""

import dataclasses
import datetime as dt
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import sqlglot
from sqlglot import exp

from sema.connectors.base import Connector
from showcase.ontology_query_benchmark import scorer
from showcase.ontology_query_benchmark.profile import Harness, Knobs

QUESTIONS, EXPECTED, SCORING = "questions.jsonl", "expected_answers.jsonl", "scoring.json"
EXPECTED_AUTHOR = "A"
RESULT_SHAPE = "table_unordered"
CONFIG, CACHE_DIR = "bench.json", "cache"
DEFAULT_VALUES_TOP_K = 25
CELLS = ("class", "hop_class", "hops", "joins", "join", "excluded")
JOIN_WORD = re.compile(r"\bjoin\b", re.IGNORECASE)


def _number(v: float | Decimal) -> Any:
    if not math.isfinite(v):
        return "nan" if math.isnan(v) else ("inf" if v > 0 else "-inf")
    if v == int(v):
        return int(v)
    return float(f"{float(v):.10g}")


def normalize(v: Any) -> Any:
    """The expected-answer writer's cell rule, so found and expected cells compare alike."""
    if v is None or isinstance(v, bool | str | int):
        return v
    if isinstance(v, float | Decimal):
        return _number(v)
    if isinstance(v, dt.datetime):
        midnight = v.tzinfo is None and v.time() == dt.time(0)
        return v.date().isoformat() if midnight else v.isoformat()
    if isinstance(v, dt.date | dt.time):
        return v.isoformat()
    if isinstance(v, dt.timedelta):
        return _number(v.total_seconds())
    if isinstance(v, bytes | bytearray | memoryview):
        return bytes(v).hex()
    if isinstance(v, dict):
        return {str(k): normalize(x) for k, x in v.items()}
    if isinstance(v, list | tuple):
        return [normalize(x) for x in v]
    return str(v)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@dataclass(frozen=True)
class SealedSet:
    questions: tuple[dict[str, Any], ...]
    expected: Mapping[str, dict[str, Any]]
    flags: Mapping[str, dict[str, Any]]

    @property
    def refused(self) -> tuple[str, ...]:
        """Ids whose expected record is truncated or missing: never scored."""
        return tuple(q["id"] for q in self.questions
                     if self.expected.get(q["id"], {}).get("rows_truncated", True))  # fmt: skip

    @property
    def scored_ids(self) -> list[str]:
        return [q["id"] for q in self.questions if q["id"] not in self.refused]


def load_set(set_dir: Path) -> SealedSet:
    sealed = set_dir / "sealed"
    expected = {r["id"]: r for r in read_jsonl(sealed / EXPECTED)
                if r["author"] == EXPECTED_AUTHOR and r.get("status") == "ok"}  # fmt: skip
    scoring = set_dir / SCORING
    flags = json.loads(scoring.read_text())["questions"] if scoring.is_file() else {}
    return SealedSet(tuple(read_jsonl(sealed / QUESTIONS)), expected, flags)


def harness_questions(harness: SimpleNamespace, sealed_set: SealedSet) -> dict[str, Any]:
    """Harness Question records with no gold; every result is an unordered table."""
    scored = set(sealed_set.scored_ids)
    return {q["id"]: harness.models.Question(
        id=q["id"], question_class=q["question_class"], question=q["text"],
        result_shape=RESULT_SHAPE, gold_answer=None,
    ) for q in sealed_set.questions if q["id"] in scored}  # fmt: skip


def contracts(sealed_set: SealedSet) -> dict[str, scorer.Contract]:
    """Author A's rows as contracts, with each question's scoring.json flag."""
    return {qid: scorer.contract_for({
        "id": qid, "result_shape": RESULT_SHAPE, "gold_answer": sealed_set.expected[qid]["rows"],
        "columns": sealed_set.expected[qid]["columns"],
        "null_equals_empty": sealed_set.flags.get(qid, {}).get("null_equals_empty", False),
    }) for qid in sealed_set.scored_ids}  # fmt: skip


def source_layout(connector: Connector, run_dir: Path, package: str, knobs: Knobs) -> Harness:
    """Write the harness config and arm A caches from the connector; return their layout."""
    top_k = knobs.values_top_k if knobs.values_top_k is not None else DEFAULT_VALUES_TOP_K
    schema: dict[str, Any] = {}
    values: dict[str, Any] = {}
    for catalog in connector.list_catalogs():
        for item in connector.discover_tables(catalog):
            relation = f"{item.catalog}.{item.schema}.{item.table_name}"
            columns = connector.extract_columns_dto(item.catalog, item.schema, item.table_name)
            schema[relation] = {"columns": [{"name": c.name, "type": c.data_type}
                                            for c in columns]}  # fmt: skip
            values[relation] = {"values": _top_values(connector, item, columns, top_k)}
    (run_dir / CACHE_DIR).mkdir(parents=True, exist_ok=True)
    config = {"sources": [{"relation": r} for r in schema], "values_top_k": top_k}
    (run_dir / CONFIG).write_text(json.dumps(config, indent=1))
    (run_dir / CACHE_DIR / "schema.json").write_text(json.dumps(schema, indent=1))
    (run_dir / CACHE_DIR / "values.json").write_text(json.dumps({"relations": values}, indent=1))
    return Harness(root=run_dir, package=package, config=CONFIG, question_dir="",
                   question_files=(), cache_dir=CACHE_DIR, kg_cache="", receipts="",
                   closure=())  # fmt: skip


def _top_values(connector: Connector, item: Any, columns: Sequence[Any],
                top_k: int) -> dict[str, Any]:  # fmt: skip
    found = {}
    for column in columns:
        top = connector.extract_top_values_dto(item.catalog, item.schema, item.table_name,
                                               column.name, top_k)  # fmt: skip
        if top is not None:
            found[column.name] = {"values": [[v["value"], v["frequency"]] for v in top.values],
                                  "distinct": top.approx_distinct}  # fmt: skip
    return found


class DialectFrame:
    """An arm's context builder whose frame names the executor's dialect and no other.

    The harness frames both arms in its own dialect; the connector runs statements as
    written, so the frame's dialect field and its leading note name the connector's
    dialect and identifier quote, and the harness's own dialect note, if any, is dropped.
    """

    def __init__(self, harness: SimpleNamespace, context: Callable[[str], Any],
                 dialect: str | None) -> None:  # fmt: skip
        sql = sqlglot.Dialect.get_or_raise(dialect)
        quote, self.name = sql.IDENTIFIER_START, f"{type(sql).__name__} SQL"
        self.note = (f"{self.name}, run as written. Quote an identifier as {quote}name{quote} "
                     "when it is a reserved word or holds a character other than letters, "
                     "digits or underscores.")  # fmt: skip
        self.harness, self.context = harness, context

    @property
    def last(self) -> Any:
        """The S provider's last context, read by the runner for its identity."""
        return getattr(self.context, "last", None)

    @property
    def last_embedding_usd(self) -> float:
        """The cost of the S provider's embedding calls for its last context."""
        return float(getattr(self.context, "last_embedding_usd", 0.0))

    def __call__(self, question: str) -> Any:
        built = self.context(question)
        own = getattr(self.harness.context, "ARM_A_NOTE", None)
        body = built.text.replace(f"{own}\n\n", "", 1) if own else built.text
        text = f"{self.note}\n\n{body}"
        return dataclasses.replace(built, dialect=self.name, text=text,
                                   tokens=self.harness.context.estimate_tokens(text))


def in_dialect(harness: SimpleNamespace, runtimes: Mapping[str, Any],
               dialect: str | None) -> dict[str, Any]:  # fmt: skip
    return {arm: dataclasses.replace(runtime, context=DialectFrame(harness, runtime.context,
                                                                   dialect))
            for arm, runtime in runtimes.items()}  # fmt: skip


def latest_receipts(path: Path) -> list[dict[str, Any]]:
    latest = {r["slot_id"]: r for r in read_jsonl(path)} if path.is_file() else {}
    return list(latest.values())


def joins_written(sql: str | None) -> int:
    if not sql:
        return 0
    try:
        return len(list(sqlglot.parse_one(sql, read="duckdb").find_all(exp.Join)))
    except sqlglot.errors.ParseError:
        # shortcut: a statement sqlglot cannot parse counts the word JOIN, comments included.
        return len(JOIN_WORD.findall(sql))


def _cells(question: Mapping[str, Any], joins: int, excluded: bool) -> list[tuple[str, str]]:
    if excluded:
        return [("excluded", question["id"])]
    return [("class", question["question_class"]), ("hop_class", str(question["hop_class"])),
            ("hops", str(question["hops"])), ("joins", str(joins)),
            ("join", "with" if joins else "without")]  # fmt: skip


def tally(sealed_set: SealedSet, receipts: Iterable[dict[str, Any]]) -> dict[tuple, list[int]]:
    """(cell, key, arm) -> [strict, order_free, attempts]; excluded ids count only there."""
    questions = {q["id"]: q for q in sealed_set.questions}
    excluded = {q for q, f in sealed_set.flags.items() if f.get("excluded_from_rate")}
    counts: dict[tuple, list[int]] = defaultdict(lambda: [0, 0, 0])
    for r in receipts:
        if r["outcome"] == "unrun":
            continue
        qid = r["question_id"]
        for cell, key in _cells(questions[qid], joins_written(r.get("final_sql")), qid in excluded):
            row = counts[(cell, key, r["arm"])]
            row[0] += bool(r.get("strict"))
            row[1] += bool(r.get("order_free"))
            row[2] += 1
    return counts


def render_report(name: str, sealed_set: SealedSet, receipts: list[dict[str, Any]]) -> str:
    counts = tally(sealed_set, receipts)
    outcomes = Counter(r["outcome"] for r in receipts)
    spent = sum(float(r.get("cost_usd") or 0.0) for r in receipts)
    lines = [f"# Sealed-set query baseline: {name}", "",
             f"Receipts {len(receipts)}; spend {spent:.4f} USD; outcomes "
             + ", ".join(f"{o} {n}" for o, n in sorted(outcomes.items())), "",
             "| cell | key | arm | strict | order_free | n | strict_rate |",
             "|---|---|---|---|---|---|---|"]  # fmt: skip
    for (cell, key, arm), (strict, order_free, n) in sorted(
            counts.items(), key=lambda kv: (CELLS.index(kv[0][0]), kv[0][1], kv[0][2])):
        lines.append(f"| {cell} | {key} | {arm} | {strict} | {order_free} | {n} | {strict / n:.2f} |")
    lines += ["", "## Refused", ""]
    lines += [f"- {qid} refused: expected rows truncated or missing" for qid in sealed_set.refused]
    lines += ["", "## Open decisions", ""]
    for qid, flags in sealed_set.flags.items():
        reason = f" ({flags['reason']})" if flags.get("reason") else ""
        lines += [f"- {qid} {flag}{reason}" for flag, on in flags.items() if on is True]
    return "\n".join(lines) + "\n"

import json
from pathlib import Path
from typing import Any

from toy_harness.benchmark_models import BenchConfig, ColumnPolicy


def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def load_bench_config(path: Path) -> BenchConfig:
    """The frozen harness's `sources[].relation` and optional keys; the toy's own policy."""
    raw = load_json(path)
    policy = ColumnPolicy(**{k: tuple(v) for k, v in raw.get("policy", {}).items()})
    return BenchConfig(
        relations=tuple(source["relation"] for source in raw.get("sources", ())),
        row_concepts=tuple(raw.get("row_concepts", ())),
        always_concepts=tuple(raw.get("always_concepts", ())),
        policy=policy,
        budget_tokens=int(raw.get("budget_tokens", 12000)),
        values_top_k=int(raw.get("values_top_k", 25)),
        near_miss_pct=float(raw.get("near_miss_pct", 1.0)),
        description_cap_chars=int(raw.get("description_cap_chars", 160)),
        conventions=tuple(raw.get("conventions", ())),
        sql_rules=tuple(raw.get("sql_rules", ())),
        sql_checks=tuple(raw.get("sql_checks", ())),
    )


def precision_and_tolerance(precision: Any, tolerance: Any) -> tuple[int | None, float]:
    return precision, float(tolerance or 0.0)

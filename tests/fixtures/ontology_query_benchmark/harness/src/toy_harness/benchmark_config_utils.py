import json
from pathlib import Path
from typing import Any

from toy_harness.benchmark_models import BenchConfig, ColumnPolicy


def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def load_bench_config(path: Path) -> BenchConfig:
    raw = load_json(path)
    raw["policy"] = ColumnPolicy(**{k: tuple(v) for k, v in raw["policy"].items()})
    return BenchConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in raw.items()})


def precision_and_tolerance(precision: Any, tolerance: Any) -> tuple[int | None, float]:
    return precision, float(tolerance or 0.0)

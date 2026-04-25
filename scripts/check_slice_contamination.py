"""CLI to verify a holdout slice has zero overlap with few-shot sources
and any other slices passed as references.

Usage:
    uv run python scripts/check_slice_contamination.py \
        --holdout showcase/cbioportal_to_omop/slices/msk_chord_holdout.yaml \
        --against showcase/cbioportal_to_omop/slices/contamination_map.yaml \
        --against showcase/cbioportal_to_omop/slices/msk_chord_dev.yaml

Exits non-zero if overlap is detected.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import yaml


class ContaminationError(RuntimeError):
    pass


def load_contamination_map(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    contaminated = data.get("contaminated_tables") or []
    return {str(t) for t in contaminated}


def load_slice_tables(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    tables = data.get("tables") or []
    names: set[str] = set()
    for entry in tables:
        if isinstance(entry, dict) and entry.get("table_name"):
            names.add(str(entry["table_name"]))
    return names


def _load_reference(path: Path) -> set[str]:
    contam = load_contamination_map(path)
    if contam:
        return contam
    return load_slice_tables(path)


def check_contamination(holdout: Path, references: Iterable[Path]) -> None:
    holdout_tables = load_slice_tables(holdout)
    reference_tables: set[str] = set()
    for ref in references:
        reference_tables.update(_load_reference(ref))
    overlap = holdout_tables & reference_tables
    if overlap:
        raise ContaminationError(
            f"Holdout slice {holdout} overlaps with reference tables: "
            f"{sorted(overlap)}"
        )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--holdout", type=Path, required=True,
        help="Path to the holdout slice YAML",
    )
    parser.add_argument(
        "--against", type=Path, action="append", default=[],
        help="Path to a contamination_map or dev slice YAML; repeatable",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        check_contamination(args.holdout, args.against)
    except ContaminationError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1
    print(f"OK: {args.holdout} is disjoint from {len(args.against)} references")
    return 0


if __name__ == "__main__":
    sys.exit(main())

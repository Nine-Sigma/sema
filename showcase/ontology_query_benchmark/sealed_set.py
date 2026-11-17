"""Measure a source with loaded data and a sealed question set: arms A and S, one runner.

python -m showcase.ontology_query_benchmark.sealed_set --profile <yaml> --arms A,S
    --repetitions <n> [--dry-run] [--report <md>]

Arm A gets the harness's schema-only context, built from the profile's DuckDB file through
the connector; arm S gets SEMA's core query path on the graph at NEO4J_URI. Both execute
through the DuckDB connector and are scored by the benchmark scorer against author A's
expected rows. The run reuses `runner.schedule` and `runner.run`. Receipts land under the
set's `sealed/` folder; stdout and the report carry ids, counts and status words only.
`--dry-run` prints the slot count and `spend 0.00`, and opens no client.
"""

import argparse
import os
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from showcase.ontology_query_benchmark import runner
from showcase.ontology_query_benchmark import runner_utils as ru
from showcase.ontology_query_benchmark import sealed_set_utils as su
from showcase.ontology_query_benchmark.profile import SealedSetProfile, load_sealed_profile
from showcase.ontology_query_benchmark.routing import require_clean_process
from showcase.ontology_query_benchmark.runner_live_utils import ConnectorExecutor, model_client

SEALED_ARMS = ("A", "S")


def run_settings(profile: SealedSetProfile) -> runner.RunSettings:
    ceilings = profile.settings
    return runner.RunSettings(profile.limits, ru.Prices(**ceilings.prices.model_dump()),
                              ceilings.max_cost_usd, ceilings.max_wall_s)  # fmt: skip


def _unpinned(relation: str) -> int:
    raise ru.SourceVersionError(f"{relation} has no pinned version in a sealed-set profile")


def run_sealed_set(profile_path: Path, arms: Sequence[str], repetitions: int, *, model: Any,
                   s_provider: ru.SProvider | None, report: Path) -> dict[str, Any]:  # fmt: skip
    # Imported here: sema's settings models load python-dotenv, which the routing guard refuses.
    from sema.connectors.duckdb import DuckDBConnector
    from sema.models.config import DuckDBConfig

    profile = load_sealed_profile(profile_path)
    sealed_set = su.load_set(profile.set_dir)
    run_dir = profile.set_dir / "sealed" / "runs" / f"{''.join(arms)}x{repetitions}"
    slots = runner.schedule(sealed_set.scored_ids, arms, repetitions)
    runner._write_slots(run_dir, slots)
    settings = run_settings(profile)
    runner._write_manifest(run_dir, profile.harness.frozen, profile.knobs, settings, {})
    harness = ru.load_harness(profile.harness.frozen, profile.harness)  # type: ignore[arg-type]
    connector = DuckDBConnector(DuckDBConfig(path=str(profile.duckdb)))
    try:
        layout = su.source_layout(connector, run_dir, profile.harness.package, profile.knobs)
        executors = {arm: ConnectorExecutor(harness, connector, profile.limits.statement_timeout_s)
                     for arm in arms}  # fmt: skip
        runtimes = su.in_dialect(harness, ru.arm_runtimes(
            harness, run_dir, layout, executors, s_provider, profile.limits, profile.knobs,
        ), connector.sql_dialect)  # fmt: skip
        summary = runner.run(slots, su.harness_questions(harness, sealed_set), runtimes, model,
                             su.contracts(sealed_set), harness, settings,
                             run_dir / "receipts.jsonl", pins={}, read_version=_unpinned)
    finally:
        connector.close()
    receipts = su.latest_receipts(run_dir / "receipts.jsonl")
    report.write_text(su.render_report(profile.name, sealed_set, receipts))
    _print_counts(summary)
    return summary


def _print_counts(summary: dict[str, Any]) -> None:
    print(f"spend {summary['spent_usd']:.2f}")
    for outcome, count in sorted(Counter(summary["outcomes"]).items()):
        print(f"{outcome} {count}")
    if summary["stopped"]:
        print(f"stopped {summary['stopped']}")


def _parse(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--arms", default=",".join(SEALED_ARMS))
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    args.arms = tuple(a.strip() for a in args.arms.split(",") if a.strip())
    if not args.arms or set(args.arms) - set(SEALED_ARMS):
        parser.error(f"--arms takes a comma list of {', '.join(SEALED_ARMS)}")
    if not args.dry_run and args.report is None:
        parser.error("--report is required unless --dry-run")
    return args


def _dry_run(profile: SealedSetProfile, arms: Sequence[str], repetitions: int) -> int:
    sealed_set = su.load_set(profile.set_dir)
    print(f"slots {len(runner.schedule(sealed_set.scored_ids, arms, repetitions))}")
    print("spend 0.00")
    for qid in sealed_set.refused:
        print(f"{qid} refused")
    return 0


def _embedder() -> Any:
    """The configured embedder, or none when EMBEDDING_PROVIDER is unset or empty."""
    if not os.environ.get("EMBEDDING_PROVIDER", "").strip():
        return None
    from sema.cli_factories import _get_embedder
    from sema.models.config import EmbeddingConfig

    return _get_embedder(EmbeddingConfig())


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    profile = load_sealed_profile(args.profile)
    if args.dry_run:
        return _dry_run(profile, args.arms, args.repetitions)
    require_clean_process(os.environ, sys.modules, None, profile.model.provider)
    harness = ru.load_harness(profile.harness.frozen, profile.harness)  # type: ignore[arg-type]
    model = model_client(harness, profile.model, profile.limits, os.environ)
    if "S" not in args.arms:
        run_sealed_set(args.profile, args.arms, args.repetitions, model=model,
                       s_provider=None, report=args.report)  # fmt: skip
        return 0
    from sema.cli_factories import _get_neo4j_driver
    from sema.models.config import Neo4jConfig
    from showcase.ontology_query_benchmark.core_s_provider import CoreSContextProvider

    driver = _get_neo4j_driver(Neo4jConfig())
    try:
        provider = CoreSContextProvider(driver, embedder=_embedder())
        run_sealed_set(args.profile, args.arms, args.repetitions, model=model,
                       s_provider=provider, report=args.report)  # fmt: skip
    finally:
        driver.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

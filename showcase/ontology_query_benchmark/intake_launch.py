"""Run the sema CLI with model and embedding calls pinned to a profile's serving host.

The process is checked before sema is imported, .env loading is switched off, and the
profile's host, Databricks profile and endpoints replace whatever the shell set.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Collection, Sequence
from pathlib import Path

import click

from showcase.ontology_query_benchmark.profile import load_profile
from showcase.ontology_query_benchmark.routing import (
    RoutingError,
    intake_env,
    require_clean_process,
)


def _sema_cli() -> click.Command:
    from sema.cli import cli

    return cli


def main(
    argv: Sequence[str] | None = None,
    *,
    modules: Collection[str] = sys.modules,
    cli: click.Command | None = None,
) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("sema_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    routing = load_profile(args.profile).routing
    try:
        require_clean_process(os.environ, modules, routing)
    except RoutingError as exc:
        raise SystemExit(f"refused: {exc}") from exc
    os.environ.update(intake_env(os.environ, routing))
    (cli or _sema_cli()).main(args=args.sema_args, prog_name="sema")


if __name__ == "__main__":
    main()

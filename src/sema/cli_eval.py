"""Click CLI for the evaluation harness: run, diff, and report on slices."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import click

from sema.eval.runner import (
    build_diff_report,
    build_run_report,
    load_slice,
)


@click.group()
def eval_group() -> None:
    """Evaluation harness commands (dev slice runner, diff, report)."""


@eval_group.command("run")
@click.option(
    "--slice", "slice_path", required=True, type=click.Path(exists=True),
    help="Path to slice YAML (e.g. eval/dev_slice.yaml).",
)
@click.option(
    "--label", required=True,
    help="Config label used in dump filenames (e.g. 'baseline', 'staged').",
)
@click.option(
    "--output-dir", required=True, type=click.Path(),
    help="Directory where per-table dumps are written.",
)
@click.option(
    "--config", "config_path", default=None, type=click.Path(),
    help="Optional BuildConfig YAML with overrides.",
)
@click.option("--use-staged/--no-use-staged", default=True)
@click.option("--enable-domain-bias/--no-enable-domain-bias", default=True)
@click.option(
    "--enable-type-inventory/--no-enable-type-inventory", default=True,
)
@click.option(
    "--enable-vocab-hints/--no-enable-vocab-hints", default=True,
)
@click.option("--enable-few-shot/--no-enable-few-shot", default=True)
@click.option("--enable-stage-c/--no-enable-stage-c", default=True)
@click.option("--skip-embeddings/--no-skip-embeddings", default=True)
def run_slice_cmd(
    slice_path: str,
    label: str,
    output_dir: str,
    config_path: str | None,
    use_staged: bool,
    enable_domain_bias: bool,
    enable_type_inventory: bool,
    enable_vocab_hints: bool,
    enable_few_shot: bool,
    enable_stage_c: bool,
    skip_embeddings: bool,
) -> None:
    """Run the pipeline on a slice of tables, dumping per-table artifacts."""
    _execute_run(
        slice_path=slice_path, label=label, output_dir=output_dir,
        config_path=config_path,
        flags={
            "use_staged": use_staged,
            "enable_domain_bias": enable_domain_bias,
            "enable_type_inventory": enable_type_inventory,
            "enable_vocab_hints": enable_vocab_hints,
            "enable_few_shot": enable_few_shot,
            "enable_stage_c": enable_stage_c,
            "skip_embeddings": skip_embeddings,
        },
    )


@eval_group.command("diff")
@click.option(
    "--baseline", "baseline_dir", required=True, type=click.Path(exists=True),
)
@click.option(
    "--current", "current_dir", required=True, type=click.Path(exists=True),
)
@click.option(
    "--output", "output_path", default=None, type=click.Path(),
    help="Write the report JSON here. Defaults to stdout.",
)
def diff_cmd(
    baseline_dir: str, current_dir: str, output_path: str | None,
) -> None:
    """Diff two slice runs by table, aggregate semantic churn."""
    report = build_diff_report(Path(baseline_dir), Path(current_dir))
    _emit_json(report, output_path)


@eval_group.command("report")
@click.option(
    "--run", "run_dir", required=True, type=click.Path(exists=True),
)
@click.option(
    "--label", required=True,
    help="Label to include in the report (e.g. 'staged+domain').",
)
@click.option(
    "--baseline", "baseline_dir", default=None, type=click.Path(),
    help="Optional baseline dir — when set, semantic churn is included.",
)
@click.option(
    "--output", "output_path", default=None, type=click.Path(),
)
def report_cmd(
    run_dir: str, label: str, baseline_dir: str | None,
    output_path: str | None,
) -> None:
    """Build a milestone-style report from telemetry + optional baseline."""
    report = build_run_report(
        Path(run_dir), label=label,
        baseline_dir=Path(baseline_dir) if baseline_dir else None,
    )
    _emit_json(report, output_path)


def _execute_run(
    *,
    slice_path: str,
    label: str,
    output_dir: str,
    config_path: str | None,
    flags: dict[str, Any],
) -> None:
    from sema.models.config import BuildConfig
    from sema.pipeline.orchestrate import run_build

    sdef = load_slice(Path(slice_path))

    overrides: dict[str, Any] = {
        "catalog": sdef.catalog,
        "schemas": [sdef.schema],
        "slice_tables": sdef.tables,
        "eval_dump_dir": output_dir,
        "eval_config_label": label,
        **flags,
    }
    if config_path:
        build_config = BuildConfig.from_file(config_path, overrides)
    else:
        build_config = BuildConfig(**overrides)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    report = run_build(build_config)
    click.echo(json.dumps(report, indent=2, default=str))


def _emit_json(payload: dict[str, Any], output_path: str | None) -> None:
    text = json.dumps(payload, indent=2, default=str)
    if output_path:
        Path(output_path).write_text(text)
    else:
        click.echo(text)

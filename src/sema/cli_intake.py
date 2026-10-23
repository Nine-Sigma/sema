"""CLI: `sema target discover`, `sema target approve`, and `sema target runs`.

`discover` starts an ingestion run from a source and runs it to its first interrupt;
`approve` answers the interrupt a run waits at (import, edited spec, decisions, reject, or
value reads); `runs` lists the run registry. Each prints JSON. The ingestion graph is
imported inside the commands, so loading the CLI loads no LangGraph.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast, get_args

import click

from sema.cli_intake_utils import (
    DEFAULT_ROOT,
    INTAKE_ERRORS,
    check_decisions,
    configured_embedder,
    echo_json,
    error_message,
    find_descriptor,
    intake_model_config,
    landed_from_warehouse,
    outcome_summary,
    read_metrics,
    refuse_unless_waiting,
    registry_of,
    reject_conflict,
    run_row,
    warehouse_schema,
    write_spec,
    written_status,
)
from sema.cli_target_utils import warehouse_connector
from sema.intake.model_calls import IntakeClient, build_intake_client
from sema.intake.profile import EvidenceTier
from sema.models.config import IntakeModelConfig

if TYPE_CHECKING:
    from sema.intake.graph import RunOutcome

EXISTING_FILE = click.Path(exists=True, dir_okay=False, path_type=Path)
ROOT_OPTION = click.option(
    "--root",
    type=click.Path(file_okay=False, path_type=Path),
    default=DEFAULT_ROOT,
    show_default=True,
    help="Ledger root holding the run registry, checkpoints, and run ledgers.",
)
OUT_OPTION = click.option(
    "--out",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Spec output folder; defaults to specs beside the ledger root.",
)


def _intake_client(config: IntakeModelConfig) -> IntakeClient:
    return build_intake_client(config)


@click.command(name="discover")
@click.argument("source", required=False, type=click.Path(exists=True, path_type=Path))
@click.option(
    "--warehouse",
    default=None,
    help="Warehouse schema to discover, as <catalog>.<schema>; replaces SOURCE.",
)
@click.option(
    "--descriptor",
    type=EXISTING_FILE,
    default=None,
    help="Run descriptor; defaults to the single *.run.yaml beside the source.",
)
@click.option("--hint", type=EXISTING_FILE, default=None, help="Partial spec to start from.")
@click.option("--schema", default=None, help="Landing schema; defaults to one from the source.")
@click.option(
    "--tier",
    type=click.Choice(get_args(EvidenceTier)),
    default="metadata_only",
    show_default=True,
    help="Evidence the model may see.",
)
@click.option("--enrich", is_flag=True, help="Run enrichment before Review.")
@click.option("--run-id", default=None, help="Run id; a new one is generated when omitted.")
@ROOT_OPTION
@OUT_OPTION
def discover_cmd(
    source: Path | None,
    warehouse: str | None,
    descriptor: Path | None,
    hint: Path | None,
    schema: str | None,
    tier: EvidenceTier,
    enrich: bool,
    run_id: str | None,
    root: Path,
    out: Path | None,
) -> None:
    """Start an ingestion run from SOURCE or a warehouse schema; stop at its first interrupt."""
    from sema.intake import graph

    named = warehouse_schema(source, warehouse, descriptor, schema)
    catalog, schema = named if named is not None else (None, schema)
    descriptor = descriptor or find_descriptor(cast(Path, source))
    try:
        config = intake_model_config()
        outcome = graph.discover(
            source,
            warehouse=warehouse_connector() if named is not None else None,
            catalog=catalog,
            root=root,
            descriptor=descriptor,
            config=config,
            client=_intake_client(config),
            hint=hint,
            enrich=enrich,
            tier=tier,
            run_id=run_id,
            schema=schema,
            embedder=configured_embedder(config),
        )
    except INTAKE_ERRORS as exc:
        raise click.ClickException(error_message(exc)) from exc
    _echo_outcome(root, outcome, out)


@click.command(name="approve")
@click.argument("run_id")
@click.option("--approver", required=True, help="Who approves, decides, or rejects the run.")
@click.option("--spec", type=EXISTING_FILE, default=None, help="Edited spec to re-enter with.")
@click.option(
    "--decisions", type=EXISTING_FILE, default=None, help="Choices settling open entries."
)
@click.option("--values", is_flag=True, help="Approve the value reads a run asked for.")
@click.option("--reject", is_flag=True, help="Reject the run; nothing is imported.")
@ROOT_OPTION
@OUT_OPTION
def approve_cmd(
    run_id: str,
    approver: str,
    spec: Path | None,
    decisions: Path | None,
    values: bool,
    reject: bool,
    root: Path,
    out: Path | None,
) -> None:
    """Answer the interrupt RUN_ID waits at: approve, decide, edit, reject, or read values."""
    conflict = reject_conflict(spec, decisions, values) if reject else None
    if conflict is not None:
        raise click.UsageError(f"--reject cannot be combined with {conflict}")
    try:
        refuse_unless_waiting(registry_of(root).get(run_id), values)
        if decisions is not None:
            check_decisions(root, run_id, decisions)
        outcome = _answer(run_id, approver, spec, decisions, values, reject, root)
    except INTAKE_ERRORS as exc:
        raise click.ClickException(error_message(exc)) from exc
    _echo_outcome(root, outcome, out)


def _answer(
    run_id: str,
    approver: str,
    spec: Path | None,
    decisions: Path | None,
    values: bool,
    reject: bool,
    root: Path,
) -> RunOutcome:
    from sema.intake import graph

    config = intake_model_config()
    common: dict[str, Any] = {
        "root": root,
        "config": config,
        "client": _intake_client(config),
        "embedder": configured_embedder(config),
        "warehouse": warehouse_connector() if landed_from_warehouse(root, run_id) else None,
    }
    if values:
        return graph.approve_values(run_id, **common)
    if reject:
        return graph.reject(run_id, approver=approver, **common)
    return graph.approve(run_id, approver=approver, spec=spec, decisions=decisions, **common)


@click.command(name="runs")
@click.option("--status", default=None, help="Only runs with this status.")
@click.option("--stage", default=None, help="Only runs at this stage.")
@click.option("--source", default=None, help="Only runs of this source.")
@click.option("--metrics", "metrics_run", default=None, help="Print one run's run_metrics.")
@ROOT_OPTION
def runs_cmd(
    status: str | None,
    stage: str | None,
    source: str | None,
    metrics_run: str | None,
    root: Path,
) -> None:
    """List ingestion runs newest first, or print one run's metrics."""
    if metrics_run is not None:
        echo_json(read_metrics(root, metrics_run))
        return
    records = registry_of(root).list(status=status, stage=stage, source=source)
    echo_json([run_row(record) for record in records])


def _echo_outcome(root: Path, outcome: RunOutcome, out: Path | None) -> None:
    run_id, node, payload = outcome.run_id, outcome.node, outcome.payload
    status = written_status(root, run_id, node, payload)
    written = None if status is None else write_spec(root, run_id, status, out)
    echo_json(outcome_summary(root, run_id, node, payload, written))


__all__ = ["approve_cmd", "discover_cmd", "runs_cmd"]

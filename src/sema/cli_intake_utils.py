"""Helpers for `sema target discover|approve|runs`: refusals, descriptors, and printed rows."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import click
import yaml

from sema import cli_factories
from sema.intake.decisions import load_decisions, validate_decisions
from sema.intake.eligibility_utils import Embedder
from sema.intake.graph_nodes import AMBIGUITY_SUFFIX, METRICS_ARTIFACT, WAITING
from sema.intake.graph_stages import (
    LAND_ARTIFACT,
    REGISTRY_FILE,
    ROLES,
    STRUCTURE,
)
from sema.intake.graph_utils import REVIEW, VALUE_ASK, AmbiguityList
from sema.intake.ledger import ARTIFACTS_DIR
from sema.intake.registry import Registry, RunRecord
from sema.llm_client import LLMStageError
from sema.models.config import EmbeddingConfig, IntakeModelConfig

DEFAULT_ROOT = Path(".sema/intake")
WAREHOUSE_FORM = "<catalog>.<schema>"
WAREHOUSE_MEDIUM = "warehouse"
DESCRIPTOR_GLOB = "*.run.yaml"
OPEN_STAGES = (STRUCTURE, ROLES)
EMBEDDING_PROVIDER_ENV = "EMBEDDING_PROVIDER"
INTAKE_ERRORS = (ValueError, KeyError, OSError, RuntimeError, LLMStageError, yaml.YAMLError)


def error_message(exc: BaseException) -> str:
    """A KeyError's message without the quotes its ``str`` adds."""
    if isinstance(exc, KeyError) and exc.args:
        return str(exc.args[0])
    return str(exc)


def registry_of(root: Path) -> Registry:
    return Registry(root / REGISTRY_FILE)


def find_descriptor(source: Path) -> Path:
    """The single run descriptor in the source's directory; none or several is refused."""
    directory = source if source.is_dir() else source.parent
    found = sorted(directory.glob(DESCRIPTOR_GLOB))
    if len(found) == 1:
        return found[0]
    names = ", ".join(path.name for path in found) or "none"
    raise click.ClickException(
        f"{directory} must hold exactly one {DESCRIPTOR_GLOB} run descriptor"
        f" (found {len(found)}: {names}); pass --descriptor"
    )


def warehouse_schema(
    source: Path | None, warehouse: str | None, descriptor: Path | None, schema: str | None
) -> tuple[str, str] | None:
    """The warehouse ``(catalog, schema)`` to discover, or None for a SOURCE run."""
    if (source is None) == (warehouse is None):
        raise click.UsageError("pass exactly one of SOURCE and --warehouse")
    if warehouse is None:
        return None
    catalog, _, name = warehouse.partition(".")
    if not catalog or not name or "." in name:
        raise click.UsageError(f"--warehouse {warehouse!r} must name {WAREHOUSE_FORM}")
    if descriptor is None:
        raise click.UsageError("--warehouse needs --descriptor; a schema has no directory")
    if schema is not None:
        raise click.UsageError("--schema names a landing schema for SOURCE, not --warehouse")
    return catalog, name


def intake_model_config() -> IntakeModelConfig:
    # The required fields come from INTAKE_MODEL_* in the environment, which mypy cannot see.
    return IntakeModelConfig()  # type: ignore[call-arg]


def configured_embedder() -> Embedder | None:
    """SEMA's embedder when one is configured for the environment, else None."""
    if not os.environ.get(EMBEDDING_PROVIDER_ENV):
        return None
    embedder: Embedder = cli_factories._get_embedder(EmbeddingConfig())
    return embedder


def reject_conflict(spec: Path | None, decisions: Path | None, values: bool) -> str | None:
    """The first approve option that cannot be combined with ``--reject``."""
    given = {"--spec": spec is not None, "--decisions": decisions is not None, "--values": values}
    return next((option for option, set_ in given.items() if set_), None)


def refuse_unless_waiting(record: RunRecord, values: bool) -> None:
    """A run is resumed only at the interrupt the requested action answers."""
    at = f"run {record.run_id} is {record.status} at {record.stage}"
    waiting = record.status == WAITING
    at_values = waiting and record.stage == VALUE_ASK
    if values and not at_values:
        raise click.ClickException(f"{at}; --values answers only a run waiting at {VALUE_ASK}")
    if not values and not waiting:
        raise click.ClickException(f"{at}; only a run waiting at {REVIEW} can be decided")
    if not values and at_values:
        raise click.ClickException(
            f"{at}; approve its requested value reads with --values first"
        )


def check_decisions(root: Path, run_id: str, path: Path) -> None:
    """Decisions are checked against the lists Review wrote, so a refusal writes nothing."""
    artifacts = root / run_id / ARTIFACTS_DIR
    presented = [
        entry
        for stage in OPEN_STAGES
        for entry in AmbiguityList.model_validate_json(
            (artifacts / f"{stage}{AMBIGUITY_SUFFIX}.json").read_text()
        ).ambiguities
    ]
    land = json.loads((artifacts / f"{LAND_ARTIFACT}.json").read_text())
    landed = [f"{r['schema']}.{r['name']}" for r in land["relations"]]
    validate_decisions(load_decisions(path), presented, landed=landed)


def outcome_summary(
    root: Path, run_id: str, node: str | None, payload: dict[str, Any] | None
) -> dict[str, Any]:
    """What discover and approve print: where the run stopped and what waits there."""
    record = registry_of(root).get(run_id)
    printed: dict[str, Any] = {"run_id": run_id, "stopped_at": node, "status": record.status}
    if node == REVIEW and payload is not None:
        printed["open_entries"] = sum(1 for entry in payload["ambiguities"] if _is_open(entry))
        printed["run_metrics"] = read_metrics(root, run_id)
    elif node == VALUE_ASK and payload is not None:
        printed["requests"] = payload["requests"]
    return printed


def _is_open(entry: dict[str, Any]) -> bool:
    return (
        entry["stage"] in OPEN_STAGES and entry["choice"] is None and entry["source"] is None
    )


def landed_from_warehouse(root: Path, run_id: str) -> bool:
    """True for a run Land read from a warehouse, whose later calls need its connector."""
    path = root / run_id / ARTIFACTS_DIR / f"{LAND_ARTIFACT}.json"
    return path.is_file() and json.loads(path.read_text()).get("medium") == WAREHOUSE_MEDIUM


def read_metrics(root: Path, run_id: str) -> Any:
    path = root / run_id / ARTIFACTS_DIR / f"{METRICS_ARTIFACT}.json"
    if not path.is_file():
        raise click.ClickException(f"run {run_id} has no {METRICS_ARTIFACT} artifact")
    return json.loads(path.read_text())


def run_row(record: RunRecord) -> dict[str, Any]:
    """Cost is an estimate from the configured price table, so it is labelled one."""
    return {
        "run_id": record.run_id,
        "source": record.source,
        "stage": record.stage,
        "status": record.status,
        "reason": record.reason,
        "approver": record.approver,
        "model_calls": record.model_calls,
        "tokens": record.tokens,
        "cost_estimate": record.cost,
        "started_at": record.started_at.isoformat(),
        "updated_at": record.updated_at.isoformat(),
    }


def echo_json(value: Any) -> None:
    click.echo(json.dumps(value, indent=2, default=str))

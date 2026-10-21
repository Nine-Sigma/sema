"""Record one live no-hint intake run per committed source, then replay it (task 14.3).

Each source runs through ``sema.intake.graph.discover`` with no hint, no decisions, and no
enrichment, and must stop at Review. The run ledger's call records are copied to
``tests/fixtures/intake/recordings/<source>/calls/`` beside ``recording.json``; a fresh
run over that directory as ``recordings_dir`` must reach Review with zero provider calls.

    uv run --env-file .env python -m scripts.record_intake_run --source a10 \\
        --workdir <scratch dir>

with ``INTAKE_MODEL_PROVIDER=openrouter``, ``INTAKE_MODEL_MODEL``, ``INTAKE_MODEL_API_KEY``,
``INTAKE_MODEL_MAX_CALLS``, ``INTAKE_MODEL_MAX_COST``, and ``INTAKE_MODEL_PRICES`` set.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from scripts.record_intake_run_utils import (
    FIXTURES,
    RECORDINGS_DIR,
    RecordedSource,
    RecordingError,
    ReplayMismatchError,
    check_provider,
    prepare,
    replace_calls,
    write_manifest,
)
from sema.intake import graph as intake_graph
from sema.intake.eligibility_utils import RunDescriptor
from sema.intake.graph_stages import REGISTRY_FILE
from sema.intake.graph_utils import REVIEW
from sema.intake.loaders.athena import VOCABULARY_SCHEMA
from sema.intake.model_calls import BUDGET_REASON, IntakeClient, IntakeProviderNotSetError
from sema.intake.registry import Registry
from sema.log import logger
from sema.models.config import IntakeModelConfig
from tests.support.athena_bundle import make_minimal_bundle

CALL_FAILED_KIND = "call_failed"
EVENTS_FILE = "events.jsonl"

RECORDED_SOURCES = (
    RecordedSource(
        name="a1",
        path=FIXTURES / "partner_sl" / "library.duckdb",
        descriptor=FIXTURES / "partner_sl" / "standard_library.run.yaml",
    ),
    RecordedSource(
        name="a2",
        path=FIXTURES / "tabular" / "codes_a2.xlsx",
        descriptor=FIXTURES / "tabular" / "codes_a2.run.yaml",
        schema="land_a2",
    ),
    RecordedSource(
        name="a3",
        path=FIXTURES / "semsql" / "ontology.db",
        descriptor=FIXTURES / "semsql" / "ontology.run.yaml",
        schema="land_a3",
    ),
    RecordedSource(
        name="a7_csv",
        path=FIXTURES / "omop" / "OMOP_CDMv5.4_Field_Level.csv",
        descriptor=RunDescriptor(
            id="omop-cdm", identity_system="omop", release="5.4", entity_case="lower"
        ),
        schema="land_cdm",
    ),
    RecordedSource(
        name="a7_athena",
        path=Path("a7_athena"),
        descriptor=RunDescriptor.model_validate(
            {
                "id": "athena-minimal",
                "identity_system": "ATH",
                "release": "1",
                "scheme": {
                    "qualified_name": "ath.vocabularies",
                    "description": "A minimal vocabulary bundle",
                },
            }
        ),
        schema=VOCABULARY_SCHEMA,
        build=make_minimal_bundle,
    ),
    RecordedSource(
        name="a10",
        path=FIXTURES / "acris",
        descriptor=FIXTURES / "acris" / "acris.run.yaml",
        schema="land_acris",
    ),
    RecordedSource(
        name="heldout",
        path=FIXTURES / "heldout",
        descriptor=RunDescriptor(id="heldout", identity_system="heldout", release="2026-10-20"),
        schema="land_heldout",
    ),
    RecordedSource(
        name="heldout2",
        path=FIXTURES / "heldout2",
        descriptor=RunDescriptor(
            id="heldout2", identity_system="heldout2", release="2026-10-21"
        ),
        schema="land_heldout2",
    ),
)


@dataclass(frozen=True)
class Recording:
    run_id: str
    outcome: intake_graph.RunOutcome
    directory: Path
    manifest: dict[str, Any]


def _discover(
    source: RecordedSource,
    workdir: Path,
    config: IntakeModelConfig,
    client: IntakeClient | None,
) -> intake_graph.RunOutcome:
    path, descriptor = prepare(source, workdir)
    return intake_graph.discover(
        path,
        root=workdir,
        descriptor=descriptor,
        config=config,
        client=client,
        hint=None,
        enrich=False,
        schema=source.schema,
    )


def record_source(
    source: RecordedSource,
    *,
    recordings: Path,
    workdir: Path,
    config: IntakeModelConfig,
    client: IntakeClient | None = None,
) -> Recording:
    """One live run to Review; its call records replace the source's recording."""
    check_provider(config)
    live = config.model_copy(update={"recordings_dir": None})
    outcome = _discover(source, workdir, live, client)
    if outcome.node != REVIEW:
        raise RecordingError(
            f"{source.name}: run {outcome.run_id} stopped at {outcome.node}, not {REVIEW}"
        )
    _warn_failed_calls(source, workdir / outcome.run_id)
    directory = recordings / source.name
    calls = replace_calls(workdir / outcome.run_id, directory)
    row = Registry(workdir / REGISTRY_FILE).get(outcome.run_id)
    manifest = write_manifest(directory, source.name, live, row, calls)
    return Recording(outcome.run_id, outcome, directory, manifest)


def verify_replay(
    source: RecordedSource,
    *,
    recordings: Path,
    workdir: Path,
    config: IntakeModelConfig,
    client: IntakeClient | None = None,
) -> intake_graph.RunOutcome:
    """A fresh run over the recording; a zero call cap makes any miss a refused call."""
    replay = config.model_copy(
        update={"recordings_dir": recordings / source.name, "max_calls": 0}
    )
    outcome = _discover(source, workdir, replay, client)
    row = Registry(workdir / REGISTRY_FILE).get(outcome.run_id)
    if outcome.node != REVIEW or row.model_calls or row.reason == BUDGET_REASON:
        raise ReplayMismatchError(
            f"{source.name}: replay run {outcome.run_id} stopped at {outcome.node}"
            f" with {row.model_calls} provider calls (reason {row.reason})"
        )
    return outcome


def _warn_failed_calls(source: RecordedSource, ledger_dir: Path) -> None:
    events = ledger_dir / EVENTS_FILE
    lines = events.read_text(encoding="utf-8").splitlines() if events.is_file() else []
    failed = sum(1 for line in lines if line.strip() and _kind(line) == CALL_FAILED_KIND)
    if failed:
        logger.warning(f"{source.name}: {failed} model calls failed during the recorded run")


def _kind(line: str) -> str:
    kind: str = json.loads(line).get("kind", "")
    return kind


def _parse(argv: Sequence[str] | None) -> argparse.Namespace:
    names = [source.name for source in RECORDED_SOURCES]
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--source", action="append", choices=names, dest="sources")
    parser.add_argument("--recordings", type=Path, default=RECORDINGS_DIR)
    parser.add_argument("--workdir", type=Path, default=None)
    args = parser.parse_args(argv)
    args.sources = args.sources or names
    return args


def _config() -> IntakeModelConfig:
    # The required fields come from INTAKE_MODEL_* in the environment, which mypy cannot see.
    config = IntakeModelConfig()  # type: ignore[call-arg]
    check_provider(config)
    return config


def _record_and_verify(
    source: RecordedSource, recordings: Path, workdir: Path, config: IntakeModelConfig
) -> None:
    recording = record_source(
        source, recordings=recordings, workdir=workdir / f"{source.name}_record", config=config
    )
    logger.info(f"{source.name}: recorded run {recording.run_id} into {recordings}")
    outcome = verify_replay(
        source, recordings=recordings, workdir=workdir / f"{source.name}_replay", config=config
    )
    logger.info(f"{source.name}: replay run {outcome.run_id} reached {outcome.node}")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse(argv)
    try:
        config = _config()
    except (IntakeProviderNotSetError, ValueError, ValidationError) as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    workdir = args.workdir or Path(tempfile.mkdtemp(prefix="record_intake_"))
    by_name = {source.name: source for source in RECORDED_SOURCES}
    failed: list[str] = []
    for name in args.sources:
        try:
            _record_and_verify(by_name[name], args.recordings, workdir, config)
        except (RecordingError, ReplayMismatchError) as error:
            print(f"failed: {error}", file=sys.stderr)
            failed.append(name)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

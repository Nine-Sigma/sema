"""The recorded-source table type, the provider guard, and the recording directory layout.

A recording is ``<recordings>/<source>/calls/*.json`` (the run ledger's call layout, so the
directory is a ``recordings_dir``) beside ``recording.json``. Held-out sources are passed
by path only: nothing here opens a file under one.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from sema.intake.eligibility_utils import RunDescriptor
from sema.intake.ledger import CALLS_DIR
from sema.intake.model_calls import IntakeProviderNotSetError
from sema.intake.registry import RunRecord
from sema.models.config import IntakeModelConfig

REPO_ROOT = Path(__file__).parents[1]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "intake"
RECORDINGS_DIR = FIXTURES / "recordings"
MANIFEST_FILE = "recording.json"
RECORDING_PROVIDER = "openrouter"
VERSION_PREFIX = "# prompt: "
DESCRIPTOR_SUFFIX = ".run.yaml"


class RecordingError(RuntimeError):
    """A run stopped anywhere but Review, so it is not recorded."""


class ReplayMismatchError(RuntimeError):
    """A recording did not replay to Review with zero provider calls."""


class ProviderRefusedError(ValueError):
    """Recordings are made with one provider only."""


@dataclass(frozen=True)
class RecordedSource:
    """``build`` writes the source under ``workdir / path`` at record time when set."""

    name: str
    path: Path
    descriptor: Path | RunDescriptor
    schema: str | None = None
    build: Callable[[Path], None] | None = None


def check_provider(config: IntakeModelConfig) -> None:
    if config.provider is None:
        raise IntakeProviderNotSetError(
            f"set INTAKE_MODEL_PROVIDER={RECORDING_PROVIDER} to record intake runs"
        )
    if config.provider != RECORDING_PROVIDER:
        raise ProviderRefusedError(
            f"recordings use provider {RECORDING_PROVIDER} only, not {config.provider}"
        )


def prepare(source: RecordedSource, workdir: Path) -> tuple[Path, Path]:
    """The source path and run descriptor path for one run, built under ``workdir``."""
    workdir.mkdir(parents=True, exist_ok=True)
    path = source.path
    if source.build is not None:
        path = workdir / source.path
        source.build(path)
    descriptor = source.descriptor
    if isinstance(descriptor, Path):
        return path, descriptor
    written = workdir / f"{source.name}{DESCRIPTOR_SUFFIX}"
    raw = descriptor.model_dump(mode="json", exclude_none=True)
    written.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path, written


def replace_calls(ledger_dir: Path, directory: Path) -> list[Path]:
    """Copy the run ledger's call records over whatever ``directory`` held before."""
    target = directory / CALLS_DIR
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    copied = []
    for call in sorted((ledger_dir / CALLS_DIR).glob("*.json")):
        copied.append(Path(shutil.copy2(call, target / call.name)))
    return copied


def prompt_versions(calls: list[Path]) -> list[str]:
    firsts = {_first_line(call) for call in calls}
    return sorted(line.removeprefix(VERSION_PREFIX) for line in firsts)


def _first_line(call: Path) -> str:
    prompt: str = json.loads(call.read_text(encoding="utf-8"))["prompt"]
    return prompt.splitlines()[0] if prompt else ""


def write_manifest(
    directory: Path, name: str, config: IntakeModelConfig, row: RunRecord, calls: list[Path]
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "source": name,
        "provider": config.provider,
        "model": config.model,
        "prompt_versions": prompt_versions(calls),
        "calls": row.model_calls,
        "tokens": row.tokens,
        "cost": row.cost,
    }
    text = json.dumps(manifest, indent=2, sort_keys=False) + "\n"
    (directory / MANIFEST_FILE).write_text(text, encoding="utf-8")
    return manifest

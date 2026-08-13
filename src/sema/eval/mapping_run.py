"""US-012 / G-06: the evaluation subject key and the immutable run artifact.

The value-mapping store's ``GRAIN_KEY`` is a 5-tuple, but the report defaulted
every filter to ``None`` and ``score()`` collapsed on the source code alone with
a last-wins dict. With one policy and one vocabulary release in the store that is
latent; the first run under a second release silently blends two resolvers into
one matrix, with the winner decided by DuckDB row order.

:class:`EvaluationSubject` fixes the four grain dimensions that identify *which*
resolver is being graded. The fifth, ``normalized_source_value``, is the unit of
evaluation — enumerated by the gold set, never filtered.

A run is written once and never rewritten. That is the eval-side answer to a
store whose grain excludes ``run_id`` and upserts in place: the store cannot
prove what it held last week, so the report records the decision set it actually
graded, alongside the digests of the gold rows and universe manifest.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from sema.eval.goldset_snapshot_utils import GoldSetHeader
from sema.eval.mapping_goldset_utils import Decision
from sema.eval.mapping_report_utils import MappingReport

__all__ = ["EvalRunExistsError", "EvaluationSubject", "write_eval_run"]


class EvalRunExistsError(FileExistsError):
    """A run artifact already exists under that ``run_id``. Runs are immutable."""


@dataclass(frozen=True)
class EvaluationSubject:
    """Which resolver a report grades — no defaults, so it cannot be implicit."""

    source_vocabulary: str
    target_property_ref: str
    resolver_policy_ref: str
    vocab_release: str

    def as_dict(self) -> dict[str, str]:
        return {
            "source_vocabulary": self.source_vocabulary,
            "target_property_ref": self.target_property_ref,
            "resolver_policy_ref": self.resolver_policy_ref,
            "vocab_release": self.vocab_release,
            "unit_of_evaluation": "normalized_source_value",
        }


def _decision_to_json(decision: Decision) -> dict[str, Any]:
    return {
        "source_code": decision.source_code,
        "concept_id": decision.concept_id,
        "status": decision.status.value,
        "resolution_status": decision.resolution_status.value,
        "no_map_reason": decision.no_map_reason,
    }


def _decisions_digest(payload: list[dict[str, Any]]) -> str:
    blob = json.dumps(sorted(payload, key=lambda d: str(d["source_code"])), sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def write_eval_run(
    root: str | Path,
    *,
    run_id: str,
    report: MappingReport,
    subject: EvaluationSubject,
    decisions: Iterable[Decision],
    header: GoldSetHeader,
    resolver_run_ids: Sequence[str] = (),
) -> Path:
    """Write one immutable, self-describing evaluation run to ``root/<run_id>/``.

    ``resolver_run_ids`` are the store's own ``run_id``s for the graded rows.
    ``GRAIN_KEY`` excludes ``run_id`` and ``upsert_sql`` overwrites in place, so
    the store cannot prove afterwards which execution produced a decision — this
    is the only place that record survives. ``run_id`` above is the EVAL's.
    """
    parent = Path(root)
    directory = parent / run_id
    if directory.exists():
        raise EvalRunExistsError(
            f"eval run {run_id} already exists at {directory}; a run is written "
            "once and never rewritten — use a new --run-id"
        )
    graded = [_decision_to_json(d) for d in decisions]
    manifest = {
        "run_id": run_id,
        "subject": subject.as_dict(),
        "gold_set": {
            "snapshot_version": report.snapshot_version,
            "rows_sha256": header.rows_sha256,
            "universe_sha256": header.universe_sha256,
            "vocab_release": header.vocab_release,
            "target_vocabulary": header.target_vocabulary,
            "target_domain": header.target_domain,
        },
        "resolver_run_ids": list(resolver_run_ids),
        "decision_count": len(graded),
        "decisions_sha256": _decisions_digest(graded),
        "verdict": report.verdict.value,
        "qualifiers": list(report.qualifiers),
    }
    _publish_run(
        parent,
        directory,
        {
            "run.json": json.dumps(manifest, indent=2) + "\n",
            "report.json": json.dumps(report.as_dict(), indent=2) + "\n",
            "decisions.jsonl": "".join(json.dumps(d) + "\n" for d in graded),
        },
    )
    return directory


def _publish_run(parent: Path, directory: Path, files: dict[str, str]) -> None:
    """Write to a temp sibling, then rename into place — all three files or none.

    A run written file-by-file could be interrupted between them, leaving a
    partial artifact that satisfied the "already exists" refusal and so blocked
    the retry that would have completed it.
    """
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{directory.name}.", dir=parent))
    try:
        for name, body in files.items():
            (staging / name).write_text(body, encoding="utf-8")
        os.replace(staging, directory)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

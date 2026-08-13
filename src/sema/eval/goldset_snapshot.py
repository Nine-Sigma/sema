"""US-002 / G-01+G-03: reading and publishing frozen gold-set snapshots.

The artifact is frozen; the data is not. Every published snapshot lives under
``tests/data/gold/snapshots/<version>/`` and is immutable — ``observe``,
``re-tier``, and ``re-scope`` all emit a NEW version rather than re-stamping one,
because ``row_count`` is the row-weighted scoring *weight*: rewriting it in place
silently changes historical metrics with labels and decisions unchanged, leaving
two reports incomparable and nothing in either to say why.

``current.json`` names the active version. It is a pointer, not a snapshot.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sema.eval.goldset_invariants import (
    SnapshotInvariantError,
    assert_row_integrity,
    assert_snapshot_invariants,
    derive_states,
)
from sema.eval.goldset_snapshot_utils import (
    GoldSetHeader,
    UniverseEntry,
    canonical_sort_key,
    canonical_universe_key,
    file_digest,
    row_from_json,
    row_to_json,
    staged_publish,
)
from sema.eval.mapping_goldset_utils import GoldRow, TierState

__all__ = [
    "GOLD_ROOT",
    "GoldSetSnapshot",
    "SnapshotInvariantError",
    "current_snapshot_rows_path",
    "load_current_snapshot",
    "load_snapshot",
    "resolve_current_version",
    "snapshot_dir",
    "write_snapshot",
]

GOLD_ROOT = Path(__file__).resolve().parents[3] / "tests" / "data" / "gold"

_ROWS_FILE = "oncotree_condition_slice0.jsonl"
_META_FILE = "meta.json"
_UNIVERSE_FILE = "universe.jsonl"
_POINTER_FILE = "current.json"


@dataclass(frozen=True)
class GoldSetSnapshot:
    """A loaded, invariant-checked snapshot: declaration + rows + universe."""

    header: GoldSetHeader
    rows: list[GoldRow]
    universe: tuple[UniverseEntry, ...]

    def by_code(self) -> dict[str, GoldRow]:
        return {r.oncotree_code: r for r in self.rows}

    def states_by_code(self) -> dict[str, TierState]:
        return derive_states(
            self.header, self.universe, frozenset(r.oncotree_code for r in self.rows)
        )

    def universe_row_total(self) -> int:
        return sum(e.frozen_row_count for e in self.universe)


def resolve_current_version(root: Path = GOLD_ROOT) -> str:
    pointer = json.loads((root / _POINTER_FILE).read_text(encoding="utf-8"))
    return str(pointer["snapshot_version"])


def snapshot_dir(version: str, root: Path = GOLD_ROOT) -> Path:
    return root / "snapshots" / version


def load_current_snapshot(root: Path = GOLD_ROOT) -> GoldSetSnapshot:
    return load_snapshot(snapshot_dir(resolve_current_version(root), root))


def current_snapshot_rows_path(root: Path = GOLD_ROOT) -> Path:
    return snapshot_dir(resolve_current_version(root), root) / _ROWS_FILE


def load_snapshot(directory: str | Path) -> GoldSetSnapshot:
    """Load a published snapshot, asserting its digests and invariants."""
    path = Path(directory)
    header = GoldSetHeader.from_dict(
        json.loads((path / _META_FILE).read_text(encoding="utf-8"))
    )
    rows = [row_from_json(obj) for obj in _read_jsonl(path / _ROWS_FILE)]
    universe = tuple(
        UniverseEntry(str(o["code"]), int(o["frozen_row_count"]))
        for o in _read_jsonl(path / _UNIVERSE_FILE)
    )
    # Row integrity first: a duplicated or reordered row is reported as that
    # specific corruption rather than as an unexplained hash diff.
    assert_row_integrity(rows)
    _assert_digests(header, path)
    assert_snapshot_invariants(header, rows, universe)
    return GoldSetSnapshot(header=header, rows=rows, universe=universe)


def write_snapshot(
    directory: str | Path,
    header: GoldSetHeader,
    rows: list[GoldRow],
    universe: tuple[UniverseEntry, ...],
) -> GoldSetHeader:
    """Publish a new snapshot, all three files or none. Never overwrites one."""
    path = Path(directory)
    ordered = sorted(rows, key=canonical_sort_key)
    manifest = tuple(sorted(universe, key=canonical_universe_key))
    assert_snapshot_invariants(header, ordered, manifest)

    def _conflict() -> BaseException:
        return SnapshotInvariantError(
            f"{path} is already published; emit a new snapshot_version instead"
        )

    with staged_publish(path, conflict=_conflict) as staging:
        # Data first, then the digests OF WHAT WAS WRITTEN, then the header
        # carrying them: stamping a projection before the write left the header
        # attesting to bytes no one had produced yet.
        _write_jsonl(staging / _ROWS_FILE, [row_to_json(r) for r in ordered])
        _write_jsonl(staging / _UNIVERSE_FILE, [e.as_dict() for e in manifest])
        stamped = header.with_digests(
            rows=file_digest(staging / _ROWS_FILE),
            universe=file_digest(staging / _UNIVERSE_FILE),
        )
        (staging / _META_FILE).write_text(
            json.dumps(stamped.as_dict(), indent=2) + "\n", encoding="utf-8"
        )
    return stamped


def _assert_digests(header: GoldSetHeader, path: Path) -> None:
    """Verify the header against the files' bytes. An empty digest is a failure.

    Treating a blank digest as "nothing to check" made an unstamped header
    indistinguishable from a verified one — the single edit that disabled the
    whole guard.
    """
    for field_name, filename, declared in (
        ("rows_sha256", _ROWS_FILE, header.rows_sha256),
        ("universe_sha256", _UNIVERSE_FILE, header.universe_sha256),
    ):
        if not declared:
            raise SnapshotInvariantError(
                f"{field_name} is empty; the snapshot cannot verify its own artifact"
            )
        if declared != file_digest(path / filename):
            raise SnapshotInvariantError(
                f"{field_name} does not match the bytes of {filename}"
            )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(json.loads(line))
    return out


def _write_jsonl(path: Path, payload: list[dict[str, Any]]) -> None:
    body = "".join(json.dumps(obj) + "\n" for obj in payload)
    path.write_text(body, encoding="utf-8")

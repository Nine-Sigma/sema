"""Registry helpers: PEP 440 specifier handling and overlap detection."""

from __future__ import annotations

import re
from collections.abc import Iterable

from packaging.specifiers import SpecifierSet
from packaging.version import Version

from sema.targets.exceptions import (
    AdapterRegistryError,
    OverlappingVersionRangeError,
)


_VERSION_LITERAL_RE = re.compile(r"(\d+(?:\.\d+)*(?:[a-zA-Z]\w*)?(?:\+[\w.]+)?)")


def parse_supported_versions(supported_versions: str) -> SpecifierSet:
    if supported_versions == "*":
        raise AdapterRegistryError(
            "supported_versions=='*' is not a PEP 440 specifier; "
            "use the empty string '' as the canonical wildcard form"
        )
    try:
        return SpecifierSet(supported_versions)
    except Exception as exc:
        raise AdapterRegistryError(
            f"supported_versions={supported_versions!r} is not a valid PEP 440 specifier"
        ) from exc


def _extract_boundary_versions(spec_set: SpecifierSet) -> set[str]:
    boundaries: set[str] = set()
    for spec in spec_set:
        match = _VERSION_LITERAL_RE.search(str(spec))
        if match is not None:
            boundaries.add(match.group(1))
    return boundaries


def ranges_overlap(s1: SpecifierSet, s2: SpecifierSet) -> bool:
    """Detect whether two PEP 440 specifier sets share any version.

    Strategy: take the union of literal anchor versions appearing in
    either set, then nudge each anchor up (`<v>.post0`) and down
    (`<v>.dev0`) to detect overlaps that miss the anchors themselves
    (e.g. `>1,<3` vs `>2,<4` overlap on `(2, 3)` even though neither
    set contains the literal anchors `2`, `3`).
    """
    if len(list(s1)) == 0 or len(list(s2)) == 0:
        return True
    anchors = _extract_boundary_versions(s1) | _extract_boundary_versions(s2)
    for v in _candidate_versions(anchors):
        if s1.contains(v, prereleases=True) and s2.contains(v, prereleases=True):
            return True
    return False


def _candidate_versions(anchors: set[str]) -> list[Version]:
    out: list[Version] = []
    for raw in anchors:
        for nudged in _nudges_around(raw):
            try:
                out.append(Version(nudged))
            except Exception:
                continue
    out.extend(_between_anchor_candidates(anchors))
    return out


def _nudges_around(raw: str) -> tuple[str, ...]:
    return (
        raw,
        f"{raw}.0.0.0.1",
        f"{raw}.post0",
        f"{raw}.dev0",
    )


def _between_anchor_candidates(anchors: set[str]) -> list[Version]:
    """For every adjacent pair of anchor versions, emit an integer-major
    candidate strictly between them when one exists. Catches strict-
    inequality overlaps like `>1,<3` vs `>2,<4` (overlap on (2, 3))."""
    parsed: list[Version] = []
    for raw in anchors:
        try:
            parsed.append(Version(raw))
        except Exception:
            continue
    parsed.sort()
    out: list[Version] = []
    for low, high in zip(parsed, parsed[1:]):
        for between in _integer_candidates_between(low, high):
            out.append(between)
    return out


def _integer_candidates_between(low: Version, high: Version) -> list[Version]:
    low_major = low.release[0] if low.release else 0
    high_major = high.release[0] if high.release else 0
    out: list[Version] = []
    for n in range(low_major + 1, high_major + 1):
        try:
            out.append(Version(f"{n}.0.0.0.1"))
        except Exception:
            continue
    return out


def check_no_overlap(
    new_str: str,
    new_set: SpecifierSet,
    existing: Iterable[tuple[str, SpecifierSet]],
    adapter_id: str,
    target_model_id: str,
) -> None:
    for existing_str, existing_set in existing:
        if ranges_overlap(new_set, existing_set):
            raise OverlappingVersionRangeError(
                f"supported_versions={new_str!r} overlaps existing registration "
                f"{existing_str!r} for (adapter_id={adapter_id!r}, "
                f"target_model_id={target_model_id!r}); supported ranges MUST be disjoint"
            )

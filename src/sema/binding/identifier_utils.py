"""Source names and releases made valid as target identifiers and card versions."""

from __future__ import annotations

import re
import unicodedata
from datetime import date

from packaging.version import InvalidVersion, Version

VALID_NAME = re.compile(r"[A-Za-z0-9_]+")
OTHER_CHARACTERS = re.compile(r"[^a-z0-9_]+")
DATE_RELEASE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")


def entity_identifier(name: str) -> str:
    """A valid name is kept, so existing identities stay; any other is converted."""
    if VALID_NAME.fullmatch(name):
        return name
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    converted = OTHER_CHARACTERS.sub("_", ascii_name.lower()).strip("_")
    if not converted:
        raise ValueError(f"entity name {name!r} holds no letter, digit, or underscore")
    return converted


def card_version(release: str) -> str:
    """A ``YYYY-MM-DD`` date becomes ``YYYY.MM.DD``; a PEP 440 release is kept."""
    dated = DATE_RELEASE.fullmatch(release)
    if dated:
        try:
            date.fromisoformat(release)
        except ValueError as exc:
            raise ValueError(f"release {release!r} is not a calendar date") from exc
        return ".".join(dated.groups())
    try:
        Version(release)
    except InvalidVersion as exc:
        raise ValueError(
            f"release {release!r} is neither YYYY-MM-DD nor a PEP 440 version"
        ) from exc
    return release

"""Files on disk: publisher counts, downloads, digests and the manifest that records them."""

from __future__ import annotations

import csv
import datetime
import hashlib
import http.client
import json
import shutil
import ssl
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

import certifi

from sema.log import logger

CHUNK = 1 << 20
TIMEOUT_SECONDS = 600
USER_AGENT = "open-data-load/1"


@dataclass(frozen=True)
class FileRecord:
    source: str
    path: str
    sha256: str
    publisher_count: int
    recorded_at: str


def _open(url: str) -> http.client.HTTPResponse:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    # A uv-managed interpreter's OpenSSL finds no system CA store; certifi ships one.
    context = ssl.create_default_context(cafile=certifi.where())
    response: http.client.HTTPResponse = urllib.request.urlopen(
        request, timeout=TIMEOUT_SECONDS, context=context
    )
    return response


def publisher_count(url: str) -> int:
    """The single value of the single row the count endpoint returns."""
    with _open(url) as response:
        rows = json.load(response)
    (value,) = rows[0].values()
    return int(value)


def download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    logger.info(f"downloading {url} -> {target}")
    with _open(url) as response, partial.open("wb") as handle:
        shutil.copyfileobj(response, handle, CHUNK)
    partial.replace(target)
    logger.info(f"downloaded {target.stat().st_size:,} bytes -> {target}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def csv_headers(path: Path) -> list[str]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return next(csv.reader(handle))


def csv_data_rows(path: Path) -> int:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return sum(1 for record in csv.reader(handle) if record) - 1


def record(source: str, path: Path, count: int) -> FileRecord:
    now = datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")
    return FileRecord(source, str(path), sha256(path), count, now)


def read_manifest(path: Path) -> dict[str, FileRecord]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {name: FileRecord(**entry) for name, entry in raw.items()}


def write_manifest(path: Path, manifest: dict[str, FileRecord]) -> None:
    body = {name: asdict(entry) for name, entry in manifest.items()}
    path.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")

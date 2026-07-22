"""Network fetch helpers for the cBioPortal datahub.

Pulled out of parsers.py to keep that module under 400 lines.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.request import Request, urlopen

from showcase.cbioportal_to_omop.cbioportal_utils import (
    GITHUB_API_TEMPLATE,
    MEDIA_URL_TEMPLATE,
    RAW_URL_TEMPLATE,
)
from sema.log import logger

# Git-LFS pointer files begin with this line. The datahub stores large tables in
# LFS; when GitHub's media CDN lacks an object it 404s and the raw host serves the
# pointer instead of the content. Accepting that pointer silently loads a 2-row
# garbage table, so we detect it and fail loudly (bug: silent LFS-pointer ingest).
_LFS_POINTER_MAGIC = b"version https://git-lfs.github.com/spec/v1"


def fetch_study_files(study_id: str, cache_dir: Path) -> Path:
    study_cache = cache_dir / study_id
    done_marker = study_cache / ".done"
    if done_marker.exists():
        logger.info("Using cached cBioPortal study files at {}", study_cache)
        return study_cache
    study_cache.mkdir(parents=True, exist_ok=True)
    entries = list_study_entries(study_id)
    downloaded = 0
    from showcase.cbioportal_to_omop.parsers import _should_download
    for entry in entries:
        if entry.get("type") != "file":
            continue
        name = entry["name"]
        if not _should_download(name):
            continue
        fetch_lfs_or_raw(study_id, name, study_cache / name)
        downloaded += 1
    done_marker.touch()
    logger.info("Fetched {} cBioPortal files for {}", downloaded, study_id)
    return study_cache


def fetch_lfs_or_raw(study_id: str, filename: str, target: Path) -> None:
    media_url = MEDIA_URL_TEMPLATE.format(study_id=study_id, filename=filename)
    try:
        download_url_to(media_url, target)
        if not _is_lfs_pointer(target):
            return
    except Exception as media_err:
        logger.debug("media URL failed for {}: {}; falling back to raw", filename, media_err)
        raw_url = RAW_URL_TEMPLATE.format(study_id=study_id, filename=filename)
        download_url_to(raw_url, target)
    if not _is_lfs_pointer(target):
        return
    # The content we have is a git-lfs pointer, not the real file. Try an
    # authenticated GitHub API fetch (LFS is resolved server-side when authorized),
    # then fail loudly rather than ingest the pointer as data.
    if _try_authenticated_lfs(study_id, filename, target) and not _is_lfs_pointer(target):
        return
    raise RuntimeError(
        f"cBioPortal file '{filename}' resolved to a git-lfs POINTER, not content "
        f"(GitHub LFS unavailable — media CDN 404, no/failed auth). Set GITHUB_TOKEN "
        f"to authenticate, or retry when the datahub LFS object is available. "
        f"Ingesting the pointer would create a 2-row garbage table."
    )


def _is_lfs_pointer(target: Path) -> bool:
    try:
        with target.open("rb") as fh:
            return fh.read(len(_LFS_POINTER_MAGIC)) == _LFS_POINTER_MAGIC
    except OSError:
        return False


def _try_authenticated_lfs(study_id: str, filename: str, target: Path) -> bool:
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if not token:
        return False
    api_url = (
        f"https://api.github.com/repos/cBioPortal/datahub/contents/"
        f"public/{study_id}/{filename}?ref=master"
    )
    req = Request(
        api_url,
        headers={
            "Accept": "application/vnd.github.raw",
            "Authorization": f"Bearer {token}",
        },
    )
    try:
        with urlopen(req) as resp:
            data = resp.read()
    except Exception as err:  # noqa: BLE001 - report and fall through to fail-fast
        logger.debug("authenticated LFS fetch failed for {}: {}", filename, err)
        return False
    target.write_bytes(data)
    return True


def list_study_entries(study_id: str) -> list[dict[str, str]]:
    url = GITHUB_API_TEMPLATE.format(study_id=study_id)
    req = Request(url, headers={"Accept": "application/vnd.github+json"})
    with urlopen(req) as resp:
        data: bytes = resp.read()
    parsed = json.loads(data.decode("utf-8"))
    if not isinstance(parsed, list):
        raise RuntimeError(f"Unexpected GitHub API response for {study_id}: {parsed!r}")
    return parsed


def download_url_to(url: str, target: Path) -> None:
    logger.info("Downloading {} -> {}", url, target)
    with urlopen(url) as resp:
        with target.open("wb") as out:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)

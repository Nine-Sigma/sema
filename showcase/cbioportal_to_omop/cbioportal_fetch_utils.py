"""Network fetch helpers for the cBioPortal datahub.

Pulled out of parsers.py to keep that module under 400 lines.
"""
from __future__ import annotations

import json
from pathlib import Path
from urllib.request import Request, urlopen

from showcase.cbioportal_to_omop.cbioportal_utils import (
    GITHUB_API_TEMPLATE,
    MEDIA_URL_TEMPLATE,
    RAW_URL_TEMPLATE,
)
from sema.log import logger


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
        return
    except Exception as media_err:
        logger.debug("media URL failed for {}: {}; falling back to raw", filename, media_err)
    raw_url = RAW_URL_TEMPLATE.format(study_id=study_id, filename=filename)
    download_url_to(raw_url, target)


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

"""Copy the corrected Timbr harness runtime closure into a content-addressed directory.

python -m showcase.ontology_query_benchmark.freeze --out <dir> [--source <harness root>]
"""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

HARNESS_ROOT = Path(
    "/Users/deanb/dev/partner/partnerplatform/partner-harness/capabilities/ontology/"
    "databricks-timbr/poc/timbr-demographics-poc"
)
CLOSURE_PATTERNS = (
    "src/timbr_demographics_poc/*.py",
    "pyproject.toml",
    "uv.lock",
    "config/benchmark/cancer_plus_demographics_v2_bench_L_W4S.json",
    "config/*_product_spec_cpd_v2.json",
    "runs/v3_20261021_rw/ab_questions/*_live.json",
    "runs/v3_20261021_rw/gold_live.json",
    "runs/v3_20261021_rw/ab_w4s/cache/*",
)


class FreezeError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def closure_files(source: Path) -> list[str]:
    files: set[str] = set()
    for pattern in CLOSURE_PATTERNS:
        matched = [p for p in source.glob(pattern) if p.is_file()]
        if not matched:
            raise FreezeError(f"declared input missing: {pattern}")
        files.update(p.relative_to(source).as_posix() for p in matched)
    return sorted(files)


def freeze(source: Path, out_root: Path) -> Path:
    """Copy the closure to ``out_root/<closure id>``; reuse an identical earlier freeze."""
    before = {rel: sha256_file(source / rel) for rel in closure_files(source)}
    closure_id = hashlib.sha256(
        "".join(f"{rel}\0{digest}\n" for rel, digest in before.items()).encode()
    ).hexdigest()
    target = out_root / closure_id
    if (target / "closure.json").is_file():
        _verify(target, before)
        return target
    staging = out_root / f"{closure_id}.partial"
    shutil.rmtree(staging, ignore_errors=True)
    for rel in before:
        (staging / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(source / rel), str(staging / rel))
    changed = [rel for rel, digest in before.items() if sha256_file(source / rel) != digest]
    if changed:
        shutil.rmtree(staging)
        raise FreezeError(f"source changed during copy: {', '.join(changed)}")
    _verify(staging, before)
    closure = {"closure_id": closure_id, "source_root": str(source), **_git_state(source),
               "files": before}  # fmt: skip
    (staging / "closure.json").write_text(json.dumps(closure, indent=1, sort_keys=True))
    staging.rename(target)
    return target


def _verify(target: Path, expected: dict[str, str]) -> None:
    differs = [rel for rel, digest in expected.items() if sha256_file(target / rel) != digest]
    if differs:
        raise FreezeError(f"copy differs from source: {', '.join(differs)}")


def _git_state(source: Path) -> dict[str, object]:
    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", "--no-optional-locks", "-C", str(source), *args],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout if result.returncode == 0 else ""

    head = git("rev-parse", "HEAD").strip() or None
    dirty = [line for line in git("status", "--porcelain", "--", ".").splitlines() if line]
    return {"git_head": head, "git_dirty": dirty}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--source", type=Path, default=HARNESS_ROOT)
    args = parser.parse_args(argv)
    print(freeze(args.source, args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

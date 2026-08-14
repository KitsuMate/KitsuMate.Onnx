#!/usr/bin/env python3
"""Download checksum-pinned integration fixtures outside UPM package directories."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
from pathlib import Path, PurePosixPath


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path) -> bool:
    if destination.exists():
        return False
    temporary = destination.with_suffix(destination.suffix + ".partial")
    with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as stream:
        shutil.copyfileobj(response, stream)
    temporary.replace(destination)
    return True


def destination_path(repository_root: Path, relative_destination: str) -> Path:
    relative = PurePosixPath(relative_destination)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError(f"Fixture destination must be a safe repository-relative path: {relative_destination}")
    if relative.parts[0] == "Packages":
        raise ValueError("Fixtures must not be installed into UPM package directories.")
    result = repository_root.joinpath(*relative.parts).resolve()
    if repository_root not in result.parents:
        raise ValueError(f"Fixture destination escapes repository root: {relative_destination}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    args = parser.parse_args()

    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    files = lock.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("Fixture lock must contain at least one file.")
    repository_root = args.repository_root.resolve()
    args.cache.mkdir(parents=True, exist_ok=True)

    for fixture in files:
        fixture_id = fixture["id"]
        if not fixture_id.replace("-", "").replace("_", "").isalnum():
            raise ValueError(f"Fixture id must be alphanumeric, '-' or '_': {fixture_id}")
        target = destination_path(repository_root, fixture["destination"])
        cache_file = args.cache / fixture["sha256"]
        if not cache_file.exists() and target.exists() and sha256(target) == fixture["sha256"]:
            shutil.copyfile(target, cache_file)
            print(f"seeded cache from {target.relative_to(repository_root)}")
        fetched = download(fixture["url"], cache_file)
        actual = sha256(cache_file)
        if actual != fixture["sha256"]:
            cache_file.unlink(missing_ok=True)
            raise RuntimeError(f"Checksum mismatch for fixture {fixture_id}: {actual}")
        print(f"{'fetched' if fetched else 'cache hit'} {fixture_id}")
        if target.exists() and sha256(target) == fixture["sha256"]:
            print(f"reused {target.relative_to(repository_root)}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cache_file, target)
        print(f"installed {target.relative_to(repository_root)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"fixture hydration failed: {error}", file=sys.stderr)
        raise SystemExit(1)

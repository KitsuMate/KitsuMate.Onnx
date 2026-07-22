#!/usr/bin/env python3
"""Download checksum-pinned ONNX Runtime artifacts into a CI workspace."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    if destination.exists():
        return
    temporary = destination.with_suffix(destination.suffix + ".partial")
    with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as stream:
        shutil.copyfileobj(response, stream)
    temporary.replace(destination)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()

    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    args.cache.mkdir(parents=True, exist_ok=True)
    if args.destination.exists():
        shutil.rmtree(args.destination)
    args.destination.mkdir(parents=True)

    for package in lock["packages"]:
        archive = args.cache / f"{package['id']}.{lock['version']}.nupkg"
        download(package["url"], archive)
        with zipfile.ZipFile(archive) as zip_file:
            for file in package["files"]:
                target = args.destination / file["destination"]
                target.parent.mkdir(parents=True, exist_ok=True)
                with zip_file.open(file["source"]) as source, target.open("wb") as destination:
                    shutil.copyfileobj(source, destination)
                actual = sha256(target)
                if actual != file["sha256"]:
                    target.unlink(missing_ok=True)
                    raise RuntimeError(f"Checksum mismatch for {package['id']}:{file['source']}: {actual}")
                print(f"downloaded {target.relative_to(args.destination)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"artifact hydration failed: {error}", file=sys.stderr)
        raise SystemExit(1)

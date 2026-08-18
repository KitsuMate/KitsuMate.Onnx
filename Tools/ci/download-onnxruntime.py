#!/usr/bin/env python3
"""Download checksum-pinned ONNX Runtime artifacts into a CI workspace."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import time
import urllib.request
import uuid
import zipfile
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path, attempts: int = 3) -> None:
    for attempt in range(1, attempts + 1):
        temporary = destination.with_name(f"{destination.name}.partial.{uuid.uuid4().hex}")
        try:
            with urllib.request.urlopen(url, timeout=120) as response, temporary.open("xb") as stream:
                shutil.copyfileobj(response, stream)
            temporary.replace(destination)
            return
        except Exception:
            temporary.unlink(missing_ok=True)
            if attempt == attempts:
                raise
            time.sleep(attempt)


def open_verified_package(url: str, archive: Path) -> zipfile.ZipFile:
    for attempt in range(1, 4):
        if not archive.exists():
            download(url, archive)
        try:
            package = zipfile.ZipFile(archive)
            corrupt = package.testzip()
            if corrupt is not None:
                package.close()
                raise zipfile.BadZipFile(f"corrupt member {corrupt}")
            return package
        except (OSError, zipfile.BadZipFile):
            archive.unlink(missing_ok=True)
            if attempt == 3:
                raise


def contained_target(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    resolved_root = root.resolve()
    if candidate == resolved_root or resolved_root not in candidate.parents:
        raise ValueError(f"artifact destination escapes hydration root: {relative}")
    return candidate


def extract_file(package: zipfile.ZipFile, file: dict, target: Path) -> None:
    archive_path = file.get("archive")
    if not archive_path:
        package.getinfo(file["source"])
        with package.open(file["source"]) as source, target.open("xb") as destination:
            shutil.copyfileobj(source, destination)
        return

    # NuGet mobile artifacts are themselves archives (for example Android AARs).
    # Spool them to disk above a small threshold instead of retaining the whole
    # platform package in managed memory on CI workers.
    with package.open(archive_path) as archive_source:
        with tempfile.SpooledTemporaryFile(max_size=8 * 1024 * 1024) as nested_archive:
            shutil.copyfileobj(archive_source, nested_archive)
            nested_archive.seek(0)
            with zipfile.ZipFile(nested_archive) as nested_zip:
                corrupt = nested_zip.testzip()
                if corrupt is not None:
                    raise zipfile.BadZipFile(f"corrupt nested member {corrupt}")
                nested_zip.getinfo(file["source"])
                with nested_zip.open(file["source"]) as source, target.open("xb") as destination:
                    shutil.copyfileobj(source, destination)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()

    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    args.cache.mkdir(parents=True, exist_ok=True)
    args.destination.mkdir(parents=True, exist_ok=True)
    # Unity-generated importer metadata is tracked separately from hydrated
    # binaries and must survive repeated CI/release hydration.
    for existing in args.destination.rglob("*"):
        if existing.is_file() and existing.suffix.lower() != ".meta":
            existing.unlink()

    expected_destinations = {
        Path(file["destination"]).as_posix()
        for package in lock["packages"]
        for file in package["files"]
    }
    if len(expected_destinations) != sum(len(package["files"]) for package in lock["packages"]):
        raise ValueError("lock contains duplicate artifact destinations")

    for package in lock["packages"]:
        archive = args.cache / f"{package['id']}.{lock['version']}.nupkg"
        with open_verified_package(package["url"], archive) as zip_file:
            for file in package["files"]:
                target = contained_target(args.destination, file["destination"])
                target.parent.mkdir(parents=True, exist_ok=True)
                extract_file(zip_file, file, target)
                actual = sha256(target)
                if actual != file["sha256"]:
                    target.unlink(missing_ok=True)
                    raise RuntimeError(f"Checksum mismatch for {package['id']}:{file['source']}: {actual}")
                print(f"downloaded {target.relative_to(args.destination.resolve())}")

    actual_destinations = {
        path.relative_to(args.destination).as_posix()
        for path in args.destination.rglob("*")
        if path.is_file() and path.suffix.lower() != ".meta"
    }
    if actual_destinations != expected_destinations:
        raise RuntimeError(
            "hydrated payload differs from lock manifest: "
            f"missing={sorted(expected_destinations - actual_destinations)}, "
            f"unexpected={sorted(actual_destinations - expected_destinations)}"
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"artifact hydration failed: {error}", file=sys.stderr)
        raise SystemExit(1)

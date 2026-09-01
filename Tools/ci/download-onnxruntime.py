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
import tarfile
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


def open_verified_package(url: str, archive: Path, archive_format: str):
    for attempt in range(1, 4):
        if not archive.exists():
            download(url, archive)
        try:
            if archive_format == "tar.gz":
                package = tarfile.open(archive, "r:gz")
                package.getmembers()
            else:
                package = zipfile.ZipFile(archive)
                corrupt = package.testzip()
                if corrupt is not None:
                    package.close()
                    raise zipfile.BadZipFile(f"corrupt member {corrupt}")
            return package
        except (OSError, zipfile.BadZipFile, tarfile.TarError):
            archive.unlink(missing_ok=True)
            if attempt == 3:
                raise


def contained_target(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    resolved_root = root.resolve()
    if candidate == resolved_root or resolved_root not in candidate.parents:
        raise ValueError(f"artifact destination escapes provisioning root: {relative}")
    return candidate


def extract_file(package, file: dict, target: Path) -> None:
    if "symlink" in file:
        target.symlink_to(file["symlink"])
        return
    archive_path = file.get("archive")
    if not archive_path:
        if isinstance(package, tarfile.TarFile):
            source = package.extractfile(package.getmember(file["source"]))
            if source is None:
                raise FileNotFoundError(file["source"])
        else:
            package.getinfo(file["source"])
            source = package.open(file["source"])
        with source, target.open("xb") as destination:
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
    parser.add_argument("--platform", choices=("Windows", "Linux", "macOS", "Android", "Managed"))
    args = parser.parse_args()

    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    args.cache.mkdir(parents=True, exist_ok=True)
    args.destination.mkdir(parents=True, exist_ok=True)
    # Unity-generated importer metadata is tracked separately from downloaded
    # binaries and must survive repeated CI/release provisioning.
    for existing in args.destination.rglob("*"):
        relative_parts = existing.relative_to(args.destination).parts
        selected_platform = args.platform is None or (relative_parts and relative_parts[0] in (args.platform, "Managed", "Licenses"))
        if selected_platform and existing.is_file() and existing.suffix.lower() != ".meta":
            existing.unlink()

    selected_files = [
        (package, file)
        for package in lock["packages"]
        for file in package["files"]
        if args.platform is None or Path(file["destination"]).parts[0] in (args.platform, "Managed", "Licenses")
    ]
    expected_destinations = {Path(file["destination"]).as_posix() for _, file in selected_files}
    if len(expected_destinations) != len(selected_files):
        raise ValueError("lock contains duplicate artifact destinations")

    for package in lock["packages"]:
        files = [file for selected_package, file in selected_files if selected_package is package]
        if not files:
            continue
        archive_format = package.get("format", "zip")
        extension = "tar.gz" if archive_format == "tar.gz" else "zip"
        archive = args.cache / f"{package['id']}.{lock['version']}.{extension}"
        try:
            with open_verified_package(package["url"], archive, archive_format) as package_archive:
                for file in files:
                    target = contained_target(args.destination, file["destination"])
                    target.parent.mkdir(parents=True, exist_ok=True)
                    extract_file(package_archive, file, target)
                    if "symlink" in file:
                        print(f"linked {target.relative_to(args.destination.resolve())}")
                        continue
                    actual = sha256(target)
                    if actual != file["sha256"]:
                        target.unlink(missing_ok=True)
                        raise RuntimeError(f"Checksum mismatch for {package['id']}:{file['source']}: {actual}")
                    print(f"downloaded {target.relative_to(args.destination.resolve())}")
        except Exception as error:
            raise RuntimeError(
                f"failed to provision package {package['id']} from {package['url']}: {error}"
            ) from error

    actual_destinations = {
        path.relative_to(args.destination).as_posix()
        for path in args.destination.rglob("*")
        if path.is_file() and path.suffix.lower() != ".meta"
        and (args.platform is None or path.relative_to(args.destination).parts[0] in (args.platform, "Managed", "Licenses"))
    }
    if actual_destinations != expected_destinations:
        raise RuntimeError(
            "provisioned payload differs from lock manifest: "
            f"missing={sorted(expected_destinations - actual_destinations)}, "
            f"unexpected={sorted(actual_destinations - expected_destinations)}"
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"artifact provisioning failed: {error}", file=sys.stderr)
        raise SystemExit(1)

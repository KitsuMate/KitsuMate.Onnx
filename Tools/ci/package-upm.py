#!/usr/bin/env python3
"""Build independent UPM .tgz archives without including test packages or source artifacts."""

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path


def is_production_package(path: Path) -> bool:
    return path.is_dir() and (path / "package.json").is_file() and not path.name.endswith(".tests")


def should_include(path: Path) -> bool:
    return not any(part in {".git", "Library", "Temp", "Artifacts"} for part in path.parts)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packages", type=Path, default=Path("Packages"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    for package in sorted(filter(is_production_package, args.packages.iterdir())):
        manifest = json.loads((package / "package.json").read_text(encoding="utf-8"))
        archive = args.output / f"{manifest['name']}-{manifest['version']}.tgz"
        with tarfile.open(archive, "w:gz") as tar:
            for file in sorted(package.rglob("*")):
                if file.is_file() and should_include(file.relative_to(package)):
                    tar.add(file, arcname=(Path("package") / file.relative_to(package)).as_posix())
        print(archive)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

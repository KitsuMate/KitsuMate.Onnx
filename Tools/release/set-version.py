#!/usr/bin/env python3
"""Set or verify one explicit SemVer across every package in the monorepo."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


SEMVER = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("version", help="Exact SemVer to apply, for example 1.0.0-alpha.2")
    parser.add_argument("--check", action="store_true", help="Verify without changing files")
    args = parser.parse_args()
    if not SEMVER.fullmatch(args.version):
        parser.error(f"invalid semantic version: {args.version}")

    repository = Path(__file__).resolve().parents[2]
    manifests = sorted((repository / "Packages").glob("*/package.json"))
    if not manifests:
        raise RuntimeError("No package manifests found.")
    packages = {json.loads(path.read_text(encoding="utf-8"))["name"] for path in manifests}
    mismatches: list[str] = []

    for path in manifests:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("version") != args.version:
            mismatches.append(f"{manifest['name']} version is {manifest.get('version')}")
            manifest["version"] = args.version
        dependencies = manifest.get("dependencies", {})
        for dependency in packages.intersection(dependencies):
            if dependencies[dependency] != args.version:
                mismatches.append(f"{manifest['name']} requires {dependency}@{dependencies[dependency]}")
                dependencies[dependency] = args.version
        if not args.check:
            path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if args.check and mismatches:
        print("Package versions do not match the release tag:")
        for mismatch in mismatches:
            print(f"- {mismatch}")
        return 1
    print(f"{'verified' if args.check else 'set'} {len(manifests)} package manifests at {args.version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

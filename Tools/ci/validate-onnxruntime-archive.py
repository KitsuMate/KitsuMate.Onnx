#!/usr/bin/env python3
"""Require the released ONNX Runtime UPM payload to exactly match its artifact lock."""

import argparse
import json
import tarfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    arguments = parser.parse_args()
    lock = json.loads(arguments.lock.read_text(encoding="utf-8"))
    expected = {
        f"package/Runtime/Plugins/{file['destination']}"
        for package in lock["packages"]
        for file in package["files"]
    }
    with tarfile.open(arguments.archive, "r:gz") as package:
        actual = {
            name for name in package.getnames()
            if name.startswith("package/Runtime/Plugins/")
            and not name.endswith(".meta")
            and "." in name.rsplit("/", 1)[-1]
        }
    if actual != expected:
        raise ValueError(
            f"ONNX Runtime archive differs from lock: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )
    print(f"Validated exact ONNX Runtime archive payload: {len(expected)} files.")


if __name__ == "__main__":
    main()

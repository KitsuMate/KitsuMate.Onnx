#!/usr/bin/env python3
"""Validate package boundaries without requiring Unity or external artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ARTIFACT_GLOBS = ("*.onnx", "*.ort", "*.gguf", "*.so", "*.dylib", "*.tgz")
CORE_RUNTIME_FORBIDDEN = (
    "Microsoft.ML.OnnxRuntime",
    "OnnxRuntimeBackend",
    "GpuProvider",
    "GraphOptimizationLevel",
    "Unity.InferenceEngine",
    "UnityAiInferenceBackend",
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fail(message: str) -> None:
    raise ValueError(message)


def main() -> int:
    repository_root = Path(__file__).resolve().parents[2]
    packages_root = repository_root / "Packages"
    packages: dict[str, tuple[dict, Path]] = {}

    for manifest_path in sorted(packages_root.glob("*/package.json")):
        manifest = read_json(manifest_path)
        name = manifest.get("name")
        if not isinstance(name, str) or not name.startswith("ai.kitsumate.onnx"):
            fail(f"Invalid package name in {manifest_path}: {name!r}")
        if name in packages:
            fail(f"Duplicate package name: {name}")
        if "sentis" in name.lower():
            fail(f"Legacy Sentis package naming is forbidden: {name}")
        packages[name] = (manifest, manifest_path.parent)

    unity_inference_packages: list[str] = []
    for name, (manifest, package_path) in packages.items():
        for dependency, version in manifest.get("dependencies", {}).items():
            if dependency.startswith("ai.kitsumate.onnx"):
                if dependency not in packages:
                    fail(f"{name} depends on missing internal package {dependency}")
                if packages[dependency][0].get("version") != version:
                    fail(f"{name} requires {dependency}@{version}; local version differs")
            if dependency == "com.unity.sentis":
                fail(f"{name} depends on unsupported com.unity.sentis")
            if dependency == "com.unity.ai.inference":
                unity_inference_packages.append(name)
        for extension in ARTIFACT_GLOBS:
            artifacts = [path for path in package_path.rglob(extension) if path.is_file()]
            if artifacts:
                fail(f"{name} contains forbidden binary/model artifacts: {artifacts[0]}")

    if unity_inference_packages != ["ai.kitsumate.onnx.backend.unity-inference"]:
        fail("com.unity.ai.inference must be a direct dependency only of ai.kitsumate.onnx.backend.unity-inference")

    example_manifest = read_json(repository_root / "ExampleProject~" / "Packages" / "manifest.json")
    example_dependencies = example_manifest.get("dependencies", {})
    for name in packages:
        expected = f"file:../../Packages/{name}"
        if example_dependencies.get(name) != expected:
            fail(f"ExampleProject~ must reference {name} using {expected}")

    test_packages = {name for name in packages if name.endswith(".tests")}
    testables = set(example_manifest.get("testables", []))
    if testables != test_packages:
        fail("ExampleProject~ testables must be exactly the test package set")

    core_runtime = packages_root / "ai.kitsumate.onnx" / "Runtime"
    for source_file in core_runtime.rglob("*.cs"):
        source = source_file.read_text(encoding="utf-8")
        for forbidden in CORE_RUNTIME_FORBIDDEN:
            if forbidden in source:
                fail(f"Core runtime is not backend-neutral: {forbidden} in {source_file}")

    print(f"Validated {len(packages)} packages; {len(test_packages)} optional test packages.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"package-layout validation failed: {error}", file=sys.stderr)
        raise SystemExit(1)

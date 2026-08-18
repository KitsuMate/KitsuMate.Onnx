#!/usr/bin/env python3
"""Validate package boundaries without requiring Unity or external artifacts."""

from __future__ import annotations

import json
import re
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
ONNXRUNTIME_DESTINATIONS = {
    "Managed/Microsoft.ML.OnnxRuntime.dll",
    "Android/arm64-v8a/libonnxruntime.so",
    "Android/armeabi-v7a/libonnxruntime.so",
    "Linux/x86_64/libonnxruntime.so",
    "Linux/x86_64/libonnxruntime_providers_shared.so",
    "Linux/x86_64/libonnxruntime_providers_cuda.so",
    "Linux/x86_64/libonnxruntime_providers_tensorrt.so",
    "Windows/x86_64/onnxruntime.dll",
    "Windows/x86_64/onnxruntime_providers_shared.dll",
    "Windows/x86_64/onnxruntime_providers_cuda.dll",
    "Windows/x86_64/onnxruntime_providers_tensorrt.dll",
}
ANDROID_PLUGIN_METADATA = {
    "Runtime/Plugins/Android/arm64-v8a/libonnxruntime.so.meta": "CPU: ARM64",
    "Runtime/Plugins/Android/armeabi-v7a/libonnxruntime.so.meta": "CPU: ARMv7",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fail(message: str) -> None:
    raise ValueError(message)


def main() -> int:
    repository_root = Path(__file__).resolve().parents[2]
    packages_root = repository_root / "Packages"
    packages: dict[str, tuple[dict, Path]] = {}

    runtime_lock = read_json(repository_root / "Dependencies" / "onnxruntime.lock.json")
    runtime_files = [file for package in runtime_lock.get("packages", []) for file in package.get("files", [])]
    runtime_destinations = [file.get("destination") for file in runtime_files]
    if len(runtime_destinations) != len(set(runtime_destinations)):
        fail("ONNX Runtime lock contains duplicate destinations")
    if set(runtime_destinations) != ONNXRUNTIME_DESTINATIONS:
        missing = sorted(ONNXRUNTIME_DESTINATIONS.difference(runtime_destinations))
        unexpected = sorted(set(runtime_destinations).difference(ONNXRUNTIME_DESTINATIONS))
        fail(f"ONNX Runtime artifact set differs (missing={missing}, unexpected={unexpected})")
    for file in runtime_files:
        if not re.fullmatch(r"[0-9a-f]{64}", file.get("sha256", "")):
            fail(f"ONNX Runtime artifact has invalid SHA-256: {file.get('destination')}")

    onnxruntime_package = packages_root / "ai.kitsumate.onnx.backend.onnxruntime"
    for relative_path, cpu_setting in ANDROID_PLUGIN_METADATA.items():
        metadata_path = onnxruntime_package / relative_path
        if not metadata_path.is_file():
            fail(f"Missing Unity-generated Android plugin metadata: {metadata_path}")
        metadata = metadata_path.read_text(encoding="utf-8")
        if "PluginImporter:" not in metadata or "Android:" not in metadata or cpu_setting not in metadata:
            fail(f"Invalid Android plugin metadata for {cpu_setting}: {metadata_path}")

    for manifest_path in sorted(packages_root.glob("*/package.json")):
        manifest = read_json(manifest_path)
        name = manifest.get("name")
        if not isinstance(name, str) or not name.startswith("ai.kitsumate.onnx"):
            fail(f"Invalid package name in {manifest_path}: {name!r}")
        if name in packages:
            fail(f"Duplicate package name: {name}")
        packages[name] = (manifest, manifest_path.parent)

    unity_inference_packages: list[str] = []
    for name, (manifest, package_path) in packages.items():
        for dependency, version in manifest.get("dependencies", {}).items():
            if dependency.startswith("ai.kitsumate.onnx"):
                if dependency not in packages:
                    fail(f"{name} depends on missing internal package {dependency}")
                if packages[dependency][0].get("version") != version:
                    fail(f"{name} requires {dependency}@{version}; local version differs")
            if dependency == "com.unity.ai.inference":
                if version != "2.6.1":
                    fail(f"{name} must use com.unity.ai.inference@2.6.1")
                unity_inference_packages.append(name)
        for extension in ARTIFACT_GLOBS:
            artifacts = [path for path in package_path.rglob(extension) if path.is_file()]
            if name == "ai.kitsumate.onnx.backend.onnxruntime":
                plugin_root = package_path / "Runtime" / "Plugins"
                artifacts = [
                    path for path in artifacts
                    if not path.is_relative_to(plugin_root)
                    or path.relative_to(plugin_root).as_posix() not in ONNXRUNTIME_DESTINATIONS
                ]
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

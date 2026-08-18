# ONNX Runtime 1.24.4 combined-core feasibility gate

Status: **blocked / not proven** on the current development host.

The repository must not publish DirectML or OpenVINO Build Profiles until both custom cores below are produced from the same pinned ONNX Runtime `v1.24.4` source tree and pass the matrix. Files from independently built official NuGet packages must never be combined into one platform payload.

## Required builds

- Windows x64: CPU + DirectML + CUDA + TensorRT.
- Linux x64: CPU + CUDA + TensorRT + OpenVINO.

Record the source commit, full build command, compiler and SDK versions, core hash, provider hashes, and external dependency closure in the artifact lock.

## Required smoke matrix

1. Start a CPU session on a clean machine without CUDA, TensorRT, or OpenVINO installed.
2. Start DirectML on Windows without CUDA/TensorRT installed.
3. Start OpenVINO on Linux without CUDA/TensorRT installed.
4. Verify CUDA and TensorRT libraries are not loaded before those providers are selected.
5. Select each provider and verify its factory export and a minimal inference session.
6. Inspect the final core and provider binaries (`dumpbin /dependents`, `ldd`, and export-table tooling) and attach the output.

If CPU startup fails because an optional vendor runtime is hard-linked, stop provider expansion. Do not work around it by mixing official packages.

## Current host evidence

- CUDA 13.2 is installed.
- CMake is not installed on the Windows host.
- No TensorRT or OpenVINO SDK/runtime environment is configured.
- A single Windows host cannot certify the Linux dependency/load matrix or the required Intel/NVIDIA hardware runs.

This is an acceptance blocker for new DirectML/OpenVINO payloads, not for the existing CPU and Android artifacts.

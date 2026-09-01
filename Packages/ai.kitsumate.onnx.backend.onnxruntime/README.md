# KitsuMate ONNX Runtime Backend

The default backend uses ONNX Runtime 1.25.1 and selects acceleration automatically:

- Windows x64: DirectML, then CPU.
- Linux x64: WebGPU over Vulkan, then CPU.
- Apple-silicon macOS: CoreML, then CPU.
- Android ARM64/ARMv7: NNAPI, then CPU.

New backend assets use `Automatic`. An Editor migration detects assets serialized
before selection mode existed, writes the zero-valued `Explicit` mode, and preserves
their provider order. Calling `SetProviderOrder` also selects Explicit mode.

Device selection is provider-relative: DirectML device 1 and CUDA device 0 can refer
to the same physical GPU. New assets use automatic device selection, which snapshots
Unity's active render-device vendor/device IDs on the main thread and matches them
against each provider's `OrtEpDevice` list. Existing assets migrate to Explicit and
retain `_gpuDeviceId`. If no exact match exists, one exposed accelerator is selected;
with several unmatched accelerators, provider index 0 is used deterministically and
reported in diagnostics. DirectML is attached through the ORT V2 device API.

Install `ai.kitsumate.onnx.backend.onnxruntime.nvidia` and activate a profile with
`KITSUMATE_ORT_NVIDIA` to prepend TensorRT-RTX and CUDA on Windows/Linux x64.
`KITSUMATE_ORT_CUDA` is a deprecated CUDA-only alias. The legacy pair
`KITSUMATE_ORT_CUDA` + `KITSUMATE_ORT_TENSORRT` maps to NVIDIA automatic behavior.
Standalone `KITSUMATE_ORT_TENSORRT` and `KITSUMATE_ORT_OPENVINO` are build errors.

The provider registry separates registration, platform support, V2 device selection,
session configuration, priority, and diagnostics. WebGPU, CUDA, and TensorRT-RTX are
registered as plug-in EP libraries and attached through the V2 device API. Legacy
`TensorRt` and `OpenVino` enum values remain serialized-compatible but have no module.

Build Profile changes recompile managed code and reselect providers. A Unity restart is
not required for selection changes, although previously loaded inactive native libraries
may remain mapped until restart.

Native files are provisioned from separate checksum locks for this package and the NVIDIA
package. Do not copy NVIDIA files into this package or introduce a second platform core.

TODO: once platform/default provider ABIs warrant independent release cadence, extract
the logically separated payload groups into dedicated packages without changing the
registry or backend asset API.

Automatic sessions also fall through to the next eligible provider when an operational
EP failure occurs during `Run`, `RunAsync`, or `RunOnDevice` (for example a DirectML
driver/operator execution failure). Invalid model graphs, invalid arguments, contract
errors, cancellation, and disposal remain strict errors. Device tensors from a retired
provider generation are staged through CPU before the retry. Explicit provider mode
never changes providers at execution time.

Editor package paths are resolved once on Unity's main thread and registered as native
search roots. Worker-thread session creation therefore does not call Package Manager,
and package installation remains agnostic to registry, Git, local, embedded, or cache
folder names and locations supported by Unity.

# KitsuMate ONNX Runtime Backend

The default backend uses ONNX Runtime 1.30.0 and selects acceleration automatically:

- Windows x64: WebGPU, then CPU.
- Linux x64: WebGPU over Vulkan, then CPU.
- Apple-silicon macOS: CoreML, then CPU.
- Android ARM64/ARMv7: NNAPI, then CPU.

New backend assets use `Automatic`. Calling `SetProviderOrder` selects Explicit mode.

Device selection is provider-relative: WebGPU device 1 and CUDA device 0 can refer
to the same physical GPU. New assets use automatic device selection, which snapshots
Unity's active render-device vendor/device IDs on the main thread and matches them
against each provider's `OrtEpDevice` list. If no exact match exists, one exposed accelerator is selected;
with several unmatched accelerators, provider index 0 is used deterministically and
reported in diagnostics. WebGPU is attached through the ORT V2 device API.

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

This package includes all default native libraries, managed bindings, licenses, and
Unity importer settings as ordinary Git files. Git/submodule and UPM archive installs
require no download step. CI verifies the payload against `Dependencies/onnxruntime.lock.json`.
The optional NVIDIA package is assembled separately as a complete release archive;
do not copy its files into this package or introduce a second platform core.

Windows availability and device discovery load the core from this package's
resolved native directory before calling ONNX Runtime. The Editor validates its
file version against the managed binding. IL2CPP players skip that check because
Unity does not support `FileVersionInfo.GetVersionInfo` there; they still load
the packaged DLL by its explicit path rather than falling through to Windows'
system copy. The Editor build callback supplies the package's linker rules so
IL2CPP preserves ONNX Runtime's marshaled API constructors.

TODO: once platform/default provider ABIs warrant independent release cadence, extract
the logically separated payload groups into dedicated packages without changing the
registry or backend asset API.

Automatic sessions also fall through to the next eligible provider when an operational
EP failure occurs during `Run`, `RunAsync`, or `RunOnDevice` (for example a WebGPU
driver/operator execution failure). Invalid model graphs, invalid arguments, contract
errors, cancellation, and disposal remain strict errors. Device tensors from a retired
provider generation are staged through CPU before the retry. Explicit provider mode
never changes providers at execution time.

`IOnnxDeviceSession.CreateDeviceTensor` allocates a tensor with fixed storage, and
`RunBound` writes outputs into caller-supplied tensors, so a graph can update a
fixed-size cache in place. With `OnnxSessionOptions.EnableGraphCapture`, WebGPU records a
bound run that passes a graph id and replays it afterwards; every other run on that
session is explicitly uncaptured. Captured runs need all nodes on WebGPU and the same
tensors and shapes each time. Bind one tensor as both input and output only for operators
built for it, such as GroupQueryAttention's shared KV cache; WebGPU loses the device when an
ordinary operator reads and writes the same buffer.

Bound tensors cannot move to a fallback provider. When `RunBound`, or `ToCpu`/`CopyFrom`
on a tensor the session created, fails in an Automatic session, the session switches to the
next provider and throws `OnnxProviderFallbackException`. The caller then creates its device
tensors again and repeats the work. A lost device can let a bound run finish and fail only
the next transfer, which is why transfers switch provider too. Threaded engine runtimes
repeat such a request once on the new provider.

On Windows, the WebGPU plugin compiles shaders with DirectX Shader Compiler 1.9 from the
`Microsoft.Direct3D.DXC` package instead of the 1.8.2502 copy in the plugin package. Dawn
targets the highest shader model that D3D12 reports; the Agility SDK 1.619 that ships with
Unity 6.7 reports Shader Model 6.9, which the older compiler rejects.

Editor package paths are resolved once on Unity's main thread and registered as native
search roots. Worker-thread session creation therefore does not call Package Manager,
and package installation remains agnostic to registry, Git, local, embedded, or cache
folder names and locations supported by Unity.

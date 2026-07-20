# KitsuMate ONNX Core

Shared ONNX model, backend, engine-runtime, validation, and Editor installation infrastructure for Unity.

## Architecture

```text
Consumer
  → feature InferenceEngine asset
      → concrete engine configuration
          → one ModelSet asset
              → required OnnxModelAsset files
  → caller-owned OnnxBackend
  → isolated disposable InferenceEngineRuntime
```

- Engine assets contain immutable serialized configuration and no sessions.
- Runtime instances own sessions, cancellation, state, and serialized inference.
- Disposing a runtime never disposes its caller-owned backend.
- One model-set asset represents one installed model variant or quantization.
- Stable model identity and structured diagnostics drive cache invalidation and compatibility checks.
- ONNX models may be imported or external-file-backed under StreamingAssets.

## Usage

```csharp
using InferenceEngineRuntime<MyRequest, MyResult> runtime =
    await engine.CreateRuntimeAsync(backend, cancellationToken);

MyResult result = await runtime.RunAsync(request, cancellationToken);
```

Inference is asynchronous. Synchronous forwarding APIs are intentionally not provided because Unity main-thread preparation combined with worker inference can deadlock when blocked synchronously.

## Model installation

Create a `ModelCatalog` asset, select a target feature model set, and install a variant through its inspector. The shared Editor installer supports resumable partial files, cancellation, SHA-256 verification, atomic finalization, external ONNX data, and automatic model-set assignment.

The default storage root is `Assets/StreamingAssets/KitsuMateModels` and can be changed in `OnnxSettings`. Player builds use installed models read-only; runtime downloading is not included.

## ONNX Runtime

The package is backend-neutral. Install an optional backend package to execute a model; the ONNX Runtime implementation and its native artifacts are distributed separately.

## License

MIT

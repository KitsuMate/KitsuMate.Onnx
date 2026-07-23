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
- A model-set asset selects one repository artifact independently for each required ONNX role.
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

Use a feature ModelSet inspector to scan a Hugging Face repository and select an artifact for each ONNX role. Standard filenames such as `model.onnx`, `model_fp16.onnx`, and `model_q4.onnx` are detected without a repository manifest. The Editor installer uses partial files, cancellation, SHA-256 verification, external ONNX data, engine validation, and automatic model-set assignment.

The default storage root is `Assets/StreamingAssets/KitsuMateModels` and can be changed in `OnnxSettings`. Player builds use installed models read-only; runtime downloading is not included.

ONNX Runtime setup exposes every discovered suffix. Sentis setup exposes only unsuffixed and explicit `fp32` artifacts, imports the raw ONNX as a Unity `ModelAsset`, and validates it before assigning the ModelSet. The same unsuffixed raw file can also be opened directly by ONNX Runtime.

## ONNX Runtime

The package is backend-neutral. Install an optional backend package to execute a model; the ONNX Runtime implementation and its native artifacts are distributed separately.

## License

MIT

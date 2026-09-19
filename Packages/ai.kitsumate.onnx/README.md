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
- ONNX models use a shared disk installation or explicitly configured local/imported sources.

## Usage

```csharp
using InferenceEngineRuntime<MyRequest, MyResult> runtime =
    await engine.CreateRuntimeAsync(backend, cancellationToken);

MyResult result = await runtime.RunAsync(request, cancellationToken);
```

Inference is asynchronous. Synchronous forwarding APIs are intentionally not provided because Unity main-thread preparation combined with worker inference can deadlock when blocked synchronously.

## Model installation

Configure the installation folder in `OnnxSettings`, relative to `Application.persistentDataPath`. Each model set saves its repository, revision, explicit folder beneath that root, and one artifact selection per submodel. The application UI and model-set Inspector use the same `ModelInstallationStore`.

The model-set Inspector exposes graph and text references directly. Each supports **Asset** or **File**, with persistent-data-relative paths for files inside `Application.persistentDataPath`. External absolute paths remain machine-specific.

**Download models** opens the download window, where quant choices stay visible for installed models. Each single-line entry offers **Download** or **Redownload**; **Download all missing files** handles the full selection. Only selected model roles, external weights, and consumer-declared supporting files are included. **Choose file** supplies an existing file; **Use files from folder** accepts a repository-layout folder. The primary action downloads missing files and assigns file references to the model set. Downloads and project asset migration are separate actions.
Downloaded graph schemas are validated before references are assigned. Imported companions use a folder tied to the downloaded identity, so a failed update cannot overwrite the previous default voice.

Completed downloads survive cancellation and are reused on retry. The interrupted file restarts from zero; byte-range resumption is not implemented. A working installation remains available until its replacement completes. `installation.json` records relative paths, repository identity, sizes, and graph metadata. Size checks do not detect same-size content changes. File contents are **not SHA-256 verified**.

**Move files into Assets** in the Inspector moves the assigned files beside the model set and changes successful references to Asset. External weights are copied to preserve any other graph that still uses their original path. Already migrated references are skipped, so the remaining file-reference count also represents partial migration state. A failed import retains a file reference to the new location so it can be retried. Existing destination files are not overwritten. Unity can still generate its own imported data in Library. Unity AI Inference requires imported graphs. Test compatibility in the target player.

**Move files into data folder** performs the reverse operation for ONNX and text source assets: it moves them under the configured persistent-data root and assigns portable File references. Imported Unity AI Inference graphs remain assets because that backend does not accept files.

The download window blocks updates while a model set uses imported Assets. Move that set into the data folder before applying a different revision or precision, then explicitly move it back into Assets after validation. This keeps the prior imported files from being silently replaced by data-folder references.

`StreamingAssets` is a read-only bundled location, not a download destination. A File reference configured from that folder saves a relative path. Desktop players open the packaged file directly. On Android, the runtime copies a graph, its referenced external weights, and text files into build-scoped persistent storage before opening a native session. This bundled-file route needs player testing for each target platform. WebGL uses imported Unity AI Inference assets; raw ONNX sessions and browser downloads are not supported by this workflow. Browser persistent storage is separate from the Editor's disk installation. Merely downloading models in the Editor does not include them in a player.

See [Engine Inspector inference tests](Documentation~/EDITOR_INFERENCE_TESTS.md) for the Edit Mode workflow. A new family provides `BindInstallationAsync` for its file-role contract and a typed family Editor; the core has no dependency on family packages.

## ONNX Runtime

The package is backend-neutral. Install an optional backend package to execute a model; the ONNX Runtime implementation and its native artifacts are distributed separately.

## License

MIT

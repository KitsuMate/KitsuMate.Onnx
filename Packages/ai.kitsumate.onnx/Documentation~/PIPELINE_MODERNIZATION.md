# KitsuMate ONNX Pipeline Modernization

Status: proposed engineering direction  
Applies to: `ai.kitsumate.onnx` and its ASR, TTS, embeddings, lipsync, and motion consumers

## Purpose

The existing packages are useful early implementations, but their conventions should not be treated as stable architecture. This document defines a migration direction that preserves useful package-family patterns while addressing lifecycle, memory, performance, testing, and platform problems discovered in the current implementations.

This work is separate from the Kimodo implementation. Kimodo may become the first consumer of the improved contracts, but other packages should migrate incrementally rather than through one repository-wide rewrite.

## Existing conventions to preserve

The following patterns are worth keeping:

- Standard Unity Package Manager structure: `Runtime`, `Editor`, `Tests`, package-specific assembly definitions, README, changelog, and `package.json`.
- Runtime assemblies depend on the base ONNX package; editor tooling remains in Editor-only assemblies.
- Request/result types are separated from model-specific implementations.
- Python-generated ground truth is used for tokenizer and embedding tests.
- Unity API access is separated from heavyweight background initialization where practical.
- Model-specific tensor names and constants are centralized.
- TTS demonstrates the right general direction by retaining recurrent KV state on the device.
- Results can contain runtime-neutral data with separate conversion to Unity objects, such as PCM data before `AudioClip` creation.

## Existing conventions to replace

The following patterns should not be carried forward:

- ScriptableObject engine assets own mutable native sessions and device state.
- Shared assets can be invoked concurrently without a defined concurrency policy.
- Large models are embedded in serialized `byte[]` data.
- Missing models frequently cause tests to be ignored, allowing an integration suite to pass without performing inference.
- Autoregressive loops recreate dictionaries, lists, masks, tensors, bindings, and CPU copies per step.
- Synchronous loading or inference can be triggered from the Unity main thread.
- Components catch exceptions and return `null`, hiding programmatic failure details.
- Backend availability is optimistic, and provider fallback is not sufficiently observable.
- Several sessions can be created concurrently through a backend that holds mutable provider-selection state.
- Runtime assemblies nominally target platforms for which no ONNX Runtime native library is packaged.

## Target layering

Every model package should use the following conceptual layers:

```text
Configuration or caller-owned model source
                    |
                    v
Plain disposable invocation service
                    |
                    v
Request/result data contracts
                    |
                    v
Optional Unity-facing MonoBehaviour or asset adapter
```

### Configuration is not runtime state

ScriptableObjects may describe models or create services, but they must not own active ONNX sessions. Active sessions, bindings, allocators, and device buffers belong to plain runtime objects with deterministic disposal.

Recommended common interface:

```csharp
public interface IInferenceService<TRequest, TResult> : IDisposable
{
    InferenceServiceState State { get; }

    Awaitable InitializeAsync(
        CancellationToken cancellationToken = default);

    Awaitable<TResult> InvokeAsync(
        TRequest request,
        CancellationToken cancellationToken = default);
}
```

Services must document whether concurrent calls are rejected, serialized, or supported. The default should be to reject concurrent invocation with a typed `InferenceBusyException`.

### Lifecycle

Use a consistent state machine:

```text
Uninitialized -> Initializing -> Ready -> Running -> Ready -> Disposed
                         \-> Faulted
```

Required behavior:

- Initialization is idempotent.
- Invocation before successful initialization throws a typed exception.
- Partial initialization failure disposes all successfully created resources.
- Disposal cancels active work and prevents new invocation.
- Cancellation is propagated through preprocessing, model loops, and postprocessing.
- Runtime services do not depend on Unity domain reload to release resources correctly.
- Editor domain-reload support is implemented as a coordinator around services, not as asset-owned native state.

### Model sources

The base package should support both small embedded models and filesystem-backed models:

```csharp
public abstract record OnnxModelSource;
public sealed record EmbeddedModelSource(OnnxModelAsset Asset) : OnnxModelSource;
public sealed record FileModelSource(string AbsolutePath) : OnnxModelSource;
```

Filesystem-backed loading is the default for large models. It must support ONNX external-data sidecars and must not create an additional full-size managed copy.

Model metadata should include:

- Stable model identifier and revision.
- Main file and sidecar paths.
- Expected SHA-256 hashes.
- Tensor contract.
- Expected precision.
- Supported providers and platforms.

Selection, downloading, Addressables, StreamingAssets extraction, and cache policy remain higher-level responsibilities.

## Provider and platform conventions

Provider selection must be explicit:

```csharp
public enum ExecutionProviderPolicy
{
    RequireCpu,
    RequireCuda,
    PreferCudaAllowCpu,
    RequireTensorRt,
    PreferTensorRtAllowCuda
}
```

Session diagnostics should expose:

- Requested policy.
- Selected provider.
- Whether fallback occurred.
- CUDA device identifier.
- ONNX Runtime/native provider versions.
- Provider warnings and unexpected CPU node assignment.

`IsAvailable` must reflect actual native runtime availability. A hot-path service must never silently become a CPU service.

Packages must distinguish:

1. The managed assembly compiles on a platform.
2. An ONNX Runtime native library is present.
3. A model can initialize on a provider.
4. Performance has been validated on that platform.

## Reusable execution API

The base package should retain the simple CPU-copy API for one-shot models and add a prepared execution API for recurrent or diffusion models.

Required primitives:

```csharp
IDeviceTensor Upload(OnnxTensor source);
IDeviceTensor Allocate(OnnxTensorElementType type, ReadOnlySpan<int> shape);
IOnnxBinding CreateBinding();
```

A reusable binding should support persistent device inputs, preallocated outputs, repeated execution, and explicit synchronization/readback.

Steady-state hot loops should aim for:

- No tensor-name lookup.
- No dictionaries or LINQ.
- No managed allocations.
- No avoidable CPU/device transfer.
- No output allocation.
- Reusable recurrent or ping-pong buffers.

The backend must be stateless or internally synchronized during session creation. Until that is guaranteed, create large sessions sequentially to avoid provider-state races and peak memory spikes.

## Request, result, and error conventions

Requests should be constructor-validated and results should expose read-only properties. Avoid public mutable fields for contracts that must remain valid during asynchronous execution.

Large outputs should use flat arrays or owned memory rather than multidimensional arrays or per-frame objects. Conversion to `AudioClip`, `AnimationClip`, Animator state, or other Unity objects belongs in an adapter that runs on the main thread.

Core services throw typed exceptions:

- `ModelNotFoundException`
- `ModelContractException`
- `ExecutionProviderUnavailableException`
- `InferenceInitializationException`
- `InferenceBusyException`
- `InferenceCancelledException`

Unity components may catch these exceptions and publish UnityEvents. Invocation services must not log-and-return-null.

## Tensor-contract validation

Each service declares its expected graph contract, including tensor name, element type, rank, fixed dimensions, and permitted dynamic dimensions. Initialization validates the loaded session before allocating runtime buffers.

Do not rely on output ordering, `Values.First()`, or dimension inference from raw array length when a declared contract is available.

The base tensor implementation must fully support every advertised element type. In particular, Boolean tensor factories and ONNX Runtime conversion must exist if `Bool` is part of the public enum.

## Testing conventions

Use three explicit test tiers.

### Always-on unit tests

- Small committed fixtures.
- Tensor conversion and shape validation.
- Preprocessing and postprocessing parity.
- Scheduler and recurrent-state math.
- Lifecycle, cancellation, and concurrency behavior.
- Typed failure modes.

### Large-model integration tests

- Models and large fixtures supplied through a configured external directory.
- Real session initialization.
- Python/ONNX numerical parity.
- End-to-end inference.
- Provider assignment and fallback checks.
- Repeated load/invoke/dispose leak tests.

If CI selects this category, missing models or fixtures are failures, not `Assert.Ignore` results.

### Performance tests

- Separate category and designated hardware.
- Warmup followed by p50/p95 measurements.
- Session-load and first-run latency.
- Managed allocations, process RAM, native memory, and VRAM.
- Repeated-inference and repeated-load leak detection.
- Results stored as benchmark artifacts rather than brittle ordinary unit-test thresholds.

Every model package should include or reference Python tooling capable of regenerating tokenizer, preprocessing, single-step, and end-to-end ground truth with model hashes.

## Migration order

1. Add filesystem model sources, strict provider capabilities, complete tensor types, and reusable binding primitives to `ai.kitsumate.onnx`.
2. Implement Kimodo against the new contracts, proving diffusion and GPU-resident execution.
3. Migrate Chatterbox because it exercises recurrent device state and currently allocates heavily inside its token loop.
4. Migrate Whisper to a plain runtime service and device-resident decoder loop.
5. Migrate embeddings and lipsync.
6. Add or update optional MonoBehaviour adapters only after service contracts stabilize.

Existing packages remain functional through compatibility adapters during migration. The modernization should not block Kimodo on completing all older-package migrations.

## Completion criteria

The modernization is successful when:

- Native sessions are no longer owned by shared ScriptableObject assets.
- Large models load by path without duplicate managed buffers.
- Requested providers and fallbacks are observable and testable.
- Recurrent/diffusion loops can execute without per-step managed allocation.
- Cancellation and concurrent invocation have consistent behavior.
- Selected large-model CI jobs cannot pass without running real inference.
- Existing packages compile and retain behavioral compatibility until explicitly migrated.

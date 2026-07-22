# Kimodo v1 Unity/ONNX Runtime Implementation Plan

Status: motion correctness reference, independent CPU text encoder, and constrained inference contracts delivered  
Target package: `ai.kitsumate.onnx.motion`  
Reference model: `nvidia/Kimodo-SOMA-RP-v1.1`

## Implementation checkpoint (2026-07-10)

The runnable Unity implementation covers phases 2, 3, the core phase-6 decoder, CPU text encoding, and model-facing constraint support:

- External-path ONNX Runtime sessions and Boolean tensors are implemented in `ai.kitsumate.onnx`.
- The fixed batch-3 FP16 denoiser runs through CUDA from Unity.
- C# reproduces separated CFG, the cosine/DDIM schedule, deterministic seeded noise, and final motion decoding.
- The runtime accepts an independently generated 4096-value embedding and returns flat frame-major Unity Humanoid rotations plus root motion.
- A deterministic two-step Python/ORT-CUDA fixture with an exact FP16-representable nonzero conditional embedding is committed and checked by the Unity large-model integration test (observed MAE 0.00892, max 0.246).
- A complete 100-step CUDA smoke run produced finite 60-frame motion; the observed cold run was 1.43 s session creation, 6.90 s sampling, 4 ms decoding, and 8.35 s total.
- The initial supplied-embedding motion checkpoint passed 4/4 tests; the current expanded motion and Character Motion authoring assembly passes 28/28 and the reusable base ONNX assembly passes 28/28 in Unity Editor.
- The exact LLM2Vec encoder has been exported with both adapters merged, fixed sequence length 64, instruction-skipping mean pooling, and a `[1,4096]` FP32 output.
- Llama 3 tokenization, Kimodo sanitation, left padding, attention masks, and pooling masks now have Python-to-C# parity fixtures, including Unicode and punctuation cases.
- A standard merged BF16 CPU reference is retained for low-VRAM comparison. On the current 32 GB RAM / RTX 3070 Laptop system it loaded in 51.06 s and encoded one prompt in 305.65 s without relying on GPU capacity.
- Bitsandbytes NF4 encoded the same prompt in 7.09 s at about 4.60 GiB steady CUDA allocation, but only reached 0.8015 cosine to standard BF16. It is therefore a performance comparison, not a correctness oracle or shipping candidate.
- The first effective ORT INT4 CPU candidate uses 224 `MatMulNBits` operators, FP32 activations, fixed length 64, and 5.41 GiB external model data. It loads in about 5.95 s and encodes in about 1.92 s, but block-128 symmetric quantization currently reaches 0.9761 cosine to the BF16 reference, just below the preferred 0.98 gate.
- Block-64 symmetric improves cosine to 0.9784 but increases warm CPU encode time to 2.48 s and external data to 5.61 GiB. Block-128 remains the provisional speed/memory choice unless end-to-end motion tests show a material quality advantage for block-64.
- The package's exact Microsoft.ML.OnnxRuntime 1.24.4 managed/native pair successfully loads the block-128 candidate and produces a finite embedding in 8.72 s cold-load plus 2.25 s encode. Its fixture output is bit-identical to ORT 1.27.
- Unity's in-editor real-model text test passes with 6.09 s session creation and the expected 0.97611733 cosine / 0.32182663 MAE. The CPU text encoder feeding the CUDA Kimodo denoiser also passes real high-level constrained motion generation. Character Motion tests additionally cover intent caching, coordinate conversion, Humanoid/SOMA mapping, project-avatar pose capture, EditorOnly guide creation, runtime compilation without the guide, and clip rebaking. The complete motion assembly passes 28/28 and base ONNX passes 28/28.
- Public contracts now cover capabilities, raw conditioning, root paths/headings, complete full-body keyframes, and hand/foot end effectors without coupling inference to future authoring `MonoBehaviour` shapes.
- The compiler matches NVIDIA Python fixtures for all three semantic constraint families. The batch-3 runtime implements separate text, constraint, and unconditional branches with independent guidance weights.
- Real CUDA tests prove prompt-only/empty identity, zero-guidance identity, finite differentiated output for every constraint family, improved root-target adherence, and a high-level CPU-prompt → semantic-constraint → CUDA-motion invocation.

The text-encoder benchmark numbers above are development-machine observations, not release claims. Block-size and quantization selection remains open until Unity ORT 1.24.4, end-to-end motion, memory, and prompt-corpus tests are complete.

This checkpoint is the inspectable correctness baseline. It deliberately does not claim the phase-4/5 GPU-resident fast path, independent ONNX CPU text encoder, player-build matrix, visual quality suite, or sustained memory benchmark. Those remain release-optimization work rather than blockers for using the first supplied-embedding implementation.

## Objective

Replace the early EMAGE implementation with a Kimodo-only invocation package that accepts a pre-generated or locally generated 4096-value text embedding, optionally compiles model-space semantic constraints, performs motion generation through Microsoft ONNX Runtime, and returns Unity-Humanoid-compatible motion.

Optimization priority:

1. Motion-inference speed.
2. VRAM, native RAM, and managed-allocation efficiency.
3. Structurally valid motion.
4. Similarity to the highest-precision reference.

V1 uses fixed 60-frame generation and supports the constraint families present in NVIDIA's v1.1 model. Pose/path authoring, interpolation policy, IK/FK, retargeting, and scene-facing controllers remain deliberately outside the inference package.

## Supported execution model

```text
Optional CPU LLM2Vec encoder ------+
                                   |
Pre-generated embedding -----------+--> KimodoTextEmbedding [4096]
                                                |
                                                v
                              KimodoMotionGenerator (GPU hot path)
                                                |
                                                v
                                  KimodoHumanoidMotion
```

Text encoding and motion generation are separate invocation services. The package does not implement embedding caching, model downloading, automatic model selection, or usage MonoBehaviours.

Windows x64 CUDA is the validated v1 target. Linux compatibility is retained when inexpensive. Android remains compile-safe but is not a claimed inference target.

## Public API

### Text encoding

```csharp
public interface IKimodoTextEncoder : IDisposable
{
    Awaitable InitializeAsync(CancellationToken cancellationToken = default);

    Awaitable<KimodoTextEmbedding> EncodeAsync(
        string prompt,
        CancellationToken cancellationToken = default);
}
```

`KimodoTextEmbedding` contains exactly 4096 finite FP32 values and a source model identifier. The motion generator converts the values to FP16 and uploads them once per generation.

The encoder implementation performs invocation only. It does not cache or choose models.

### Motion generation

```csharp
public interface IKimodoMotionGenerator : IDisposable
{
    Awaitable InitializeAsync(CancellationToken cancellationToken = default);

    Awaitable<KimodoHumanoidMotion> GenerateAsync(
        KimodoTextEmbedding embedding,
        KimodoGenerationRequest request,
        CancellationToken cancellationToken = default);
}
```

The runtime implementation is a plain sealed class, not a ScriptableObject. It owns its session, bindings, allocators, and device buffers; rejects concurrent invocation; propagates typed errors; and disposes deterministically.

`KimodoGenerationRequest` contains:

- Seed.
- Denoising-step count.
- Text guidance, default `2.0`.
- First heading in radians, default zero.
- Frame count, required to match the fixed model shape.
- Optional immutable semantic constraint set.
- Independent constraint guidance, default `2.0`.

The high-level pipeline compiles semantic constraints. Advanced callers may supply validated `KimodoConditioning` tensors directly. See `CONSTRAINT_CONTRACTS.md` for the stable boundary intended for future controllers.

### Humanoid result

```csharp
public sealed class KimodoHumanoidMotion
{
    public int FrameCount { get; }
    public float FramesPerSecond { get; }

    // Frame-major arrays.
    public Quaternion[] BoneRotationDeltas { get; }
    public Vector3[] RootPositions { get; }
    public Quaternion[] RootRotations { get; }

    public bool[] BoneAvailability { get; }
    public KimodoGenerationDiagnostics Diagnostics { get; }
}
```

Rotations are bind-pose-relative deltas keyed by `HumanBodyBones` in Unity coordinates. A later adapter can apply:

```text
target bind local rotation * generated rotation delta
```

This representation does not require an Avatar, Animator, GameObject, or main-thread Unity API during inference.

Diagnostics are disabled by default and must not retain raw motion tensors unless explicitly requested.

## Package layout

```text
Runtime/
  Common/
    KimodoTextEmbedding.cs
    KimodoGenerationRequest.cs
    KimodoHumanoidMotion.cs
    KimodoCapabilities.cs
  Kimodo/
    KimodoMotionGenerator.cs
    KimodoTextEncoder.cs
    KimodoSampler.cs
    KimodoMotionDecoder.cs
    KimodoHumanoidMapper.cs
    KimodoTensorContract.cs
Editor/
  KimodoModelValidator.cs
  KimodoBenchmarkWindow.cs
Tests/
  Editor/
  PlayMode/
  Fixtures/
```

Runtime, Editor, and test code remain in separate assembly definitions. Large models are not committed to the package.

## Phase 1: authoritative reference fixtures

Extend the upstream Kimodo ONNX exporter to generate fixtures from the original PyTorch pipeline.

Fixture prompts include:

- Empty prompt.
- Walk, run, turn, sit, jump, fall, crawl, dance, and arm gestures.
- Short and long prompts.
- Punctuation-heavy and Unicode prompts.
- At least three fixed seeds for end-to-end tests.

Record:

- Original and sanitized prompt.
- Exact Llama 3 token IDs and attention mask.
- BF16 or FP32 reference embedding.
- FP16 embedding supplied to motion inference.
- Initial Gaussian motion noise.
- Timesteps and diffusion coefficients.
- Predicted-clean outputs at first, middle, and final steps.
- Final normalized `[60,369]` motion.
- Decoded SOMA-77 positions and rotations.
- Final Humanoid mapping in Unity coordinates.

Use compact binary tensors plus a JSON manifest containing shape, dtype, hashes, model revision, seed, guidance, denoising steps, and coordinate conventions.

Commit one small single-step fixture. Resolve full models through installed engine/model-set test fixtures.

## Phase 2: base ONNX prerequisites

Implement these reusable changes in `ai.kitsumate.onnx`:

- Complete Boolean tensor creation, conversion, output reading, and validation.
- Filesystem-backed session creation.
- ONNX external-data support.
- Strict provider availability and selected-provider diagnostics.
- Thread-safe or stateless session creation.
- Device tensor upload and allocation.
- Persistent reusable I/O binding.
- Preallocated named outputs.
- Device tensor shape and element-type metadata.
- Deterministic disposal of all native resources.

Retain the simple CPU-copy API for existing one-shot models. Existing ASR, TTS, embeddings, and lipsync compilation/tests are regression gates.

## Phase 3: correctness reference generator

Use the existing fixed batch-3 FP16 denoiser to implement an easily inspectable C# reference path:

- Validate a supplied 4096-value embedding.
- Expand and zero-pad it to the denoiser text contract.
- Construct conditional, constraint-empty, and unconditional branches.
- Construct an all-valid 60-frame mask.
- Construct empty constraint mask and observed motion.
- Generate deterministic Gaussian initial motion from the request seed.
- Reproduce the Kimodo cosine schedule and timestep spacing.
- Reproduce separated classifier-free guidance.
- Reproduce the deterministic DDIM update.
- Check cancellation between steps.

CFG and DDIM may run on CPU in this phase. This backend exists for parity and diagnosis, not shipping performance.

Do not begin Humanoid decoding until the final normalized `[60,369]` output passes Python fixture comparison.

## Phase 4: fused prompt-only sampling model

Export a fixed-shape ONNX sampling-step graph:

```text
Build regular conditional/unconditional batch
-> Kimodo denoiser
-> CFG combination
-> DDIM update
-> next motion tensor
```

Inputs:

- Current motion `[1,60,369]`, FP16.
- Prompt embedding `[1,4096]`, FP16.
- Step or timestep.
- First heading.
- Guidance weight if retained as a dynamic option.

Constants:

- Text-axis padding to 50.
- Conditional/unconditional branch construction.
- All-valid frame mask.
- Empty constraints.
- Diffusion schedule values where practical.

Output:

- Next motion `[1,60,369]`, FP16.

Use regular batch-2 CFG for prompt-only generation. Preserve the batch-3 export as the constrained/reference basis.

Do not unroll all denoising steps into one enormous graph.

## Phase 5: fully GPU-resident sampling

At initialization or generation preparation:

- Upload the prompt embedding once.
- Allocate two fixed motion buffers.
- Allocate scalar/schedule buffers.
- Build two reusable bindings: A to B and B to A.
- Preallocate all outputs.

During sampling:

- Alternate bindings and motion buffers.
- Update only the step scalar if it cannot be preloaded.
- Perform no per-step managed allocation.
- Perform no per-step motion readback.
- Copy only the final motion tensor to CPU.

Benchmark CUDA Graph capture after the stable-buffer path works. Enable it only if it improves end-to-end median latency by at least 5%.

## Phase 6: motion decoding

Port and parity-test in this order:

1. Feature unnormalization.
2. 369-feature slicing.
3. Continuous-6D rotation normalization.
4. Rotation-matrix and quaternion conversion.
5. Root trajectory reconstruction.
6. Root heading reconstruction.
7. SOMA-30 internal to SOMA-77 external conversion.
8. Global-to-local rotation conversion.
9. Kimodo-to-Unity handedness and axis conversion.
10. SOMA-77 to `HumanBodyBones` mapping.
11. Bind-pose-relative rotation deltas.

Use flat reusable arrays. Avoid multidimensional arrays, LINQ, Euler conversions, and per-frame objects in the decoding path.

## Phase 7: independent CPU text encoder

After supplied embeddings work end to end:

- Merge the exact MNTP and supervised LLM2Vec adapters before export.
- Preserve bidirectional attention and mean pooling.
- Export an encoder-only graph with one pooled `[B,4096]` result.
- Keep sanitation and Llama 3 tokenization outside ONNX.
- Compare fixed token lengths 64 and 128; prefer 64 unless corpus testing shows material truncation.
- Export FP32 reference, INT8, and weight-only INT4 candidates.
- Keep external model data rather than embedding a multi-gigabyte graph.
- Use ONNX Runtime CPU execution.
- Load and unload independently of the CUDA motion generator.

Maintain a non-ONNX PyTorch reference matrix for export validation:

- Run the ordinary merged BF16 LLM2Vec/Llama encoder fully on CUDA when it fits.
- When VRAM is insufficient, run the same unquantized merged encoder with Accelerate `device_map="auto"`, explicit CPU offload, per-device `max_memory`, and a disk offload directory if required.
- Treat CPU offload as a slow authoritative reference, not a Unity deployment option. It must preserve the normal Llama weights, both merged adapters, bidirectional attention, prompt mask, and mean pooling.
- Retain bitsandbytes NF4 CUDA as an additional low-memory PyTorch comparison, but never use its output as the sole reference for ONNX correctness.
- Record the effective device map, peak CPU RAM, peak VRAM, load time, and encode time so an apparent numerical difference is not confused with a different model composition or execution mode.

Do not use bitsandbytes NF4 for the Unity CPU runtime. It remains a PyTorch/CUDA reference.

Reuse the existing tokenizer library only after exact Llama 3 tokenization parity passes.

## Model test plan

### Static contract tests

For every candidate model:

- Run `onnx.checker`.
- Verify tensor names, dtypes, ranks, and fixed dimensions.
- Verify opset and custom domains.
- Verify external-data completeness and SHA-256.
- Initialize from a filesystem path.
- Record provider node assignment.
- Reject CUDA candidates with substantial unexpected CPU assignment.
- Verify the model manifest matches the actual graph.

### Numerical tests

Denoiser FP16 single-step limits:

- Mean absolute error no greater than `0.005`.
- Maximum absolute error no greater than `0.12`.
- Every output finite.

Text encoder:

- Token IDs and attention mask match Hugging Face exactly.
- Pooling masks match the Python LLM2Vec instruction-skipping behavior exactly.
- Establish the primary reference with the standard merged BF16 encoder, using full CUDA or CPU offload according to available VRAM.
- Verify that full-CUDA and CPU-offloaded standard Llama executions agree within expected BF16 tolerance before either is used to judge ONNX.
- FP32 ONNX pooling matches the standard merged Python reference within floating-point tolerance.
- Compare INT8, INT4, and NF4 embeddings independently against both the standard reference and FP32 ONNX output.
- Quantized embeddings target cosine similarity of at least `0.98`; permit a lower candidate only when end-to-end motion testing demonstrates that it is materially faster or smaller without prompt collapse.
- Output is exactly 4096 finite values.

Sampler:

- Timesteps and diffusion coefficients match Python.
- CFG and DDIM have small hand-computable unit cases.
- First, middle, and final checkpoint tensors match precision-specific reference tolerances.

### Motion validity tests

For every prompt, seed, precision, and step-count candidate:

- No NaN or infinity.
- Quaternion norm error below `1e-3`.
- No discontinuous quaternion sign changes.
- Bone lengths remain invariant after forward kinematics.
- Required Humanoid torso and limb bones are mapped.
- Root and joint velocities remain inside broad physical sanity limits.
- Requested first heading is respected.
- Identical runs are deterministic on the same provider.
- Action prompts do not collapse into identical stationary poses.
- No explosive root trajectory or severe frame-to-frame jitter.

Produce side-by-side reference/candidate clips and numerical reports.

## Optimization matrix

Motion candidates:

- Existing batch-3 FP16 CUDA model.
- Fused batch-2 FP16 CUDA sampling model.
- TensorRT FP16 with engine caching.
- TensorRT INT8 only if calibration remains straightforward.
- Denoising counts 10, 20, 25, 50, and 100.

Do not pursue FP8 on RTX 3070-class hardware.

A candidate replaces the baseline only when it:

- Improves median end-to-end latency by at least 10%, or peak model memory by at least 20%.
- Passes structural motion tests.
- Avoids unexpected CPU fallback.
- Produces recognizably prompt-dependent motion.

The default step count is the lowest candidate passing those gates.

CPU text candidates:

- Standard merged BF16 PyTorch, full CUDA when it fits, as the primary reference.
- Standard merged BF16 PyTorch with CPU/disk offload when VRAM is insufficient, as the fallback reference.
- Bitsandbytes NF4 CUDA as a low-memory comparison reference.
- FP32.
- INT8.
- Weight-only INT4 with supported block-size variants.

Only ONNX Runtime candidates are eligible for the Unity CPU runtime. Choose the fastest candidate meeting the embedding and end-to-end motion gates, and skip quantization when speed/RAM improvement is negligible.

## Runtime and memory testing

For each motion candidate:

- Warm up 10 generations.
- Measure at least 50 generations.
- Record session-load, first-run, p50/p95 step, final-copy, decode, and total latency.
- Record managed heap, process RAM, native memory, steady VRAM, and peak VRAM.
- Run 20 consecutive generations and reject monotonic memory growth.
- Run 10 load/generate/dispose cycles and verify memory returns near baseline.
- Cancel during loading and during multiple sampling steps.
- Confirm the Unity main thread remains responsive.

Required steady-state properties:

- No duplicate 551 MB managed model array.
- No managed allocation inside the denoising loop.
- No per-step motion-tensor readback.
- Motion inference coexists with the current Unity renderer inside 8 GB VRAM.

## Test assemblies and CI

Use separate categories:

- Always-on Editor unit tests with small committed fixtures.
- PlayMode/player lifecycle tests.
- Large-model integration tests using explicitly installed engine/model-set assets.
- Hardware-specific performance tests.

When large-model CI is selected, missing models or fixtures fail the job rather than producing ignored tests.

Mandatory v1 matrix:

- Windows Editor CUDA.
- Windows x64 Mono development player.
- Windows x64 IL2CPP release player.
- Windows provider-unavailable and strict-fallback tests.
- Base ONNX and dependent-package regressions.

Best effort:

- Linux x64 compilation and ORT smoke test.
- Linux CUDA test when hardware is available.
- Android compilation and clear unsupported-provider reporting.

Android runtime inference is not a v1 release gate.

## Migration and delivery

- Remove EMAGE runtime/editor implementation.
- Remove the audio-specific `MotionEngine`.
- Replace Euler and multidimensional `MotionResult`.
- Release as a breaking motion-package version.
- Keep namespace `KitsuMate.Onnx.Motion`, with Kimodo implementation under `.Kimodo`.
- Update README, changelog, model contract, provider requirements, and benchmark results.
- Include examples for supplied embeddings and independent CPU encoding.
- Do not include downloader, cache, automatic model selection, or usage MonoBehaviours.

## V1 completion criteria

Kimodo v1 is complete when:

- Supplied embeddings generate deterministic 60-frame Humanoid motion.
- Python and C# reference paths pass numerical checkpoints.
- The selected fast path retains motion buffers on the GPU through the denoising loop.
- Model precision and step count are selected from recorded benchmarks rather than assumptions.
- Repeated inference and load/dispose tests show no monotonic memory growth.
- Windows CUDA works in Editor, Mono, and IL2CPP builds.
- CPU text encoding is independently invocable when its model is supplied.
- Unsupported constraints and providers fail explicitly; supported constraints pass Python parity and real-model behavior tests.

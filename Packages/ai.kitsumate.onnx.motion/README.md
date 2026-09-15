# KitsuMate ONNX Motion

Kimodo character-motion inference and Unity Humanoid authoring using the shared KitsuMate ONNX engine/model-set architecture.

## Configuration

Create and assign these assets:

1. `KimodoModelSet`, referencing an installed Kimodo ONNX artifact.
2. `KimodoEngine`, referencing that model set.
3. `Llm2VecModelSet` and `Llm2VecEmbeddingEngine` from `ai.kitsumate.onnx.embeddings`.
4. Caller-owned `OnnxBackend` assets for motion and embedding inference.

Each ONNX role can select its own available artifact type in the ModelSet inspector. Engine assets are immutable configuration; inference state lives in disposable runtimes.

```csharp
using CharacterMotionEngineRuntime runtime =
    (CharacterMotionEngineRuntime)await motionEngine.CreateRuntimeAsync(motionBackend, cancellationToken);

CharacterMotionResult result = await runtime.RunAsync(
    new CharacterMotionRequest
    {
        Segments = new[] { new CharacterMotionSegment(embedding,
            new KimodoGenerationRequest(frameCount: 1800, denoisingSteps: 25)) },
        Constraints = constraints,
    },
    cancellationToken);
```

Install the model through **Download Models** in the model-set inspector. The engine uses that model set and its assigned backend.

## Authoring

`CharacterMotion` compiles root, full-body, and end-effector constraints without depending on the Editor guide skeleton. Assign its motion engine/backend and embedding engine/backend, then use the inspector to bake embeddings and animation clips.

See [Character Motion Authoring](Documentation~/CHARACTER_MOTION_AUTHORING.md) and [Constraint Contracts](Documentation~/CONSTRAINT_CONTRACTS.md).

## Duration and continuation

Each segment specifies its output frame count at 30 FPS (minimum two frames). A 60-second segment requests 1800 frames. The runtime balances long segments into inference windows of at most 300 frames, including 1–19 conditioning frames from the preceding window (five by default). History is generated only once in the returned timeline; it does not shorten the requested duration. Segments can use different embeddings and sampling settings.

`CharacterMotionRequest.PreviousMotion` can continue an earlier generated result. Authoring bakes retain the last 19 canonical frames for the same purpose. Constraints use frame indices over the complete new output; previous-motion history precedes frame zero. Cancellation and completed-frame progress are available on the runtime request.

The model has dynamic temporal axes, not an unlimited trained context. Longer requests use conditioned continuation rather than one growing attention sequence. Total output memory still grows with duration. There is no fixed product duration cap, subject to available memory and integer frame counts.

The supported exports are `onnx/model_fp16.onnx` and `onnx/model_fp32.onnx` in [KitsuMate's Kimodo model repository](https://huggingface.co/KitsuMate/Kimodo-SOMA-RP-v1.1-ONNX). Fixed-length exports are replaced. Existing clips should be rebaked to store continuation history and refresh model identity.

## Reference behavior

Window conditioning follows [NVIDIA Kimodo's sequence implementation](https://github.com/nv-tlabs/kimodo/blob/1aece8c124d73d255ceff5086d983b844c9f4e94/kimodo/model/kimodo_model.py): previous full-body positions and hand/foot orientations, planar recentering, and separated classifier-free guidance. The denoiser exports were compared with the original PyTorch model at lengths 2–300, including complete 25-step sampling at 60 and 300 frames.

The managed runtime uses canonical forward kinematics, constrained branch fitting, two-bone limb correction and planted-foot anchors before retaining continuation history. This is a smaller managed implementation, not a numerical port of NVIDIA's native MotionCorrection solver. Feasible constraints and sequence continuity need motion-specific evaluation; impossible targets cannot preserve both exact position and bone lengths.

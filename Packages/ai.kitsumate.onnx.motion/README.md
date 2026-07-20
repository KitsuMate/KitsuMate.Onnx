# KitsuMate ONNX Motion

Kimodo character-motion inference and Unity Humanoid authoring using the shared KitsuMate ONNX engine/model-set architecture.

## Configuration

Create and assign these assets:

1. `KimodoModelSet`, referencing one installed Kimodo ONNX variant.
2. `KimodoEngine`, referencing that model set.
3. `Llm2VecModelSet` and `Llm2VecEmbeddingEngine` from `ai.kitsumate.onnx.embeddings`.
4. Caller-owned `OnnxBackend` assets for motion and embedding inference.

Each precision or quantization is a separate model-set asset. Engine assets are immutable configuration; inference state lives in disposable runtimes.

```csharp
using CharacterMotionEngineRuntime runtime =
    (CharacterMotionEngineRuntime)await motionEngine.CreateRuntimeAsync(motionBackend, cancellationToken);

CharacterMotionResult result = await runtime.RunAsync(
    new CharacterMotionRequest
    {
        Embedding = embedding,
        Constraints = constraints,
        Generation = generationSettings,
    },
    cancellationToken);
```

Models are installed through shared ONNX `ModelCatalog` assets. Public raw filesystem model paths and environment-variable configuration are not supported.

## Authoring

`CharacterMotion` compiles root, full-body, and end-effector constraints without depending on the Editor guide skeleton. Assign its motion engine/backend and embedding engine/backend, then use the inspector to bake embeddings and animation clips.

See [Character Motion Authoring](Documentation~/CHARACTER_MOTION_AUTHORING.md) and [Constraint Contracts](Documentation~/CONSTRAINT_CONTRACTS.md).

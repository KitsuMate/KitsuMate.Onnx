# KitsuMate ONNX Embeddings

Text embedding engines using the shared configuration/runtime convention.

- `TextEmbeddingEngine` supports conventional transformer sentence embeddings.
- `Llm2VecEmbeddingEngine` implements the fixed Llama 3 LLM2Vec contract required by Kimodo.
- Each model size, precision, or quantization is represented by a separate model-set asset.

```csharp
using InferenceEngineRuntime<EmbeddingRequest, EmbeddingResult> runtime =
    await embeddingEngine.CreateRuntimeAsync(callerOwnedBackend, cancellationToken);

EmbeddingResult result = await runtime.RunAsync(
    new EmbeddingRequest(new[] { "first text", "second text" }),
    cancellationToken);
```

The backend remains caller-owned. Disposing the runtime releases sessions but never disposes the backend.

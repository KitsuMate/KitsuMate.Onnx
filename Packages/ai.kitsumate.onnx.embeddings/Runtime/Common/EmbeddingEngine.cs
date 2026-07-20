using System;
using KitsuMate.Onnx;

namespace KitsuMate.Onnx.Embeddings
{
    public enum EmbeddingPooling { ModelDefault, Mean, Cls }

    [Serializable]
    public sealed class EmbeddingRequest
    {
        public string[] Texts;
        public bool? Normalize;
        public int? MaxTokenCount;
        public EmbeddingPooling Pooling;
        public EmbeddingRequest(string text) { Texts = new[] { text ?? string.Empty }; }
        public EmbeddingRequest(string[] texts) { Texts = texts ?? Array.Empty<string>(); }
    }

    public abstract class EmbeddingEngine : InferenceEngine<EmbeddingRequest, EmbeddingResult>
    {
        public abstract int EmbeddingDimension { get; }
    }

    public abstract class EmbeddingEngineRuntime : InferenceEngineRuntime<EmbeddingRequest, EmbeddingResult>
    {
        protected EmbeddingEngineRuntime(bool parallel = false) : base(parallel) { }
    }
}

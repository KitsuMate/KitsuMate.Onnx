using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using KitsuMate.Tokenizers;
using UnityEngine;

namespace KitsuMate.Onnx.Embeddings
{
    [CreateAssetMenu(fileName = "Llm2VecEmbeddingEngine", menuName = "KitsuMate/ONNX/Embeddings/LLM2Vec Engine")]
    public sealed class Llm2VecEmbeddingEngine : EmbeddingEngine
    {
        [SerializeField] private Llm2VecModelSet modelSet;
        public override ModelSet ModelSet => modelSet;
        public override int EmbeddingDimension => modelSet != null ? modelSet.EmbeddingDimension : 0;
        protected override InferenceEngineRuntime<EmbeddingRequest, EmbeddingResult> CreateRuntime() => new Llm2VecEmbeddingEngineRuntime(modelSet);
    }

    public sealed class Llm2VecEmbeddingEngineRuntime : EmbeddingEngineRuntime
    {
        private readonly Llm2VecModelSet modelSet;
        private IOnnxSession session;
        private Tokenizer tokenizer;
        public Llm2VecEmbeddingEngineRuntime(Llm2VecModelSet modelSet) { this.modelSet = modelSet; }
        protected override Task OnLoadAsync(CancellationToken cancellationToken)
        {
            tokenizer = Tokenizer.FromTokenizerJson(modelSet.TokenizerJson.bytes, null, modelSet.TokenizerConfig != null ? modelSet.TokenizerConfig.bytes : null);
            return Task.Run(() =>
            {
            session = Backend.CreateSession(modelSet.Encoder);
            foreach (string required in new[] { "input_ids", "attention_mask", "pooling_mask" }) if (!session.InputNames.Contains(required)) throw new InvalidOperationException($"LLM2Vec model is missing '{required}'.");
            if (!session.OutputNames.Contains("embedding")) throw new InvalidOperationException("LLM2Vec model is missing 'embedding'.");
            }, cancellationToken);
        }
        protected override Task<EmbeddingResult> OnRunAsync(EmbeddingRequest request, CancellationToken cancellationToken)
        {
            if (request?.Texts == null || request.Texts.Length == 0) throw new ArgumentException("At least one text is required.", nameof(request));
            return Task.Run(() =>
            {
            var vectors = new float[request.Texts.Length][];
            for (int i = 0; i < vectors.Length; i++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                (long[] ids, long[] attention, long[] pooling) = Tokenize(request.Texts[i]);
                var inputs = new Dictionary<string, OnnxTensor>
                {
                    ["input_ids"] = OnnxTensor.FromArray(ids, new[] { 1, modelSet.SequenceLength }),
                    ["attention_mask"] = OnnxTensor.FromArray(attention, new[] { 1, modelSet.SequenceLength }),
                    ["pooling_mask"] = OnnxTensor.FromArray(pooling, new[] { 1, modelSet.SequenceLength })
                };
                float[] vector = session.Run(inputs)["embedding"].AsFloatArray();
                if (vector.Length != modelSet.EmbeddingDimension) throw new InvalidOperationException($"LLM2Vec returned {vector.Length} values, expected {modelSet.EmbeddingDimension}.");
                vectors[i] = vector;
            }
            return new EmbeddingResult(vectors[0], normalized: false, modelIdentity: modelSet.Identity) { Batch = vectors };
            }, cancellationToken);
        }
        private (long[], long[], long[]) Tokenize(string prompt)
        {
            string sanitized = Sanitize(prompt ?? string.Empty);
            var content = tokenizer.Encode(sanitized, new TokenizerEncodeOptions { AddSpecialTokens = false, Padding = TokenizerPaddingMode.None, Truncation = TokenizerTruncationMode.None, ReturnAttentionMask = true });
            var full = new List<long>(content.Ids.Count + 6) { 128000, 128006, 882, 128007, 271 };
            for (int i = 0; i < content.Ids.Count; i++) full.Add(content.Ids[i]);
            full.Add(128009);
            int retained = Math.Min(full.Count, modelSet.SequenceLength), padding = modelSet.SequenceLength - retained;
            var ids = new long[modelSet.SequenceLength]; var attention = new long[ids.Length]; var pooling = new long[ids.Length];
            for (int i = 0; i < padding; i++) ids[i] = 128001;
            for (int source = 0; source < retained; source++) { int destination = padding + source; ids[destination] = full[source]; attention[destination] = 1; pooling[destination] = source >= 5 ? 1 : 0; }
            if (!pooling.Any(x => x != 0)) throw new ArgumentException("Prompt is too long for the LLM2Vec model.", nameof(prompt));
            return (ids, attention, pooling);
        }
        private static string Sanitize(string text)
        {
            text = string.Join(" ", text.Split((char[])null, StringSplitOptions.RemoveEmptyEntries));
            if (text.Length == 0) return string.Empty;
            int first = 0; while (first < text.Length && !char.IsLetterOrDigit(text[first])) first++; if (first == text.Length) first--;
            text = text.Substring(first).ToLowerInvariant();
            text = char.ToUpper(text[0], CultureInfo.InvariantCulture) + text.Substring(1);
            return ".!?".Contains(text[text.Length - 1]) ? text : text + ".";
        }
        protected override Task OnUnloadAsync(CancellationToken cancellationToken) { session?.Dispose(); session = null; tokenizer = null; return Task.CompletedTask; }
        protected override void OnDispose() { session?.Dispose(); session = null; tokenizer = null; }
    }
}

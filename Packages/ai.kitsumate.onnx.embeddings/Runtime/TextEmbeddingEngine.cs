using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using KitsuMate.Tokenizers;
using UnityEngine;

namespace KitsuMate.Onnx.Embeddings
{
    [CreateAssetMenu(fileName = "TextEmbeddingEngine", menuName = "KitsuMate/ONNX/Embeddings/Text Embedding Engine")]
    public class TextEmbeddingEngine : EmbeddingEngine
    {
        [SerializeField] protected TextEmbeddingModelSet modelSet;
        [SerializeField] private OnnxSessionOptions sessionOptions;
        public void Configure(TextEmbeddingModelSet set, OnnxSessionOptions options = null)
        { modelSet = set; sessionOptions = options; }
        public TextEmbeddingModelSet TypedModelSet => modelSet;
        public override ModelSet ModelSet => modelSet;
        public override int EmbeddingDimension => modelSet != null ? modelSet.EmbeddingDimension : 0;
        protected override InferenceEngineRuntime<EmbeddingRequest, EmbeddingResult> CreateRuntime() => new TextEmbeddingEngineRuntime(modelSet, sessionOptions);
    }

    public sealed class TextEmbeddingEngineRuntime : EmbeddingEngineRuntime
    {
        private readonly TextEmbeddingModelSet modelSet;
        private IOnnxSession session;
        private Tokenizer tokenizer;
        private readonly OnnxSessionOptions options;

        public TextEmbeddingEngineRuntime(TextEmbeddingModelSet modelSet, OnnxSessionOptions options = null) { this.modelSet = modelSet; this.options = options; }

        protected override Task OnLoadAsync(CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();
            // Capture Unity asset data on the main thread; parse large tokenizers off-thread.
            string directory = modelSet.TokenizerDirectory;
            byte[] tokenizerBytes = modelSet.TokenizerModel != null ? modelSet.TokenizerModel.bytes : null;
            byte[] vocabularyBytes = modelSet.Vocabulary != null ? modelSet.Vocabulary.bytes : null;
            return Task.Run(() =>
            {
                tokenizer = !string.IsNullOrEmpty(directory) ? Tokenizer.FromLocal(directory)
                    : tokenizerBytes != null && System.Text.Encoding.UTF8.GetString(tokenizerBytes).TrimStart().StartsWith("{")
                        ? Tokenizer.FromTokenizerJson(tokenizerBytes)
                        : tokenizerBytes != null ? Tokenizer.CreateBpe(vocabularyBytes, tokenizerBytes)
                        : Tokenizer.CreateWordPiece(vocabularyBytes);
                cancellationToken.ThrowIfCancellationRequested();
                session = Backend.CreateSession(modelSet.EmbeddingModel, options);
            }, cancellationToken);
        }

        protected override Task<EmbeddingResult> OnRunAsync(EmbeddingRequest request, CancellationToken cancellationToken)
        {
            if (request?.Texts == null || request.Texts.Length == 0) throw new ArgumentException("At least one input text is required.", nameof(request));
            return Task.Run(() =>
            {
            var sw = Stopwatch.StartNew();
            float[][] vectors = new float[request.Texts.Length][];
            bool normalize = request.Normalize ?? modelSet.NormalizeEmbeddings;
            int maxTokens = Math.Min(request.MaxTokenCount ?? modelSet.MaxSequenceLength, modelSet.MaxSequenceLength);
            if (maxTokens < 2) throw new ArgumentOutOfRangeException(nameof(request.MaxTokenCount));
            for (int index = 0; index < request.Texts.Length; index++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                var tokens = tokenizer.Encode(modelSet.FormatInput(request.Texts[index], request.Purpose), addSpecialTokens: true, maxTokenCount: maxTokens);
                int[] mask = tokens.AttentionMask.ToArray();
                long[] ids = tokens.Ids.Select(x => (long)x).ToArray();
                long[] types = tokens.TypeIds.Count > 0 ? tokens.TypeIds.Select(x => (long)x).ToArray() : new long[ids.Length];
                var inputs = new Dictionary<string, OnnxTensor>
                {
                    ["input_ids"] = OnnxTensor.FromArray(ids, new[] { 1, ids.Length }, "input_ids"),
                    ["attention_mask"] = OnnxTensor.FromArray(mask.Select(x => (long)x).ToArray(), new[] { 1, mask.Length }, "attention_mask"),
                    ["token_type_ids"] = OnnxTensor.FromArray(types, new[] { 1, types.Length }, "token_type_ids")
                };
                foreach (var name in inputs.Keys.Where(name => !session.InputNames.Contains(name)).ToArray()) inputs.Remove(name);
                var outputs = session.Run(inputs);
                float[] vector = Extract(outputs, mask, ids.Length, request.Pooling);
                if (vector.Length != modelSet.EmbeddingDimension || vector.Any(value => float.IsNaN(value) || float.IsInfinity(value)))
                    throw new InvalidOperationException("Embedding output has an invalid dimension or non-finite values.");
                if (normalize) Normalize(vector);
                vectors[index] = vector;
            }
            sw.Stop();
            return new EmbeddingResult(vectors[0], (float)sw.Elapsed.TotalMilliseconds, normalize, modelSet.Identity) { Batch = vectors };
            }, cancellationToken);
        }

        private float[] Extract(IReadOnlyDictionary<string, OnnxTensor> outputs, int[] mask, int length, EmbeddingPooling pooling)
        {
            if (outputs.TryGetValue("sentence_embedding", out OnnxTensor sentence)) return sentence.AsFloatArray();
            float[] data = outputs.TryGetValue("last_hidden_state", out OnnxTensor hidden) ? hidden.AsFloatArray() : outputs.Values.First().AsFloatArray();
            int dim = modelSet.EmbeddingDimension;
            if (data.Length <= dim) return data;
            bool mean = pooling == EmbeddingPooling.Mean || (pooling == EmbeddingPooling.ModelDefault && modelSet.UseMeanPooling);
            var result = new float[dim];
            if (!mean) { Array.Copy(data, result, dim); return result; }
            int count = 0;
            for (int t = 0; t < length; t++) if (mask[t] > 0) { count++; for (int h = 0; h < dim; h++) result[h] += data[t * dim + h]; }
            if (count > 0) for (int h = 0; h < dim; h++) result[h] /= count;
            return result;
        }

        private static void Normalize(float[] value) { float sum = value.Sum(x => x * x); if (sum <= 0) return; float scale = 1f / Mathf.Sqrt(sum); for (int i = 0; i < value.Length; i++) value[i] *= scale; }
        protected override Task OnUnloadAsync(CancellationToken cancellationToken) { DisposeSessions(); return Task.CompletedTask; }
        protected override void OnDispose() => DisposeSessions();
        private void DisposeSessions() { session?.Dispose(); session = null; tokenizer = null; }
    }
}

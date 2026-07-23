using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using KitsuMate.Tokenizers;
using UnityEngine;

namespace KitsuMate.Onnx.Embeddings.Sentis
{
    [CreateAssetMenu(fileName = "SentisEmbeddingModelSet", menuName = "KitsuMate/ONNX/Embeddings/Sentis Model Set")]
    public sealed class SentisEmbeddingModelSet : StandardModelSet
    {
        [SerializeField] private UnityAiInferenceModelAsset embeddingModel;
        [SerializeField] private TextAsset vocabulary;
        [SerializeField] private TextAsset tokenizerModel;
        [SerializeField] private int embeddingDimension = 384;
        [SerializeField] private int maxSequenceLength = 512;
        [SerializeField] private bool useMeanPooling = true;
        [SerializeField] private bool normalizeEmbeddings = true;

        public UnityAiInferenceModelAsset EmbeddingModel => embeddingModel;
        public TextAsset Vocabulary => vocabulary;
        public TextAsset TokenizerModel => tokenizerModel;
        public int EmbeddingDimension => embeddingDimension;
        public int MaxSequenceLength => maxSequenceLength;
        public bool UseMeanPooling => useMeanPooling;
        public bool NormalizeEmbeddings => normalizeEmbeddings;
        public override string DisplayName => string.IsNullOrEmpty(name) ? "Sentis Embedding Model Set" : name;
        public override bool IsComplete => embeddingModel != null && embeddingModel.IsAvailable && (vocabulary != null || tokenizerModel != null);
        public override IOnnxModelSource[] GetAllModels() => embeddingModel == null ? Array.Empty<IOnnxModelSource>() : new IOnnxModelSource[] { embeddingModel };

#if UNITY_EDITOR
        public void SetModels(UnityAiInferenceModelAsset model, TextAsset vocabularyAsset, TextAsset tokenizerAsset)
        {
            embeddingModel = model;
            vocabulary = vocabularyAsset;
            tokenizerModel = tokenizerAsset;
            UnityEditor.EditorUtility.SetDirty(this);
        }
#endif

        public override ModelValidationResult Validate(ModelValidationContext context)
        {
            var result = new ModelValidationResult();
            if (embeddingModel == null || !embeddingModel.IsAvailable) result.Error("missing_model", "A Unity AI Inference ModelAsset is required.");
            if (vocabulary == null && tokenizerModel == null) result.Error("missing_tokenizer", "Vocabulary or tokenizer model file is required.");
            if (tokenizerModel != null && vocabulary == null && !tokenizerModel.text.TrimStart().StartsWith("{"))
                result.Error("missing_vocabulary", "A vocabulary is required for a non-JSON tokenizer model.");
            if (embeddingDimension < 1) result.Error("invalid_dimension", "Embedding dimension must be positive.");
            if (maxSequenceLength < 1) result.Error("invalid_sequence", "Maximum sequence length must be positive.");
            return ValidateCommon(context, result);
        }
    }

    [CreateAssetMenu(fileName = "SentisEmbeddingEngine", menuName = "KitsuMate/ONNX/Embeddings/Sentis Engine")]
    public sealed class SentisEmbeddingEngine : EmbeddingEngine
    {
        [SerializeField] private SentisEmbeddingModelSet modelSet;
        public override ModelSet ModelSet => modelSet;
        public override int EmbeddingDimension => modelSet != null ? modelSet.EmbeddingDimension : 0;
        protected override InferenceEngineRuntime<EmbeddingRequest, EmbeddingResult> CreateRuntime() => new SentisEmbeddingRuntime(modelSet);
    }

    internal sealed class SentisEmbeddingRuntime : EmbeddingEngineRuntime
    {
        private readonly SentisEmbeddingModelSet modelSet;
        private IOnnxSession session;
        private Tokenizer tokenizer;

        public SentisEmbeddingRuntime(SentisEmbeddingModelSet modelSet)
        {
            this.modelSet = modelSet;
        }

        protected override Task OnLoadAsync(CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (Backend is not UnityAiInferenceBackend)
                throw new InvalidOperationException("SentisEmbeddingEngine requires a UnityAiInferenceBackend.");

            tokenizer = CreateTokenizer(modelSet);
            session = Backend.CreateSession(modelSet.EmbeddingModel);
            return Task.CompletedTask;
        }

        protected override Task<EmbeddingResult> OnRunAsync(EmbeddingRequest request, CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (request?.Texts == null || request.Texts.Length == 0)
                throw new ArgumentException("At least one input text is required.", nameof(request));

            var vectors = new float[request.Texts.Length][];
            var stopwatch = Stopwatch.StartNew();
            bool normalize = request.Normalize ?? modelSet.NormalizeEmbeddings;
            int maxTokens = request.MaxTokenCount ?? modelSet.MaxSequenceLength;
            for (int index = 0; index < request.Texts.Length; index++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                var tokens = tokenizer.Encode(request.Texts[index] ?? string.Empty, addSpecialTokens: true, maxTokenCount: maxTokens);
                int[] ids = tokens.Ids.ToArray();
                int[] attention = tokens.AttentionMask.ToArray();
                int[] types = tokens.TypeIds.Count > 0 ? tokens.TypeIds.ToArray() : new int[ids.Length];
                var inputs = new Dictionary<string, OnnxTensor>
                {
                    ["input_ids"] = OnnxTensor.FromArray(ids, new[] { 1, ids.Length }, "input_ids"),
                    ["attention_mask"] = OnnxTensor.FromArray(attention, new[] { 1, attention.Length }, "attention_mask")
                };
                if (session.InputNames.Contains("token_type_ids"))
                    inputs["token_type_ids"] = OnnxTensor.FromArray(types, new[] { 1, types.Length }, "token_type_ids");
                IReadOnlyDictionary<string, OnnxTensor> outputs = session.Run(inputs);
                float[] vector = Pool(outputs, attention, request.Pooling, modelSet.UseMeanPooling);
                if (normalize) Normalize(vector);
                vectors[index] = vector;
            }
            stopwatch.Stop();
            return Task.FromResult(new EmbeddingResult(vectors[0], (float)stopwatch.Elapsed.TotalMilliseconds, normalize, modelSet.Identity) { Batch = vectors });
        }

        protected override Task OnUnloadAsync(CancellationToken cancellationToken)
        {
            session?.Dispose();
            session = null;
            tokenizer = null;
            return Task.CompletedTask;
        }

        protected override void OnDispose()
        {
            session?.Dispose();
            session = null;
            tokenizer = null;
        }

        private static Tokenizer CreateTokenizer(SentisEmbeddingModelSet modelSet)
        {
            TextAsset tokenizerAsset = modelSet.TokenizerModel;
            if (tokenizerAsset != null && tokenizerAsset.text.TrimStart().StartsWith("{"))
                return Tokenizer.FromTokenizerJson(tokenizerAsset.bytes);
            if (tokenizerAsset != null)
            {
                if (modelSet.Vocabulary == null)
                    throw new InvalidOperationException("A vocabulary asset is required when the Unity AI Inference embedding tokenizer is not tokenizer.json.");
                return Tokenizer.CreateBpe(modelSet.Vocabulary.bytes, tokenizerAsset.bytes);
            }
            if (modelSet.Vocabulary == null)
                throw new InvalidOperationException("A vocabulary or tokenizer asset is required for Unity AI Inference embeddings.");
            return Tokenizer.CreateWordPiece(modelSet.Vocabulary.bytes);
        }

        private static float[] Pool(IReadOnlyDictionary<string, OnnxTensor> outputs, int[] attentionMask, EmbeddingPooling pooling, bool modelDefaultMean)
        {
            if (outputs.TryGetValue("sentence_embedding", out OnnxTensor sentenceEmbedding))
                return sentenceEmbedding.AsFloatArray();
            OnnxTensor output = outputs.Values.FirstOrDefault(value => value.ElementType == OnnxTensorElementType.Float)
                ?? throw new InvalidOperationException("Unity AI Inference embedding model did not produce a float output.");
            float[] values = output.AsFloatArray();
            bool meanPooling = pooling == EmbeddingPooling.Mean || (pooling == EmbeddingPooling.ModelDefault && modelDefaultMean);
            if (!meanPooling || output.Shape.Length < 3)
            {
                int outputDimension = output.Shape.Length > 0 ? output.Shape[output.Shape.Length - 1] : values.Length;
                if (values.Length <= outputDimension) return values;
                var cls = new float[outputDimension];
                Array.Copy(values, cls, outputDimension);
                return cls;
            }

            int sequenceLength = output.Shape[output.Shape.Length - 2];
            int dimension = output.Shape[output.Shape.Length - 1];
            var result = new float[dimension];
            int count = 0;
            for (int token = 0; token < Math.Min(sequenceLength, attentionMask.Length); token++)
            {
                if (attentionMask[token] == 0) continue;
                int offset = token * dimension;
                for (int dimensionIndex = 0; dimensionIndex < dimension; dimensionIndex++) result[dimensionIndex] += values[offset + dimensionIndex];
                count++;
            }
            if (count > 0) for (int index = 0; index < result.Length; index++) result[index] /= count;
            return result;
        }

        private static void Normalize(float[] values)
        {
            double sum = 0;
            foreach (float value in values) sum += value * value;
            if (sum <= 0) return;
            float inverseLength = 1f / Mathf.Sqrt((float)sum);
            for (int index = 0; index < values.Length; index++) values[index] *= inverseLength;
        }
    }
}

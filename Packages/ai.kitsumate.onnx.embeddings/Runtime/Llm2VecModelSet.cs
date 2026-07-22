using UnityEngine;

namespace KitsuMate.Onnx.Embeddings
{
    [CreateAssetMenu(fileName = "Llm2VecModelSet", menuName = "KitsuMate/ONNX/Embeddings/LLM2Vec Model Set")]
    public sealed class Llm2VecModelSet : StandardModelSet
    {
        [SerializeField] private OnnxModelReference encoderSource = new();
        [SerializeField] private TextAsset tokenizerJson;
        [SerializeField] private TextAsset tokenizerConfig;
        [SerializeField] private int sequenceLength = 64;
        [SerializeField] private int embeddingDimension = 4096;

        public OnnxModelReference Encoder => encoderSource;
        public TextAsset TokenizerJson => tokenizerJson;
        public TextAsset TokenizerConfig => tokenizerConfig;
        public int SequenceLength => sequenceLength;
        public int EmbeddingDimension => embeddingDimension;
        public override string DisplayName => string.IsNullOrWhiteSpace(name) ? "LLM2Vec" : name;
        public override bool IsComplete => encoderSource.IsAvailable && tokenizerJson != null;
        public override IOnnxModelSource[] GetAllModels() => new IOnnxModelSource[] { encoderSource };

#if UNITY_EDITOR
        public void SetFiles(TextAsset tokenizer, TextAsset config)
        {
            tokenizerJson = tokenizer;
            tokenizerConfig = config;
            UnityEditor.EditorUtility.SetDirty(this);
        }
#endif

        public override ModelValidationResult Validate(ModelValidationContext context)
        {
            var result = new ModelValidationResult();
            if (!encoderSource.IsAvailable) result.Error("missing_encoder", "LLM2Vec encoder is not available.");
            if (tokenizerJson == null) result.Error("missing_tokenizer", "LLM2Vec tokenizer JSON is not assigned.");
            if (sequenceLength < 4) result.Error("invalid_sequence", "Sequence length must be at least four.");
            if (embeddingDimension < 1) result.Error("invalid_dimension", "Embedding dimension must be positive.");
            RequireInput(encoderSource, "input_ids", result);
            RequireInput(encoderSource, "attention_mask", result);
            RequireInput(encoderSource, "pooling_mask", result);
            RequireOutput(encoderSource, "embedding", result);
            return ValidateCommon(context, result);
        }
    }
}

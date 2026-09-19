using KitsuMate.Onnx.Download;
using System.Threading;
using System.Threading.Tasks;
using System.Linq;
using UnityEngine;

namespace KitsuMate.Onnx.Embeddings
{
    [CreateAssetMenu(fileName = "Llm2VecModelSet", menuName = "KitsuMate/ONNX/Embeddings/LLM2Vec Model Set")]
    public sealed class Llm2VecModelSet : StandardModelSet
    {
        public override string[] DownloadCompanionRoles => new[] { "tokenizer", "tokenizer-config" };

        public override TextFileReference[] GetAllTextFiles() => new[] { tokenizerJson, tokenizerConfig };

        public override System.Collections.Generic.IEnumerable<(string Family, string Repository)> RepositorySuggestions
        {
            get
            {
                yield return ("llm2vec", "KitsuMate/Llama-3-LLM2Vec-MNTP-Supervised-ONNX");
            }
        }

        protected override Task BindInstallationAsync(DownloadedModel installation, ResolvedModelSet resources, CancellationToken cancellationToken)
        {
            installation.ConfigureModel(encoderSource, "encoder");
            tokenizerJson = resources.ReadText(installation, "tokenizer");
            tokenizerConfig = resources.ReadText(installation, "tokenizer-config", optional: true);
            return Task.CompletedTask;
        }

        [SerializeField] private OnnxModelReference encoderSource = new();
        [SerializeField] private TextFileReference tokenizerJson = new();
        [SerializeField] private TextFileReference tokenizerConfig = new();
        [SerializeField] private int sequenceLength = 64;
        [SerializeField] private int embeddingDimension = 4096;

        public OnnxModelReference Encoder => encoderSource;
        public TextFileReference TokenizerJson => tokenizerJson?.IsAvailable == true ? tokenizerJson : null;
        public TextFileReference TokenizerConfig => tokenizerConfig?.IsAvailable == true ? tokenizerConfig : null;
        public int SequenceLength => sequenceLength;
        public int EmbeddingDimension => embeddingDimension;
        public override string DisplayName => string.IsNullOrWhiteSpace(name) ? "LLM2Vec" : name;
        public override bool IsComplete => encoderSource.IsAvailable && tokenizerJson?.IsAvailable == true;
        public override IOnnxModelSource[] GetAllModels() => new IOnnxModelSource[] { encoderSource };

#if UNITY_EDITOR
        public void SetFiles(TextFileReference tokenizer, TextFileReference config)
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
            if (tokenizerJson?.IsAvailable != true) result.Error("missing_tokenizer", "LLM2Vec tokenizer JSON is not assigned.");
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

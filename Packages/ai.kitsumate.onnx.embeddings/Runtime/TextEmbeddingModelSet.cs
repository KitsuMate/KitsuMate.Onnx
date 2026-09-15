using KitsuMate.Onnx.Download;
using System.Threading;
using System.Threading.Tasks;
using System.Linq;
using System.Collections.Generic;
using UnityEngine;
using KitsuMate.Onnx;

namespace KitsuMate.Onnx.Embeddings
{
    /// <summary>
    /// Model set for text embedding models.
    /// </summary>
    [CreateAssetMenu(fileName = "TextEmbeddingModelSet", menuName = "KitsuMate/ONNX/Embeddings/Text Embedding Model Set")]
    public class TextEmbeddingModelSet : StandardModelSet
    {
        public override string[] DownloadCompanionRoles => new[] { "tokenizer", "vocabulary", "tokenizer-config" };

        public override TextFileReference[] GetAllTextFiles() => new[] { _vocabulary, _tokenizerModel, _tokenizerConfig };

        public override System.Collections.Generic.IEnumerable<(string Family, string Repository)> RepositorySuggestions => SupportedRepositories;
        public static System.Collections.Generic.IEnumerable<(string Family, string Repository)> SupportedRepositories
        {
            get
            {
                yield return ("text-embedding", "KitsuMate/all-MiniLM-L6-v2-onnx");
                yield return ("text-embedding", "KitsuMate/embeddinggemma-300m-onnx");
            }
        }

        protected override Task BindInstallationAsync(DownloadedModel installation, ResolvedModelSet resources, CancellationToken cancellationToken)
        {
            installation.ConfigureModel(_embeddingModelSource, "model");
            _vocabulary = resources.ReadText(installation, "vocabulary", optional: true);
            _tokenizerModel = resources.ReadText(installation, "tokenizer", optional: true);
            _tokenizerConfig = resources.ReadText(installation, "tokenizer-config", optional: true);
            tokenizerDirectory = installation.DirectoryPath;
            return Task.CompletedTask;
        }

        [Header("Text Embedding Model")]
        [SerializeField] private OnnxModelReference _embeddingModelSource = new();
        
        [Header("Tokenizer")]
        [SerializeField, Tooltip("Tokenizer vocabulary file")]
        private TextFileReference _vocabulary = new();
        
        [SerializeField, Tooltip("Tokenizer model file (sentencepiece .model or tokenizer.json)")]
        private TextFileReference _tokenizerModel = new();
        [SerializeField] private TextFileReference _tokenizerConfig = new();
        public TextFileReference TokenizerConfig => _tokenizerConfig?.IsAvailable == true ? _tokenizerConfig : null;
        [System.NonSerialized] private string tokenizerDirectory;
        [SerializeField] private string queryPrefix = "";
        [SerializeField] private string documentPrefix = "";
        public string TokenizerDirectory => tokenizerDirectory;
        public string FormatInput(string text, EmbeddingPurpose purpose) =>
            (purpose == EmbeddingPurpose.Query ? queryPrefix : purpose == EmbeddingPurpose.Document ? documentPrefix : "") + (text ?? "");

        [Header("Configuration")]
        [SerializeField, Tooltip("Embedding dimension")]
        private int _embeddingDimension = 384;
        
        [SerializeField, Tooltip("Maximum sequence length")]
        private int _maxSequenceLength = 512;
        
        [SerializeField, Tooltip("Apply mean pooling to token embeddings")]
        private bool _useMeanPooling = true;
        
        [SerializeField, Tooltip("Normalize embeddings to unit length")]
        private bool _normalizeEmbeddings = true;

        
        /// <summary>Display name for this model set.</summary>
        public override string DisplayName => string.IsNullOrEmpty(name) ? "Text Embedding Model Set" : name;
        
        /// <summary>The embedding model.</summary>
        public OnnxModelReference EmbeddingModel => _embeddingModelSource;
        
        /// <summary>Tokenizer vocabulary.</summary>
        public TextFileReference Vocabulary => _vocabulary?.IsAvailable == true ? _vocabulary : null;
        
        /// <summary>Tokenizer model file.</summary>
        public TextFileReference TokenizerModel => _tokenizerModel?.IsAvailable == true ? _tokenizerModel : null;
        
        /// <summary>Embedding dimension.</summary>
        public int EmbeddingDimension => _embeddingDimension;
        
        /// <summary>Maximum sequence length.</summary>
        public int MaxSequenceLength => _maxSequenceLength;
        
        /// <summary>Whether to use mean pooling.</summary>
        public bool UseMeanPooling => _useMeanPooling;
        
        /// <summary>Whether to normalize embeddings.</summary>
        public bool NormalizeEmbeddings => _normalizeEmbeddings;
        
        /// <summary>Check if required models are assigned.</summary>
        private bool HasDownloadedTokenizer => !string.IsNullOrEmpty(tokenizerDirectory) && System.IO.File.Exists(System.IO.Path.Combine(tokenizerDirectory, "tokenizer.json"));

        public override bool IsComplete => _embeddingModelSource.IsAvailable && (HasDownloadedTokenizer || _vocabulary?.IsAvailable == true || _tokenizerModel?.IsAvailable == true);
        
        /// <summary>Gets all models in this set.</summary>
        public override IOnnxModelSource[] GetAllModels()
        {
            return new IOnnxModelSource[] { _embeddingModelSource };
        }
        
        /// <summary>
        /// Assigns model assets, typically called after downloading.
        /// </summary>
        public void SetModels(OnnxModelAsset embeddingModel, TextFileReference vocabulary = null, TextFileReference tokenizerModel = null)
        {
            _embeddingModelSource.ConfigureAsset(embeddingModel);
            if (vocabulary != null) _vocabulary = vocabulary;
            if (tokenizerModel != null) _tokenizerModel = tokenizerModel;
#if UNITY_EDITOR
            UnityEditor.EditorUtility.SetDirty(this);
#endif
        }
        
        /// <summary>Validates the model set configuration.</summary>
        public override ModelValidationResult Validate(ModelValidationContext context)
        {
            var result = new ModelValidationResult();
            if (!_embeddingModelSource.IsAvailable)
                result.Error("missing_model", "Embedding model is required and must be available.");
            if (!HasDownloadedTokenizer && _vocabulary?.IsAvailable != true && _tokenizerModel?.IsAvailable != true)
                result.Error("missing_tokenizer", "Vocabulary or tokenizer model file is required.");
            RequireInput(_embeddingModelSource, "input_ids", result);
            RequireInput(_embeddingModelSource, "attention_mask", result);
            RequireSchema(_embeddingModelSource, result);
            return ValidateCommon(context, result);
        }

#if UNITY_EDITOR
        public void SetTokenizer(TextFileReference vocabulary, TextFileReference tokenizerModel)
        {
            _vocabulary = vocabulary;
            _tokenizerModel = tokenizerModel;
            UnityEditor.EditorUtility.SetDirty(this);
        }
#endif
    }
}

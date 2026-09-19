using KitsuMate.Onnx.Download;
using System.Threading;
using System.Threading.Tasks;
using System.Linq;
using System;
using System.Collections.Generic;
using UnityEngine;

namespace KitsuMate.Onnx.Tts.OmniVoice
{
    public enum OmniVoiceBackboneTopology { Split, Merged }

    [CreateAssetMenu(fileName = "OmniVoiceModelSet", menuName = "KitsuMate/ONNX/TTS/OmniVoice Model Set")]
    public sealed class OmniVoiceModelSet : StandardModelSet
    {
        public override string[] DownloadCompanionRoles => new[] { "tokenizer" };

        public override TextFileReference[] GetAllTextFiles() => new[] { tokenizer };

        public override System.Collections.Generic.IEnumerable<(string Family, string Repository)> RepositorySuggestions
        {
            get
            {
                yield return ("omnivoice", "KitsuMate/omnivoice-onnx");
            }
        }

        protected override Task BindInstallationAsync(DownloadedModel installation, ResolvedModelSet resources, CancellationToken cancellationToken)
        {
            topology = installation.Files.Any(file => file.Role == "merged-backbone") ? OmniVoiceBackboneTopology.Merged : OmniVoiceBackboneTopology.Split;
            if (topology == OmniVoiceBackboneTopology.Merged) installation.ConfigureModel(mergedBackbone, "merged-backbone");
            else
            {
                installation.ConfigureModel(audioEmbeddingsEncoder, "audio-embeddings");
                installation.ConfigureModel(languageDecoder, "language-decoder");
                installation.ConfigureModel(audioHeadsDecoder, "audio-heads");
            }
            installation.ConfigureModel(acousticEncoder, "acoustic-encoder");
            installation.ConfigureModel(semanticEncoder, "semantic-encoder");
            installation.ConfigureModel(quantizerEncoder, "quantizer-encoder");
            installation.ConfigureModel(higgsDecoder, "higgs-decoder");
            tokenizer = resources.ReadText(installation, "tokenizer");
            return Task.CompletedTask;
        }

        [SerializeField] private OmniVoiceBackboneTopology topology = OmniVoiceBackboneTopology.Merged;
        [SerializeField] private OnnxModelReference mergedBackbone = new();
        [SerializeField] private OnnxModelReference audioEmbeddingsEncoder = new();
        [SerializeField] private OnnxModelReference languageDecoder = new();
        [SerializeField] private OnnxModelReference audioHeadsDecoder = new();
        [SerializeField] private OnnxModelReference acousticEncoder = new();
        [SerializeField] private OnnxModelReference semanticEncoder = new();
        [SerializeField] private OnnxModelReference quantizerEncoder = new();
        [SerializeField] private OnnxModelReference higgsDecoder = new();
        [SerializeField] private TextFileReference tokenizer = new();

        public OmniVoiceBackboneTopology Topology => topology;
        public OnnxModelReference MergedBackbone => mergedBackbone;
        public OnnxModelReference AudioEmbeddingsEncoder => audioEmbeddingsEncoder;
        public OnnxModelReference LanguageDecoder => languageDecoder;
        public OnnxModelReference AudioHeadsDecoder => audioHeadsDecoder;
        public OnnxModelReference AcousticEncoder => acousticEncoder;
        public OnnxModelReference SemanticEncoder => semanticEncoder;
        public OnnxModelReference QuantizerEncoder => quantizerEncoder;
        public OnnxModelReference HiggsDecoder => higgsDecoder;
        public TextFileReference Tokenizer => tokenizer?.IsAvailable == true ? tokenizer : null;
        public override string DisplayName => string.IsNullOrWhiteSpace(name) ? "OmniVoice" : name;

        private bool BackboneComplete => topology == OmniVoiceBackboneTopology.Merged
            ? mergedBackbone.IsAvailable
            : audioEmbeddingsEncoder.IsAvailable && languageDecoder.IsAvailable && audioHeadsDecoder.IsAvailable;

        public override bool IsComplete => BackboneComplete && acousticEncoder.IsAvailable && semanticEncoder.IsAvailable &&
            quantizerEncoder.IsAvailable && higgsDecoder.IsAvailable && tokenizer?.IsAvailable == true;

        public override IOnnxModelSource[] GetAllModels()
        {
            var models = new List<IOnnxModelSource>();
            if (topology == OmniVoiceBackboneTopology.Merged) models.Add(mergedBackbone);
            else { models.Add(audioEmbeddingsEncoder); models.Add(languageDecoder); models.Add(audioHeadsDecoder); }
            models.Add(acousticEncoder); models.Add(semanticEncoder); models.Add(quantizerEncoder); models.Add(higgsDecoder);
            return models.ToArray();
        }

        public override ModelValidationResult Validate(ModelValidationContext context)
        {
            var result = new ModelValidationResult();
            if (!BackboneComplete) result.Error("missing_backbone", $"The {topology} OmniVoice backbone is incomplete.");
            if (!acousticEncoder.IsAvailable) result.Error("missing_acoustic_encoder", "Acoustic encoder is unavailable.");
            if (!semanticEncoder.IsAvailable) result.Error("missing_semantic_encoder", "Semantic encoder is unavailable.");
            if (!quantizerEncoder.IsAvailable) result.Error("missing_quantizer_encoder", "Quantizer encoder is unavailable.");
            if (!higgsDecoder.IsAvailable) result.Error("missing_higgs_decoder", "Higgs decoder is unavailable.");
            if (tokenizer?.IsAvailable != true) result.Error("missing_tokenizer", "Tokenizer is not assigned.");
            foreach (IOnnxModelSource model in GetAllModels()) RequireSchema(model, result);
            return ValidateCommon(context, result);
        }

#if UNITY_EDITOR
        public void Configure(OmniVoiceBackboneTopology selectedTopology, TextFileReference selectedTokenizer)
        {
            topology = selectedTopology;
            tokenizer = selectedTokenizer;
            UnityEditor.EditorUtility.SetDirty(this);
        }
#endif
    }
}

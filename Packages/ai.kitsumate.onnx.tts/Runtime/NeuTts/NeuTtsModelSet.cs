using System;
using System.Threading;
using System.Threading.Tasks;
using KitsuMate.Onnx.Download;
using UnityEngine;

namespace KitsuMate.Onnx.Tts.NeuTts
{
    [CreateAssetMenu(fileName = "NeuTtsModelSet", menuName = "KitsuMate/ONNX/TTS/NeuTTS Model Set")]
    public sealed class NeuTtsModelSet : StandardModelSet
    {
        public override string[] DownloadCompanionRoles => new[] { "tokenizer", "neutts-metadata" };

        public override TextFileReference[] GetAllTextFiles() => new[] { tokenizer, metadata };

        public override System.Collections.Generic.IEnumerable<(string Family, string Repository)> RepositorySuggestions
        {
            get
            {
                yield return ("neutts", "KitsuMate/neutts-2e-onnx");
            }
        }

        [SerializeField] private OnnxModelReference backbone = new();
        [SerializeField] private OnnxModelReference codecDecoder = new();
        [Tooltip("Use CPU for the original NeuCodec graph. Disable only when using a compatible overlap-Conv codec.")]
        [SerializeField] private bool useCpuCodec = true;
        [SerializeField] private TextFileReference tokenizer = new();
        [SerializeField] private TextFileReference metadata = new();
        public OnnxModelReference Backbone => backbone;
        public OnnxModelReference CodecDecoder => codecDecoder;
        public bool UseCpuCodec => useCpuCodec;
        public TextFileReference Tokenizer => tokenizer?.IsAvailable == true ? tokenizer : null;
        public TextFileReference Metadata => metadata?.IsAvailable == true ? metadata : null;
        protected override string DefaultFamily => "neutts";
        protected override string DefaultModelId => "neutts-2e";
        public override string DisplayName => "NeuTTS-2E";
        public override bool IsComplete => backbone.IsAvailable && codecDecoder.IsAvailable && tokenizer?.IsAvailable == true && metadata?.IsAvailable == true;
        public override IOnnxModelSource[] GetAllModels() => new IOnnxModelSource[] { backbone, codecDecoder };

        protected override Task BindInstallationAsync(DownloadedModel installation, ResolvedModelSet resources, CancellationToken token)
        {
            installation.ConfigureModel(backbone, "backbone");
            installation.ConfigureModel(codecDecoder, "codec-decoder");
            // A new download has not been verified for GPU decoding. Do not carry
            // a previous graph's GPU opt-in over to a replacement codec.
            useCpuCodec = true;
            tokenizer = resources.ReadText(installation, "tokenizer");
            metadata = resources.ReadText(installation, "neutts-metadata");
            return Task.CompletedTask;
        }

        public override ModelValidationResult Validate(ModelValidationContext context)
        {
            var result = new ModelValidationResult();
            if (!IsComplete) result.Error("incomplete_neutts", "NeuTTS requires a backbone, codec, tokenizer and speaker metadata.");
            if (metadata?.IsAvailable == true)
                try { JsonUtility.FromJson<NeuTtsMetadata>(metadata.text).Validate(); }
                catch (Exception e) { result.Error("invalid_neutts_metadata", e.Message); }
            foreach (var model in GetAllModels()) RequireSchema(model, result);
            RequireInput(backbone, "input_ids", result);
            RequireInput(backbone, "attention_mask", result);
            RequireInput(backbone, "position_ids", result);
            RequireInput(codecDecoder, "codes", result);
            return ValidateCommon(context, result);
        }
    }
}

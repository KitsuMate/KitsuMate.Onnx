using KitsuMate.Onnx.Download;
using System.Threading;
using System.Threading.Tasks;
using System;
using KitsuMate.Onnx.Asr.Whisper;
using UnityEngine;

namespace KitsuMate.Onnx.Asr.Sentis
{
    [CreateAssetMenu(fileName = "SentisWhisperModelSet", menuName = "KitsuMate/ONNX/ASR/Sentis Whisper Model Set")]
    public sealed class SentisWhisperModelSet : StandardModelSet
    {
        public override string[] DownloadCompanionRoles => new[] { "tokenizer" };

        public override TextFileReference[] GetAllTextFiles() => new[] { tokenizerJson };

        public override System.Collections.Generic.IEnumerable<(string Family, string Repository)> RepositorySuggestions =>
            WhisperModelSet.SupportedRepositories;

        protected override Task BindInstallationAsync(DownloadedModel installation, ResolvedModelSet resources, CancellationToken cancellationToken)
        {
            tokenizerJson = resources.ReadText(installation, "tokenizer");
            return Task.CompletedTask;
        }

        [SerializeField] private UnityAiInferenceModelAsset melProcessor;
        [SerializeField] private UnityAiInferenceModelAsset encoder;
        [SerializeField] private UnityAiInferenceModelAsset decoder;
        [SerializeField] private UnityAiInferenceModelAsset decoderWithPast;
        [SerializeField] private TextFileReference tokenizerJson = new();

        public UnityAiInferenceModelAsset MelProcessor => melProcessor;
        public UnityAiInferenceModelAsset Encoder => encoder;
        public UnityAiInferenceModelAsset Decoder => decoder;
        public UnityAiInferenceModelAsset DecoderWithPast => decoderWithPast;
        public TextFileReference TokenizerJson => tokenizerJson?.IsAvailable == true ? tokenizerJson : null;
        public override string DisplayName => string.IsNullOrWhiteSpace(name) ? "Sentis Whisper" : name;
        public override bool IsComplete => IsReady(melProcessor) && IsReady(encoder) && IsReady(decoder) &&
            tokenizerJson?.IsAvailable == true;

        public override IOnnxModelSource[] GetAllModels() => IsReady(decoderWithPast)
            ? new IOnnxModelSource[] { melProcessor, encoder, decoder, decoderWithPast }
            : new IOnnxModelSource[] { melProcessor, encoder, decoder };

        public override ModelValidationResult Validate(ModelValidationContext context)
        {
            var result = new ModelValidationResult();
            if (!IsReady(melProcessor)) result.Error("missing_mel", "A Sentis mel model is required.");
            if (!IsReady(encoder)) result.Error("missing_encoder", "A Sentis encoder model is required.");
            if (!IsReady(decoder)) result.Error("missing_decoder", "A Sentis decoder model is required.");
            if (tokenizerJson?.IsAvailable != true) result.Error("missing_tokenizer", "Tokenizer JSON is required.");
            if (context.Backend != null && context.Backend is not UnityAiInferenceBackend)
                result.Error("invalid_backend", "SentisWhisperEngine requires UnityAiInferenceBackend.");
            return ValidateCommon(context, result);
        }

#if UNITY_EDITOR
        public void SetModels(UnityAiInferenceModelAsset mel, UnityAiInferenceModelAsset audioEncoder,
            UnityAiInferenceModelAsset initialDecoder, UnityAiInferenceModelAsset cachedDecoder, TextFileReference tokenizer)
        {
            melProcessor = mel;
            encoder = audioEncoder;
            decoder = initialDecoder;
            decoderWithPast = cachedDecoder;
            tokenizerJson = tokenizer;
            UnityEditor.EditorUtility.SetDirty(this);
        }
#endif

        private static bool IsReady(UnityAiInferenceModelAsset model) => model != null && model.IsAvailable;
    }

    [CreateAssetMenu(fileName = "SentisWhisperEngine", menuName = "KitsuMate/ONNX/ASR/Sentis Whisper Engine")]
    public sealed class SentisWhisperEngine : AsrEngine
    {
        [SerializeField] private SentisWhisperModelSet modelSet;
        [SerializeField] private string languageOverride = "";
        [SerializeField] private WhisperTask task = WhisperTask.Transcribe;
        [SerializeField] private int maxTokens = 224;
        [SerializeField] private bool verboseLogging;

        public override ModelSet ModelSet => modelSet;

        protected override InferenceEngineRuntime<AsrRequest, TranscriptionResult> CreateRuntime(ModelSet resolvedModelSet)
        {
            var resolved = (SentisWhisperModelSet)resolvedModelSet;
            return new WhisperEngineRuntime(resolved.MelProcessor, resolved.Encoder, resolved.Decoder,
                resolved.DecoderWithPast, resolved.TokenizerJson, languageOverride, task, maxTokens, verboseLogging);
        }

    }
}

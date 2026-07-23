using System;
using KitsuMate.Onnx.Asr.Whisper;
using UnityEngine;

namespace KitsuMate.Onnx.Asr.Sentis
{
    [CreateAssetMenu(fileName = "SentisWhisperModelSet", menuName = "KitsuMate/ONNX/ASR/Sentis Whisper Model Set")]
    public sealed class SentisWhisperModelSet : StandardModelSet
    {
        [SerializeField] private UnityAiInferenceModelAsset melProcessor;
        [SerializeField] private UnityAiInferenceModelAsset encoder;
        [SerializeField] private UnityAiInferenceModelAsset decoder;
        [SerializeField] private UnityAiInferenceModelAsset decoderWithPast;
        [SerializeField] private TextAsset tokenizerJson;

        public UnityAiInferenceModelAsset MelProcessor => melProcessor;
        public UnityAiInferenceModelAsset Encoder => encoder;
        public UnityAiInferenceModelAsset Decoder => decoder;
        public UnityAiInferenceModelAsset DecoderWithPast => decoderWithPast;
        public TextAsset TokenizerJson => tokenizerJson;
        public override string DisplayName => string.IsNullOrWhiteSpace(name) ? "Sentis Whisper" : name;
        public override bool IsComplete => IsReady(melProcessor) && IsReady(encoder) && IsReady(decoder) &&
            tokenizerJson != null;

        public override IOnnxModelSource[] GetAllModels() => IsReady(decoderWithPast)
            ? new IOnnxModelSource[] { melProcessor, encoder, decoder, decoderWithPast }
            : new IOnnxModelSource[] { melProcessor, encoder, decoder };

        public override ModelValidationResult Validate(ModelValidationContext context)
        {
            var result = new ModelValidationResult();
            if (!IsReady(melProcessor)) result.Error("missing_mel", "A Sentis mel model is required.");
            if (!IsReady(encoder)) result.Error("missing_encoder", "A Sentis encoder model is required.");
            if (!IsReady(decoder)) result.Error("missing_decoder", "A Sentis decoder model is required.");
            if (tokenizerJson == null) result.Error("missing_tokenizer", "Tokenizer JSON is required.");
            if (context.Backend != null && context.Backend is not UnityAiInferenceBackend)
                result.Error("invalid_backend", "SentisWhisperEngine requires UnityAiInferenceBackend.");
            return ValidateCommon(context, result);
        }

#if UNITY_EDITOR
        public void SetModels(UnityAiInferenceModelAsset mel, UnityAiInferenceModelAsset audioEncoder,
            UnityAiInferenceModelAsset initialDecoder, UnityAiInferenceModelAsset cachedDecoder, TextAsset tokenizer)
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

        protected override InferenceEngineRuntime<AsrRequest, TranscriptionResult> CreateRuntime() =>
            new WhisperEngineRuntime(modelSet.MelProcessor, modelSet.Encoder, modelSet.Decoder,
                modelSet.DecoderWithPast, modelSet.TokenizerJson, languageOverride, task, maxTokens, verboseLogging);
    }
}

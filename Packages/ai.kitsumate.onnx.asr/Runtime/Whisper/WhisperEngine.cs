using UnityEngine;

namespace KitsuMate.Onnx.Asr.Whisper
{
    [CreateAssetMenu(fileName = "WhisperEngine", menuName = "KitsuMate/ONNX/ASR/Whisper Engine")]
    public sealed class WhisperEngine : AsrEngine
    {
        [SerializeField] private WhisperModelSet modelSet;
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

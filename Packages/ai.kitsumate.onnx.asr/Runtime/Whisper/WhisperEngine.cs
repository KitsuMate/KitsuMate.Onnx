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
        [SerializeField] private OnnxSessionOptions sessionOptions;
        public override ModelSet ModelSet => modelSet;
        public void Configure(WhisperModelSet models, OnnxSessionOptions options = null) { modelSet = models; sessionOptions = options; }
        protected override InferenceEngineRuntime<AsrRequest, TranscriptionResult> CreateRuntime(ModelSet resolvedModelSet)
        {
            var resolved = (WhisperModelSet)resolvedModelSet;
            return new WhisperEngineRuntime(resolved.MelProcessor, resolved.Encoder, resolved.Decoder,
                resolved.DecoderWithPast, resolved.TokenizerJson, languageOverride, task, maxTokens, verboseLogging, sessionOptions);
        }
    }
}

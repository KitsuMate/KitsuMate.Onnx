using UnityEngine;
using UnityEngine.Serialization;

namespace KitsuMate.Onnx.Asr.Whisper
{
    [CreateAssetMenu(fileName = "WhisperEngine", menuName = "KitsuMate/ONNX/ASR/Whisper Engine")]
    public sealed class WhisperEngine : AsrEngine
    {
        [FormerlySerializedAs("_modelSet"), SerializeField] private WhisperModelSet modelSet;
        [FormerlySerializedAs("_languageOverride"), SerializeField] private string languageOverride = "";
        [FormerlySerializedAs("_task"), SerializeField] private WhisperTask task = WhisperTask.Transcribe;
        [FormerlySerializedAs("_maxTokens"), SerializeField] private int maxTokens = 224;
        [SerializeField] private bool verboseLogging;
        public override ModelSet ModelSet => modelSet;
        protected override InferenceEngineRuntime<AsrRequest, TranscriptionResult> CreateRuntime() => new WhisperEngineRuntime(modelSet, languageOverride, task, maxTokens, verboseLogging);
    }
}

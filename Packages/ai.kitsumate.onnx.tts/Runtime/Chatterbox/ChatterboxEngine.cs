using UnityEngine;
using UnityEngine.Serialization;

namespace KitsuMate.Onnx.Tts.Chatterbox
{
    [CreateAssetMenu(fileName = "ChatterboxEngine", menuName = "KitsuMate/ONNX/TTS/Chatterbox Engine")]
    public sealed class ChatterboxEngine : TtsEngine
    {
        [FormerlySerializedAs("_modelSet"), SerializeField] private ChatterboxModelSet modelSet;
        [FormerlySerializedAs("_voiceCacheCapacity"), SerializeField] private int voiceCacheCapacity = 4;
        [SerializeField] private bool verboseLogging;
        public override ModelSet ModelSet => modelSet;
        public override int OutputSampleRate => ChatterboxConstants.SampleRate;
        public override bool IsMultilingual => modelSet != null && modelSet.Capabilities.Contains("multilingual");
        public override string[] SupportedLanguages => IsMultilingual ? new System.Collections.Generic.List<string>(ChatterboxConstants.SupportedLanguages.Keys).ToArray() : System.Array.Empty<string>();
        protected override InferenceEngineRuntime<TtsRequest, TtsResult> CreateRuntime() => new ChatterboxEngineRuntime(modelSet, voiceCacheCapacity, verboseLogging);
    }
}

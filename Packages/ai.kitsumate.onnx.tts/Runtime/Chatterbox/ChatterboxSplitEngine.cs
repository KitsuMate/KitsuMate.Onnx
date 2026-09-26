using UnityEngine;

namespace KitsuMate.Onnx.Tts.Chatterbox
{
    [CreateAssetMenu(fileName = "ChatterboxSplitEngine", menuName = "KitsuMate/ONNX/TTS/Chatterbox Split Engine")]
    public sealed class ChatterboxSplitEngine : TtsEngine
    {
        [SerializeField] private ChatterboxSplitModelSet modelSet;
        [SerializeField, Tooltip("Number of encoded reference voices to keep (0 = encode on every request)")]
        private int voiceCacheCapacity = 4;
        [SerializeField] private bool verboseLogging;

        public override ModelSet ModelSet => modelSet;
        public override int OutputSampleRate => ChatterboxConstants.SampleRate;
        public override bool IsMultilingual => modelSet != null && modelSet.Capabilities.Contains("multilingual");
        public override string[] SupportedLanguages => IsMultilingual
            ? new System.Collections.Generic.List<string>(ChatterboxConstants.SupportedLanguages.Keys).ToArray()
            : System.Array.Empty<string>();
        protected override InferenceEngineRuntime<TtsRequest, TtsResult> CreateRuntime(ModelSet resolvedModelSet) =>
            new ChatterboxSplitEngineRuntime((ChatterboxSplitModelSet)resolvedModelSet, voiceCacheCapacity, verboseLogging);
    }
}

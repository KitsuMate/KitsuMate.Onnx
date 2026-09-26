using UnityEngine;

namespace KitsuMate.Onnx.Tts.Chatterbox
{
    [CreateAssetMenu(fileName = "ChatterboxEngine", menuName = "KitsuMate/ONNX/TTS/Chatterbox Engine")]
    public sealed class ChatterboxEngine : TtsEngine
    {
        [SerializeField] private ChatterboxModelSet modelSet;
        [SerializeField] private int voiceCacheCapacity = 4;
        [SerializeField, Min(0), Tooltip("ONNX Runtime threads per graph (0 = up to 4). Phones with two fast cores run fastest with 2.")]
        private int inferenceThreads;
        [SerializeField] private bool verboseLogging;
        public override ModelSet ModelSet => modelSet;
        public override int OutputSampleRate => ChatterboxConstants.SampleRate;
        public override bool IsMultilingual => modelSet != null && modelSet.Capabilities.Contains("multilingual");
        public override string[] SupportedLanguages => IsMultilingual ? new System.Collections.Generic.List<string>(ChatterboxConstants.SupportedLanguages.Keys).ToArray() : System.Array.Empty<string>();
        protected override InferenceEngineRuntime<TtsRequest, TtsResult> CreateRuntime(ModelSet resolvedModelSet) => new ChatterboxEngineRuntime((ChatterboxModelSet)resolvedModelSet, voiceCacheCapacity, verboseLogging, inferenceThreads);
    }
}

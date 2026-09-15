using UnityEngine;

namespace KitsuMate.Onnx.Tts.NeuTts
{
    [CreateAssetMenu(fileName = "NeuTtsEngine", menuName = "KitsuMate/ONNX/TTS/NeuTTS Engine")]
    public sealed class NeuTtsEngine : TtsEngine
    {
        [SerializeField] private NeuTtsModelSet modelSet;
        [SerializeField] private NeuTtsGenerationConfig generation = new();
        [SerializeField] private bool verboseLogging;
        public NeuTtsGenerationConfig Generation => generation;
        public override ModelSet ModelSet => modelSet;
        public override int OutputSampleRate => 24000;
        public override bool IsMultilingual => false;
        public override string[] SupportedLanguages => System.Array.Empty<string>();
        protected override InferenceEngineRuntime<TtsRequest, TtsResult> CreateRuntime(ModelSet resolvedModelSet) =>
            new NeuTtsEngineRuntime((NeuTtsModelSet)resolvedModelSet, generation, verboseLogging);
    }
}

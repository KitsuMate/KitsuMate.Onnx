using UnityEngine;

namespace KitsuMate.Onnx.Tts.OmniVoice
{
    [CreateAssetMenu(fileName = "OmniVoiceEngine", menuName = "KitsuMate/ONNX/TTS/OmniVoice Engine")]
    public sealed class OmniVoiceEngine : TtsEngine
    {
        [SerializeField] private OmniVoiceModelSet modelSet;
        [SerializeField] private OmniVoiceGenerationConfig generation = new();
        [SerializeField] private bool verboseLogging;

        public override ModelSet ModelSet => modelSet;
        public override int OutputSampleRate => OmniVoiceConstants.SampleRate;
        public override bool IsMultilingual => true;
        public override string[] SupportedLanguages => System.Array.Empty<string>();
        public OmniVoiceGenerationConfig Generation => generation;
        protected override InferenceEngineRuntime<TtsRequest, TtsResult> CreateRuntime() =>
            new OmniVoiceEngineRuntime(modelSet, generation, verboseLogging);
    }
}

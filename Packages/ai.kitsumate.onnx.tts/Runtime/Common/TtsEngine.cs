using UnityEngine;
using KitsuMate.Onnx;

namespace KitsuMate.Onnx.Tts
{
    /// <summary>
    /// Base class for TTS (Text-to-Speech) engines.
    /// Provides a common interface for Chatterbox, Kokoro, and other TTS implementations.
    /// </summary>
    public abstract class TtsEngine : InferenceEngine<TtsRequest, TtsResult>
    {
        /// <summary>
        /// The sample rate of audio produced by this engine.
        /// </summary>
        public abstract int OutputSampleRate { get; }

        /// <summary>
        /// Whether this engine supports multilingual synthesis.
        /// Determined at runtime from the loaded model/tokenizer.
        /// </summary>
        public abstract bool IsMultilingual { get; }

        /// <summary>
        /// Supported language codes, if multilingual. Empty for English-only models.
        /// </summary>
        public abstract string[] SupportedLanguages { get; }
    }

    public abstract class TtsEngineRuntime : ThreadedInferenceEngineRuntime<TtsRequest, TtsResult>
    {
        public abstract int OutputSampleRate { get; }
        public abstract bool IsMultilingual { get; }
        public abstract string[] SupportedLanguages { get; }
    }
}

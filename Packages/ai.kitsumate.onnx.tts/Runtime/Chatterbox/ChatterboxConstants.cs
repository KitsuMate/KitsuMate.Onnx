using System;
using System.Collections.Generic;

namespace KitsuMate.Onnx.Tts.Chatterbox
{
    /// <summary>
    /// Constants for Chatterbox TTS models (both English and Multilingual variants).
    /// </summary>
    public static class ChatterboxConstants
    {
        /// <summary>Output audio sample rate in Hz.</summary>
        public const int SampleRate = 24000;

        /// <summary>Token ID for [EXAGGERATION] — first token in the template sequence.</summary>
        public const int ExaggerationToken = 6563;

        /// <summary>Token ID for [START] — second token in the template (start-of-text).</summary>
        public const int StartTextToken = 255;

        /// <summary>Token ID for [STOP] — marks end-of-text in the template.</summary>
        public const int StopTextToken = 0;

        /// <summary>Token ID for [START_SPEECH] — marks the start of speech generation.</summary>
        public const int StartSpeechToken = 6561;

        /// <summary>Token ID for [STOP_SPEECH] — marks the end of speech generation.</summary>
        public const int StopSpeechToken = 6562;

        /// <summary>Number of hidden layers in the Llama backbone.</summary>
        public const int NumHiddenLayers = 30;

        /// <summary>Number of key-value heads per layer.</summary>
        public const int NumKeyValueHeads = 16;

        /// <summary>Dimension of each attention head.</summary>
        public const int HeadDim = 64;

        /// <summary>Default exaggeration level for emotion control.</summary>
        public const float DefaultExaggeration = 0.5f;

        /// <summary>Default repetition penalty during generation.</summary>
        public const float DefaultRepetitionPenalty = 1.2f;

        /// <summary>Default maximum tokens to generate.</summary>
        public const int DefaultMaxNewTokens = 256;

        /// <summary>
        /// Supported languages for the multilingual Chatterbox model.
        /// Maps language codes to display names.
        /// </summary>
        public static readonly IReadOnlyDictionary<string, string> SupportedLanguages =
            new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
            {
                { "ar", "Arabic" },
                { "da", "Danish" },
                { "de", "German" },
                { "el", "Greek" },
                { "en", "English" },
                { "es", "Spanish" },
                { "fi", "Finnish" },
                { "fr", "French" },
                { "he", "Hebrew" },
                { "hi", "Hindi" },
                { "it", "Italian" },
                { "ja", "Japanese" },
                { "ko", "Korean" },
                { "ms", "Malay" },
                { "nl", "Dutch" },
                { "no", "Norwegian" },
                { "pl", "Polish" },
                { "pt", "Portuguese" },
                { "ru", "Russian" },
                { "sv", "Swedish" },
                { "sw", "Swahili" },
                { "tr", "Turkish" },
                { "zh", "Chinese" },
            };
    }
}

using System;
using UnityEngine;

namespace KitsuMate.Onnx.Tts
{
    /// <summary>
    /// Input request for text-to-speech synthesis.
    /// </summary>
    [Serializable]
    public class TtsRequest
    {
        /// <summary>Text to synthesize into speech.</summary>
        public string Text;

        /// <summary>
        /// Optional voice reference audio for voice cloning.
        /// If null, the engine's default voice will be used.
        /// </summary>
        public AudioClip VoiceReference;

        /// <summary>
        /// Language code (e.g. "en", "fr", "ja"). Only used with multilingual models.
        /// If null or empty, no language token is prepended.
        /// </summary>
        public string LanguageId;

        /// <summary>
        /// Emotion exaggeration level. Higher values produce more expressive speech.
        /// Default: 0.5. Range: 0.0 - 1.0+.
        /// </summary>
        public float Exaggeration = 0.5f;

        /// <summary>
        /// Maximum number of speech tokens to generate.
        /// Higher values allow longer utterances but take more time.
        /// </summary>
        public int MaxNewTokens = 256;

        /// <summary>
        /// Repetition penalty applied during generation.
        /// Higher values reduce repetitive patterns. Default: 1.2.
        /// </summary>
        public float RepetitionPenalty = 1.2f;

        public TtsRequest() { }

        public TtsRequest(string text, string languageId = null)
        {
            Text = text;
            LanguageId = languageId;
        }
    }
}

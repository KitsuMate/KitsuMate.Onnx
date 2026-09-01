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

        /// <summary>Transcript matching <see cref="VoiceReference"/>. Required by OmniVoice cloning.</summary>
        public string VoiceReferenceText;

        /// <summary>Optional voice-design instruction, for example "female, young adult, british accent".</summary>
        public string VoiceInstruction;

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

        /// <summary>Speaking-rate multiplier. Values above one are faster.</summary>
        public float Speed = 1f;

        /// <summary>Requested duration in seconds. Zero lets the engine estimate it.</summary>
        public float DurationSeconds;

        /// <summary>Optional OmniVoice decoding overrides. Null uses the engine defaults.</summary>
        public OmniVoice.OmniVoiceGenerationConfig OmniVoice;

        public TtsRequest() { }

        public TtsRequest(string text, string languageId = null)
        {
            Text = text;
            LanguageId = languageId;
        }
    }
}

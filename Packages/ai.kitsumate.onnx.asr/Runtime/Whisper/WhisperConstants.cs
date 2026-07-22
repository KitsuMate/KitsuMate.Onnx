using System;
using System.Collections.Generic;

namespace KitsuMate.Onnx.Asr.Whisper
{
    /// <summary>
    /// Configuration and constants for Whisper ASR models.
    /// </summary>
    public static class WhisperConstants
    {
        // Audio processing
        public const int MelBins = 80;
        public const int MaxAudioLengthSeconds = 30;
        public const int SampleRate = 16000;
        public const int MaxSamples = MaxAudioLengthSeconds * SampleRate; // 480000
        public const int MelTimeSteps = 3000;
        
        // Token generation
        public const int MaxTokens = 224;
        public const int VocabularySize = 51865;
        
        // Special tokens
        public const int EndOfText = 50257;
        public const int StartOfTranscript = 50258;
        public const int Transcribe = 50359;
        public const int Translate = 50358;
        public const int NoTimestamps = 50363;
        public const int StartTime = 50364;
        
        // Language tokens
        private static readonly Dictionary<string, int> LanguageTokens = new(StringComparer.OrdinalIgnoreCase)
        {
            { "en", 50259 }, { "zh", 50260 }, { "de", 50261 }, { "es", 50262 }, { "ru", 50263 },
            { "ko", 50264 }, { "fr", 50265 }, { "ja", 50266 }, { "pt", 50267 }, { "tr", 50268 },
            { "pl", 50269 }, { "ca", 50270 }, { "nl", 50271 }, { "ar", 50272 }, { "sv", 50273 },
            { "it", 50274 }, { "id", 50275 }, { "hi", 50276 }, { "fi", 50277 }, { "vi", 50278 },
            { "iw", 50279 }, { "uk", 50280 }, { "el", 50281 }, { "ms", 50282 }, { "cs", 50283 },
            { "ro", 50284 }, { "da", 50285 }, { "hu", 50286 }, { "ta", 50287 }, { "no", 50288 },
            { "th", 50289 }, { "ur", 50290 }, { "hr", 50291 }, { "bg", 50292 }, { "lt", 50293 },
            { "la", 50294 }, { "mi", 50295 }, { "ml", 50296 }, { "cy", 50297 }, { "sk", 50298 },
            { "te", 50299 }, { "fa", 50300 }, { "lv", 50301 }, { "bn", 50302 }, { "sr", 50303 },
            { "az", 50304 }, { "sl", 50305 }, { "kn", 50306 }, { "et", 50307 }, { "mk", 50308 },
            { "br", 50309 }, { "eu", 50310 }, { "is", 50311 }, { "hy", 50312 }, { "ne", 50313 },
            { "mn", 50314 }, { "bs", 50315 }, { "kk", 50316 }, { "sq", 50317 }, { "sw", 50318 },
            { "gl", 50319 }, { "mr", 50320 }, { "pa", 50321 }, { "si", 50322 }, { "km", 50323 },
            { "sn", 50324 }, { "yo", 50325 }, { "so", 50326 }, { "af", 50327 }, { "oc", 50328 },
            { "ka", 50329 }, { "be", 50330 }, { "tg", 50331 }, { "sd", 50332 }, { "gu", 50333 },
            { "am", 50334 }, { "yi", 50335 }, { "lo", 50336 }, { "uz", 50337 }, { "fo", 50338 },
            { "ht", 50339 }, { "ps", 50340 }, { "tk", 50341 }, { "nn", 50342 }, { "mt", 50343 },
            { "sa", 50344 }, { "lb", 50345 }, { "my", 50346 }, { "bo", 50347 }, { "tl", 50348 },
            { "mg", 50349 }, { "as", 50350 }, { "tt", 50351 }, { "haw", 50352 }, { "ln", 50353 },
            { "ha", 50354 }, { "ba", 50355 }, { "jw", 50356 }, { "su", 50357 }
        };
        
        /// <summary>All supported language codes.</summary>
        public static IReadOnlyCollection<string> SupportedLanguages => LanguageTokens.Keys;
        
        /// <summary>Get the token ID for a language code.</summary>
        public static int GetLanguageToken(string languageCode)
        {
            if (string.IsNullOrEmpty(languageCode))
                return LanguageTokens["en"];
                
            return LanguageTokens.TryGetValue(languageCode, out int token) 
                ? token 
                : LanguageTokens["en"];
        }
        
        /// <summary>Get the language code for a token ID.</summary>
        public static string GetLanguageCode(int tokenId)
        {
            foreach (var kvp in LanguageTokens)
            {
                if (kvp.Value == tokenId)
                    return kvp.Key;
            }
            return "en";
        }
        
        /// <summary>Check if a token is a language token.</summary>
        public static bool IsLanguageToken(int tokenId)
        {
            return tokenId >= 50259 && tokenId <= 50357;
        }
        
        /// <summary>Check if a token is a timestamp token.</summary>
        public static bool IsTimestampToken(int tokenId)
        {
            return tokenId >= StartTime && tokenId <= StartTime + 1500;
        }
        
        /// <summary>Convert timestamp token to seconds.</summary>
        public static float TokenToTimestamp(int tokenId)
        {
            return (tokenId - StartTime) * 0.02f;
        }
        
        /// <summary>Convert seconds to timestamp token.</summary>
        public static int TimestampToToken(float seconds)
        {
            return StartTime + (int)(seconds / 0.02f);
        }
    }
}

using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;
using UnityEngine;

namespace KitsuMate.Onnx.Tts.Chatterbox
{
    /// <summary>
    /// Preprocesses text for multilingual TTS by applying language-specific transformations.
    /// Matches the Python MTLTokenizer.encode() pipeline:
    ///   1. lowercase + NFKD normalization
    ///   2. Language-specific processing (Korean Jamo, Chinese Cangjie, etc.)
    /// The language token prefix and [SPACE] replacement are handled externally.
    /// </summary>
    public class LanguagePreprocessor
    {
        private readonly Dictionary<char, string> _cangjieMapping;

        public LanguagePreprocessor(string cangjieJson = null)
        {
            if (!string.IsNullOrEmpty(cangjieJson))
                _cangjieMapping = ParseCangjieMapping(cangjieJson);
        }

        /// <summary>
        /// Apply multilingual preprocessing: lowercase, NFKD normalize, then
        /// language-specific transformations.
        /// </summary>
        public string Process(string text, string languageId)
        {
            if (string.IsNullOrEmpty(text) || string.IsNullOrEmpty(languageId))
                return text;

            // Step 1: lowercase + NFKD normalize (matching Python preprocess_text)
            text = text.ToLowerInvariant();
            text = text.Normalize(NormalizationForm.FormKD);

            // Step 2: language-specific processing
            return languageId.ToLowerInvariant() switch
            {
                "ko" => DecomposeKoreanJamo(text),
                "zh" => ProcessChinese(text),
                "ja" => ProcessJapanese(text),
                "he" => ProcessHebrew(text),
                "ru" => ProcessRussian(text),
                _ => text
            };
        }

        #region Korean Jamo Decomposition

        // Unicode ranges for Korean
        private const int SyllableBase = 0xAC00;
        private const int SyllableEnd = 0xD7A3;
        private const int LeadCount = 19;
        private const int VowelCount = 21;
        private const int TailCount = 28;

        // Jamo character tables
        private static readonly char[] LeadJamo =
        {
            'ᄀ', 'ᄁ', 'ᄂ', 'ᄃ', 'ᄄ', 'ᄅ', 'ᄆ', 'ᄇ', 'ᄈ', 'ᄉ',
            'ᄊ', 'ᄋ', 'ᄌ', 'ᄍ', 'ᄎ', 'ᄏ', 'ᄐ', 'ᄑ', 'ᄒ'
        };

        private static readonly char[] VowelJamo =
        {
            'ᅡ', 'ᅢ', 'ᅣ', 'ᅤ', 'ᅥ', 'ᅦ', 'ᅧ', 'ᅨ', 'ᅩ', 'ᅪ',
            'ᅫ', 'ᅬ', 'ᅭ', 'ᅮ', 'ᅯ', 'ᅰ', 'ᅱ', 'ᅲ', 'ᅳ', 'ᅴ',
            'ᅵ'
        };

        private static readonly char[] TailJamo =
        {
            '\0', // no tail
            'ᆨ', 'ᆩ', 'ᆪ', 'ᆫ', 'ᆬ', 'ᆭ', 'ᆮ', 'ᆯ', 'ᆰ', 'ᆱ',
            'ᆲ', 'ᆳ', 'ᆴ', 'ᆵ', 'ᆶ', 'ᆷ', 'ᆸ', 'ᆹ', 'ᆺ', 'ᆻ',
            'ᆼ', 'ᆽ', 'ᆾ', 'ᆿ', 'ᇀ', 'ᇁ', 'ᇂ'
        };

        /// <summary>
        /// Decompose Korean syllables into constituent Jamo characters.
        /// E.g., '한' (U+D55C) → 'ᄒ' + 'ᅡ' + 'ᆫ'
        /// Non-Korean characters pass through unchanged.
        /// </summary>
        private static string DecomposeKoreanJamo(string text)
        {
            var sb = new StringBuilder(text.Length * 3);
            foreach (char c in text)
            {
                int code = c;
                if (code >= SyllableBase && code <= SyllableEnd)
                {
                    int offset = code - SyllableBase;
                    int lead = offset / (VowelCount * TailCount);
                    int vowel = (offset % (VowelCount * TailCount)) / TailCount;
                    int tail = offset % TailCount;

                    sb.Append(LeadJamo[lead]);
                    sb.Append(VowelJamo[vowel]);
                    if (tail > 0)
                        sb.Append(TailJamo[tail]);
                }
                else
                {
                    sb.Append(c);
                }
            }
            return sb.ToString();
        }

        #endregion

        #region Chinese (Cangjie / character-level fallback)

        /// <summary>
        /// Process Chinese text using Cangjie encoding.
        /// Each CJK character is converted to [cj_X][cj_Y][cj_.] tokens.
        /// Non-CJK characters pass through unchanged.
        /// </summary>
        private string ProcessChinese(string text)
        {
            if (_cangjieMapping == null)
            {
                Debug.LogWarning("[LanguagePreprocessor] Cangjie mapping not loaded. Chinese text may not tokenize correctly.");
                return text;
            }

            return ApplyCangjieDecomposition(text);
        }

        /// <summary>
        /// Apply Cangjie decomposition matching the Python ChineseCangjieConverter.__call__():
        /// For each CJK character ("Lo" Unicode category), look up the cangjie code,
        /// then produce [cj_X] for each letter in the code, followed by [cj_.] separator.
        /// </summary>
        private string ApplyCangjieDecomposition(string text)
        {
            var sb = new StringBuilder(text.Length * 8);
            foreach (char c in text)
            {
                if (char.GetUnicodeCategory(c) == UnicodeCategory.OtherLetter &&
                    _cangjieMapping.TryGetValue(c, out var code))
                {
                    foreach (char cjChar in code)
                        sb.Append($"[cj_{cjChar}]");
                    sb.Append("[cj_.]");
                }
                else
                {
                    sb.Append(c);
                }
            }
            return sb.ToString();
        }

        private static Dictionary<char, string> ParseCangjieMapping(string json)
        {
            // Simple JSON object parser for { "char": "decomposition", ... }
            var mapping = new Dictionary<char, string>();
            try
            {
                // Use Unity's built-in JSON utility or simple parsing
                // The Cangjie mapping is a flat { "漢": "abc", ... } object
                int i = json.IndexOf('{');
                if (i < 0) return mapping;
                i++;

                while (i < json.Length)
                {
                    // Find next key
                    int keyStart = json.IndexOf('"', i);
                    if (keyStart < 0) break;
                    int keyEnd = json.IndexOf('"', keyStart + 1);
                    if (keyEnd < 0) break;

                    string key = json.Substring(keyStart + 1, keyEnd - keyStart - 1);

                    // Find colon then value
                    int colon = json.IndexOf(':', keyEnd + 1);
                    if (colon < 0) break;
                    int valStart = json.IndexOf('"', colon + 1);
                    if (valStart < 0) break;
                    int valEnd = json.IndexOf('"', valStart + 1);
                    if (valEnd < 0) break;

                    string value = json.Substring(valStart + 1, valEnd - valStart - 1);

                    if (key.Length == 1)
                        mapping[key[0]] = value;

                    i = valEnd + 1;
                    // Skip comma or closing brace
                    while (i < json.Length && (json[i] == ',' || json[i] == ' ' || json[i] == '\n' || json[i] == '\r'))
                        i++;
                    if (i < json.Length && json[i] == '}')
                        break;
                }
            }
            catch (Exception e)
            {
                Debug.LogWarning($"[LanguagePreprocessor] Failed to parse Cangjie mapping: {e.Message}");
            }
            return mapping;
        }

        #endregion

        #region Japanese / Hebrew / Russian (TODO stubs)

        private static string ProcessJapanese(string text)
        {
            // Python uses pykakasi for kanji→hiragana conversion + NFKD.
            // TODO: Implement kanji→hiragana conversion when a C# pykakasi equivalent is available.
            Debug.LogWarning("[LanguagePreprocessor] Japanese preprocessing (kanji→hiragana) not yet implemented. Using pass-through.");
            return text;
        }

        private static string ProcessHebrew(string text)
        {
            // Python uses dicta_onnx for Hebrew diacritization.
            // TODO: Implement when a C# dicta equivalent or ONNX model is available.
            Debug.LogWarning("[LanguagePreprocessor] Hebrew diacritization not yet implemented. Using pass-through.");
            return text;
        }

        private static string ProcessRussian(string text)
        {
            // Python uses russian_text_stresser for stress mark insertion.
            // TODO: Implement when a C# equivalent is available.
            Debug.LogWarning("[LanguagePreprocessor] Russian stress labeling not yet implemented. Using pass-through.");
            return text;
        }

        #endregion
    }
}

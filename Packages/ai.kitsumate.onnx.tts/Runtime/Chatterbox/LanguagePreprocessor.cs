using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using UnityEngine;

namespace KitsuMate.Onnx.Tts.Chatterbox
{
    /// <summary>
    /// Applies the language-specific text transformations used by multilingual Chatterbox.
    /// Large language resources are supplied as optional model companion files.
    /// </summary>
    public sealed class LanguagePreprocessor
    {
        private const char CombiningAcute = '\u0301';

        private readonly Dictionary<int, string> cangjieMapping;
        private readonly Dictionary<string, string> japaneseReadings;
        private readonly Dictionary<string, int> russianStressPositions;
        private readonly HashSet<string> chineseWords;
        private readonly int longestJapaneseEntry;
        private readonly int longestChineseWord;

        public LanguagePreprocessor(string cangjieJson = null, string japaneseReadingData = null,
            string russianStressData = null, string chineseWordData = null)
        {
            if (!string.IsNullOrWhiteSpace(cangjieJson))
                cangjieMapping = ParseCangjieMapping(cangjieJson);
            if (!string.IsNullOrWhiteSpace(japaneseReadingData))
                japaneseReadings = ParseJapaneseReadings(japaneseReadingData, out longestJapaneseEntry);
            if (!string.IsNullOrWhiteSpace(russianStressData))
                russianStressPositions = ParseRussianStressPositions(russianStressData);
            if (!string.IsNullOrWhiteSpace(chineseWordData))
                chineseWords = ParseChineseWords(chineseWordData, out longestChineseWord);
        }

        /// <summary>Lowercases and normalizes text before applying the legacy multilingual transforms.</summary>
        public string Process(string text, string languageId)
        {
            if (string.IsNullOrEmpty(text) || string.IsNullOrEmpty(languageId)) return text;
            text = text.ToLowerInvariant().Normalize(NormalizationForm.FormKD);
            return ApplyLanguageProcessing(text, languageId.ToLowerInvariant());
        }

        /// <summary>V3 preserves case and applies language processing before its final NFKD normalization.</summary>
        public string ProcessV3(string text, string languageId)
        {
            if (string.IsNullOrEmpty(text)) return text;
            text = ApplyLanguageProcessing(text, languageId?.ToLowerInvariant());
            return text.Normalize(NormalizationForm.FormKD);
        }

        private string ApplyLanguageProcessing(string text, string languageId)
        {
            return languageId switch
            {
                "ko" => DecomposeKoreanJamo(text),
                "zh" => ProcessChinese(text),
                "ja" => ProcessJapanese(text),
                "he" => ProcessHebrew(text),
                "ru" => ProcessRussian(text),
                _ => text
            };
        }

        #region Korean

        private const int SyllableBase = 0xAC00;
        private const int SyllableEnd = 0xD7A3;
        private const int VowelCount = 21;
        private const int TailCount = 28;

        private static readonly char[] LeadJamo =
        {
            'ᄀ', 'ᄁ', 'ᄂ', 'ᄃ', 'ᄄ', 'ᄅ', 'ᄆ', 'ᄇ', 'ᄈ', 'ᄉ',
            'ᄊ', 'ᄋ', 'ᄌ', 'ᄍ', 'ᄎ', 'ᄏ', 'ᄐ', 'ᄑ', 'ᄒ'
        };

        private static readonly char[] VowelJamo =
        {
            'ᅡ', 'ᅢ', 'ᅣ', 'ᅤ', 'ᅥ', 'ᅦ', 'ᅧ', 'ᅨ', 'ᅩ', 'ᅪ',
            'ᅫ', 'ᅬ', 'ᅭ', 'ᅮ', 'ᅯ', 'ᅰ', 'ᅱ', 'ᅲ', 'ᅳ', 'ᅴ', 'ᅵ'
        };

        private static readonly char[] TailJamo =
        {
            '\0', 'ᆨ', 'ᆩ', 'ᆪ', 'ᆫ', 'ᆬ', 'ᆭ', 'ᆮ', 'ᆯ', 'ᆰ', 'ᆱ',
            'ᆲ', 'ᆳ', 'ᆴ', 'ᆵ', 'ᆶ', 'ᆷ', 'ᆸ', 'ᆹ', 'ᆺ', 'ᆻ', 'ᆼ',
            'ᆽ', 'ᆾ', 'ᆿ', 'ᇀ', 'ᇁ', 'ᇂ'
        };

        private static string DecomposeKoreanJamo(string text)
        {
            var result = new StringBuilder(text.Length * 3);
            foreach (char value in text)
            {
                int codePoint = value;
                if (codePoint < SyllableBase || codePoint > SyllableEnd)
                {
                    result.Append(value);
                    continue;
                }

                int offset = codePoint - SyllableBase;
                int lead = offset / (VowelCount * TailCount);
                int vowel = offset % (VowelCount * TailCount) / TailCount;
                int tail = offset % TailCount;
                result.Append(LeadJamo[lead]);
                result.Append(VowelJamo[vowel]);
                if (tail > 0) result.Append(TailJamo[tail]);
            }
            return result.ToString();
        }

        #endregion

        #region Chinese

        private string ProcessChinese(string text)
        {
            if (cangjieMapping == null)
            {
                Debug.LogWarning("[LanguagePreprocessor] Cangjie mapping not loaded. Chinese text may not tokenize correctly.");
                return text;
            }

            return ApplyCangjieDecomposition(SegmentChinese(text));
        }

        private string SegmentChinese(string text)
        {
            if (chineseWords == null || chineseWords.Count == 0) return text;

            var result = new StringBuilder(text.Length + 8);
            var run = new List<string>();
            for (int index = 0; index < text.Length;)
            {
                int width = char.IsSurrogatePair(text, index) ? 2 : 1;
                int codePoint = char.ConvertToUtf32(text, index);
                if (IsHan(codePoint))
                {
                    run.Add(text.Substring(index, width));
                }
                else
                {
                    AppendSegmentedRun(result, run);
                    result.Append(text, index, width);
                }
                index += width;
            }
            AppendSegmentedRun(result, run);
            return result.ToString();
        }

        private void AppendSegmentedRun(StringBuilder result, List<string> run)
        {
            if (run.Count == 0) return;

            bool first = true;
            for (int start = 0; start < run.Count;)
            {
                int matchLength = LongestChineseMatch(run, start);
                if (!first) result.Append(' ');
                for (int offset = 0; offset < matchLength; offset++) result.Append(run[start + offset]);
                first = false;
                start += matchLength;
            }
            run.Clear();
        }

        private int LongestChineseMatch(IReadOnlyList<string> run, int start)
        {
            int limit = Math.Min(longestChineseWord, run.Count - start);
            int longest = 1;
            var candidate = new StringBuilder(limit * 2);
            for (int length = 1; length <= limit; length++)
            {
                candidate.Append(run[start + length - 1]);
                if (length > 1 && chineseWords.Contains(candidate.ToString())) longest = length;
            }
            return longest;
        }

        private string ApplyCangjieDecomposition(string text)
        {
            var result = new StringBuilder(text.Length * 8);
            for (int index = 0; index < text.Length;)
            {
                int width = char.IsSurrogatePair(text, index) ? 2 : 1;
                int codePoint = char.ConvertToUtf32(text, index);
                if (CharUnicodeInfo.GetUnicodeCategory(text, index) == UnicodeCategory.OtherLetter &&
                    cangjieMapping.TryGetValue(codePoint, out string code))
                {
                    foreach (char cangjieCharacter in code) result.Append("[cj_").Append(cangjieCharacter).Append(']');
                    result.Append("[cj_.]");
                }
                else
                {
                    result.Append(text, index, width);
                }
                index += width;
            }
            return result.ToString();
        }

        private static Dictionary<int, string> ParseCangjieMapping(string data)
        {
            var mapping = new Dictionary<int, string>();
            ParseStringMap(data, (key, value) =>
            {
                if (TryGetSingleCodePoint(key, out int codePoint) && !string.IsNullOrEmpty(value))
                    mapping[codePoint] = value;
            });
            return mapping;
        }

        private static HashSet<string> ParseChineseWords(string data, out int longestWord)
        {
            var words = new HashSet<string>(StringComparer.Ordinal);
            longestWord = 0;
            using var reader = new StringReader(data);
            while (reader.ReadLine() is { } line)
            {
                line = line.Trim();
                if (line.Length == 0 || line[0] == '#') continue;
                int separator = line.IndexOfAny(new[] { ' ', '\t' });
                string word = separator < 0 ? line : line.Substring(0, separator);
                int codePoints = CountCodePoints(word);
                if (codePoints < 2 || !ContainsHan(word)) continue;
                words.Add(word);
                longestWord = Math.Max(longestWord, codePoints);
            }
            return words;
        }

        #endregion

        #region Japanese

        private string ProcessJapanese(string text)
        {
            if (japaneseReadings == null || japaneseReadings.Count == 0) return text;
            text = text.Normalize(NormalizationForm.FormKD);

            var result = new StringBuilder(text.Length * 2);
            for (int index = 0; index < text.Length;)
            {
                int width = char.IsSurrogatePair(text, index) ? 2 : 1;
                int codePoint = char.ConvertToUtf32(text, index);
                if (IsHan(codePoint) && TryFindJapaneseReading(text, index, out string reading, out int matchedLength))
                {
                    if (reading.Length > 0 && (reading[0] == 'は' || reading[0] == 'へ')) result.Append(' ');
                    result.Append(reading);
                    index += matchedLength;
                }
                else
                {
                    result.Append(text, index, width);
                    index += width;
                }
            }
            return result.ToString();
        }

        private bool TryFindJapaneseReading(string text, int start, out string reading, out int matchedLength)
        {
            int limit = Math.Min(longestJapaneseEntry, text.Length - start);
            for (int length = limit; length > 0; length--)
            {
                string candidate = text.Substring(start, length);
                if (!japaneseReadings.TryGetValue(candidate, out reading)) continue;
                matchedLength = length;
                return true;
            }
            reading = null;
            matchedLength = 0;
            return false;
        }

        private static Dictionary<string, string> ParseJapaneseReadings(string data, out int longestEntry)
        {
            var readings = new Dictionary<string, string>(StringComparer.Ordinal);
            var ambiguous = new HashSet<string>(StringComparer.Ordinal);
            int maximumLength = 0;
            ParseStringMap(data, (surface, reading) =>
            {
                surface = surface.Normalize(NormalizationForm.FormKD);
                reading = KatakanaToHiragana(reading).Normalize(NormalizationForm.FormKD);
                if (surface.Length == 0 || reading.Length == 0 || !ContainsHan(surface) || ambiguous.Contains(surface))
                    return;
                if (readings.TryGetValue(surface, out string existing) && existing != reading)
                {
                    readings.Remove(surface);
                    ambiguous.Add(surface);
                    return;
                }
                readings[surface] = reading;
                maximumLength = Math.Max(maximumLength, surface.Length);
            });
            longestEntry = maximumLength;
            return readings;
        }

        private static string KatakanaToHiragana(string text)
        {
            var result = new StringBuilder(text.Length);
            foreach (char value in text)
                result.Append(value >= '\u30A1' && value <= '\u30F6' ? (char)(value - 0x60) : value);
            return result.ToString();
        }

        #endregion

        #region Russian

        private string ProcessRussian(string text)
        {
            if (russianStressPositions == null || russianStressPositions.Count == 0) return text;

            var result = new StringBuilder(text.Length + 8);
            for (int index = 0; index < text.Length;)
            {
                if (!IsCyrillicLetter(text[index]))
                {
                    result.Append(text[index++]);
                    continue;
                }

                int end = index + 1;
                while (end < text.Length && (IsCyrillicLetter(text[end]) || text[end] == CombiningAcute)) end++;
                string word = text.Substring(index, end - index);
                string key = NormalizeRussianKey(word);
                if (word.IndexOf(CombiningAcute) < 0 &&
                    russianStressPositions.TryGetValue(key, out int stressPosition) && stressPosition < word.Length)
                {
                    result.Append(word, 0, stressPosition + 1).Append(CombiningAcute);
                    result.Append(word, stressPosition + 1, word.Length - stressPosition - 1);
                }
                else
                {
                    result.Append(word);
                }
                index = end;
            }
            return result.ToString();
        }

        private static Dictionary<string, int> ParseRussianStressPositions(string data)
        {
            var positions = new Dictionary<string, int>(StringComparer.Ordinal);
            var ambiguous = new HashSet<string>(StringComparer.Ordinal);
            ParseStringMap(data, (word, value) =>
            {
                string key = NormalizeRussianKey(word);
                if (key.Length == 0 || ambiguous.Contains(key) || !TryParseStressPosition(key, value, out int position))
                    return;
                if (positions.TryGetValue(key, out int existing) && existing != position)
                {
                    positions.Remove(key);
                    ambiguous.Add(key);
                    return;
                }
                positions[key] = position;
            });
            return positions;
        }

        private static bool TryParseStressPosition(string word, string value, out int position)
        {
            if (int.TryParse(value, NumberStyles.None, CultureInfo.InvariantCulture, out position))
                return position >= 0 && position < word.Length && IsRussianVowel(word[position]);

            int acute = value.IndexOf(CombiningAcute);
            if (acute > 0)
            {
                string unstressed = value.Remove(acute, 1);
                position = acute - 1;
                if (NormalizeRussianKey(unstressed) == word && IsRussianVowel(word[position])) return true;
            }

            position = -1;
            return false;
        }

        private static string NormalizeRussianKey(string word)
        {
            return word.Replace(CombiningAcute.ToString(), string.Empty).ToLowerInvariant().Replace('ё', 'е');
        }

        private static bool IsCyrillicLetter(char value)
        {
            return (value >= '\u0400' && value <= '\u052F' || value >= '\uA640' && value <= '\uA69F') &&
                char.IsLetter(value);
        }

        private static bool IsRussianVowel(char value)
        {
            return "аеёиоуыэюя".IndexOf(char.ToLowerInvariant(value)) >= 0;
        }

        #endregion

        private static string ProcessHebrew(string text)
        {
            // Upstream uses an optional neural diacritizer. No equivalent portable model is bundled.
            return text;
        }

        private static void ParseStringMap(string data, Action<string, string> add)
        {
            int first = 0;
            while (first < data.Length && char.IsWhiteSpace(data[first])) first++;
            if (first < data.Length && data[first] == '{')
            {
                ParseJsonStringMap(data, first + 1, add);
                return;
            }

            using var reader = new StringReader(data);
            while (reader.ReadLine() is { } line)
            {
                line = line.Trim();
                if (line.Length == 0 || line[0] == '#') continue;
                int separator = line.IndexOf('\t');
                if (separator <= 0 || separator == line.Length - 1) continue;
                add(line.Substring(0, separator), line.Substring(separator + 1).Trim());
            }
        }

        private static void ParseJsonStringMap(string json, int index, Action<string, string> add)
        {
            try
            {
                while (index < json.Length)
                {
                    SkipJsonWhitespaceAndCommas(json, ref index);
                    if (index >= json.Length || json[index] == '}') return;
                    string key = ReadJsonString(json, ref index);
                    SkipJsonWhitespace(json, ref index);
                    if (index >= json.Length || json[index++] != ':') return;
                    SkipJsonWhitespace(json, ref index);
                    string value = json[index] == '"' ? ReadJsonString(json, ref index) : ReadJsonScalar(json, ref index);
                    add(key, value);
                }
            }
            catch (FormatException exception)
            {
                Debug.LogWarning($"[LanguagePreprocessor] Could not parse companion dictionary: {exception.Message}");
            }
        }

        private static string ReadJsonString(string json, ref int index)
        {
            if (index >= json.Length || json[index++] != '"') throw new FormatException("Expected a JSON string.");
            var result = new StringBuilder();
            while (index < json.Length)
            {
                char value = json[index++];
                if (value == '"') return result.ToString();
                if (value != '\\')
                {
                    result.Append(value);
                    continue;
                }
                if (index >= json.Length) break;
                char escape = json[index++];
                switch (escape)
                {
                    case '"': case '\\': case '/': result.Append(escape); break;
                    case 'b': result.Append('\b'); break;
                    case 'f': result.Append('\f'); break;
                    case 'n': result.Append('\n'); break;
                    case 'r': result.Append('\r'); break;
                    case 't': result.Append('\t'); break;
                    case 'u':
                        if (index + 4 > json.Length) throw new FormatException("Incomplete JSON Unicode escape.");
                        if (!ushort.TryParse(json.Substring(index, 4), NumberStyles.HexNumber,
                                CultureInfo.InvariantCulture, out ushort codeUnit))
                            throw new FormatException("Invalid JSON Unicode escape.");
                        result.Append((char)codeUnit);
                        index += 4;
                        break;
                    default: throw new FormatException($"Unsupported JSON escape: \\{escape}");
                }
            }
            throw new FormatException("Unterminated JSON string.");
        }

        private static string ReadJsonScalar(string json, ref int index)
        {
            int start = index;
            while (index < json.Length && json[index] != ',' && json[index] != '}') index++;
            return json.Substring(start, index - start).Trim();
        }

        private static void SkipJsonWhitespace(string json, ref int index)
        {
            while (index < json.Length && char.IsWhiteSpace(json[index])) index++;
        }

        private static void SkipJsonWhitespaceAndCommas(string json, ref int index)
        {
            while (index < json.Length && (char.IsWhiteSpace(json[index]) || json[index] == ',')) index++;
        }

        private static bool TryGetSingleCodePoint(string text, out int codePoint)
        {
            codePoint = 0;
            if (string.IsNullOrEmpty(text)) return false;
            int width = char.IsSurrogatePair(text, 0) ? 2 : 1;
            if (text.Length != width) return false;
            codePoint = char.ConvertToUtf32(text, 0);
            return true;
        }

        private static int CountCodePoints(string text)
        {
            int count = 0;
            for (int index = 0; index < text.Length; count++) index += char.IsSurrogatePair(text, index) ? 2 : 1;
            return count;
        }

        private static bool ContainsHan(string text)
        {
            for (int index = 0; index < text.Length;)
            {
                int width = char.IsSurrogatePair(text, index) ? 2 : 1;
                if (IsHan(char.ConvertToUtf32(text, index))) return true;
                index += width;
            }
            return false;
        }

        private static bool IsHan(int codePoint)
        {
            return codePoint >= 0x3400 && codePoint <= 0x4DBF ||
                codePoint >= 0x4E00 && codePoint <= 0x9FFF ||
                codePoint >= 0xF900 && codePoint <= 0xFAFF ||
                codePoint >= 0x20000 && codePoint <= 0x323AF;
        }
    }
}

using System.Collections.Generic;
using System.Text;
using UnityEngine;

namespace KitsuMate.Onnx.Asr
{
    /// <summary>
    /// Dynamic Time Warping (DTW) based word alignment.
    /// Provides optimal alignment between expected and transcribed word sequences.
    /// </summary>
    public class DTWWordAligner
    {
        private readonly bool _verboseLogging;

        public DTWWordAligner(bool verboseLogging = false)
        {
            _verboseLogging = verboseLogging;
        }

        /// <summary>
        /// Align transcribed words to expected text using Dynamic Time Warping.
        /// </summary>
        public TranscriptionResult AlignWords(
            string expectedText,
            List<TranscribedWord> transcribedWords,
            float totalDuration)
        {
            var result = new TranscriptionResult
            {
                Text = expectedText,
                Duration = totalDuration
            };

            string[] originalWords = ExtractOriginalWords(expectedText);
            string[] normalizedWords = NormalizeWords(expectedText);

            if (_verboseLogging)
                Debug.Log($"[DTW] {normalizedWords.Length} expected words, {transcribedWords.Count} transcribed words");

            if (normalizedWords.Length == 0)
                return result;

            if (transcribedWords.Count == 0)
                return FallbackUniformDistribution(originalWords, expectedText, totalDuration);

            if (TryDirectAlignment(normalizedWords, originalWords, transcribedWords, out var directTimestamps))
            {
                if (_verboseLogging)
                    Debug.Log("[DTW] Direct word match detected, skipping DTW");

                result.WordTimestamps = directTimestamps;
                return result;
            }

            var alignment = ComputeDTWAlignment(normalizedWords, transcribedWords);

            result.WordTimestamps = CreateTimestampsFromAlignment(
                originalWords, transcribedWords, alignment, totalDuration);

            if (_verboseLogging)
            {
                Debug.Log($"[DTW] Alignment complete: {result.WordTimestamps.Count} timestamps generated");
                LogAlignmentQuality(result.WordTimestamps);
            }

            return result;
        }

        #region Direct Alignment

        private bool TryDirectAlignment(
            string[] normalizedExpected,
            string[] originalExpected,
            List<TranscribedWord> transcribedWords,
            out List<WordTimestamp> timestamps)
        {
            timestamps = null;

            if (normalizedExpected == null || originalExpected == null)
                return false;
            if (normalizedExpected.Length == 0 || normalizedExpected.Length != transcribedWords.Count)
                return false;
            if (originalExpected.Length != normalizedExpected.Length)
                return false;

            var direct = new List<WordTimestamp>(normalizedExpected.Length);

            for (int i = 0; i < normalizedExpected.Length; i++)
            {
                string expectedNorm = normalizedExpected[i];
                string transcribedNorm = NormalizeWord(transcribedWords[i].Word);

                if (string.IsNullOrEmpty(expectedNorm) || string.IsNullOrEmpty(transcribedNorm))
                    return false;
                if (!expectedNorm.Equals(transcribedNorm))
                    return false;

                direct.Add(new WordTimestamp(
                    originalExpected[i],
                    transcribedWords[i].StartTime,
                    transcribedWords[i].EndTime,
                    1.0f));
            }

            timestamps = direct;
            return true;
        }

        #endregion

        #region DTW Core

        private List<AlignmentPair> ComputeDTWAlignment(
            string[] expectedWords,
            List<TranscribedWord> transcribedWords)
        {
            int n = expectedWords.Length;
            int m = transcribedWords.Count;

            float[,] dtw = new float[n + 1, m + 1];
            int[,] path = new int[n + 1, m + 1]; // 0=diagonal, 1=left, 2=up

            for (int i = 0; i <= n; i++)
                for (int j = 0; j <= m; j++)
                    dtw[i, j] = float.MaxValue;

            dtw[0, 0] = 0;

            for (int i = 1; i <= n; i++)
            {
                for (int j = 1; j <= m; j++)
                {
                    float matchCost = CalculateWordMatchCost(expectedWords[i - 1], transcribedWords[j - 1].Word);

                    float diagonalCost = dtw[i - 1, j - 1] + matchCost;
                    float leftCost = dtw[i, j - 1] + 0.5f;
                    float upCost = dtw[i - 1, j] + 1.0f;

                    float minCost = Mathf.Min(diagonalCost, Mathf.Min(leftCost, upCost));
                    dtw[i, j] = minCost;

                    if (minCost == diagonalCost) path[i, j] = 0;
                    else if (minCost == leftCost) path[i, j] = 1;
                    else path[i, j] = 2;
                }
            }

            // Backtrack
            var alignment = new List<AlignmentPair>();
            int ei = n, ti = m;

            while (ei > 0 || ti > 0)
            {
                if (ei > 0 && ti > 0 && path[ei, ti] == 0)
                {
                    alignment.Add(new AlignmentPair(
                        ei - 1, ti - 1,
                        CalculateWordMatchCost(expectedWords[ei - 1], transcribedWords[ti - 1].Word)));
                    ei--;
                    ti--;
                }
                else if (ti > 0 && (ei == 0 || path[ei, ti] == 1))
                {
                    ti--;
                }
                else if (ei > 0)
                {
                    alignment.Add(new AlignmentPair(ei - 1, -1, 1.0f));
                    ei--;
                }
            }

            alignment.Reverse();
            return alignment;
        }

        #endregion

        #region Timestamp Creation

        private List<WordTimestamp> CreateTimestampsFromAlignment(
            string[] expectedWords,
            List<TranscribedWord> transcribedWords,
            List<AlignmentPair> alignment,
            float totalDuration)
        {
            var timestamps = new List<WordTimestamp>();

            for (int i = 0; i < expectedWords.Length; i++)
            {
                var pair = alignment.Find(a => a.ExpectedIndex == i);

                if (pair != null && pair.TranscribedIndex >= 0)
                {
                    var tw = transcribedWords[pair.TranscribedIndex];
                    float confidence = ConvertCostToConfidence(pair.Cost);

                    string expectedNorm = NormalizeWord(expectedWords[i]);
                    string transcribedNorm = NormalizeWord(tw.Word);

                    bool haveNormalized = !string.IsNullOrEmpty(expectedNorm) &&
                                           !string.IsNullOrEmpty(transcribedNorm);

                    if (haveNormalized)
                    {
                        if (expectedNorm == transcribedNorm)
                        {
                            confidence = 1.0f;
                        }
                        else
                        {
                            float lexical = LevenshteinSimilarity(expectedNorm, transcribedNorm);
                            confidence = Mathf.Clamp01(Mathf.Max(confidence, lexical));
                        }
                    }

                    timestamps.Add(new WordTimestamp(expectedWords[i], tw.StartTime, tw.EndTime, confidence));
                }
                else
                {
                    timestamps.Add(InterpolateUnmatchedWord(
                        i, expectedWords, transcribedWords, alignment, totalDuration));
                }
            }

            return timestamps;
        }

        private static float ConvertCostToConfidence(float cost)
        {
            if (float.IsNaN(cost)) return 0f;
            return Mathf.Clamp01(1.0f - (cost * 0.5f));
        }

        private WordTimestamp InterpolateUnmatchedWord(
            int wordIndex,
            string[] expectedWords,
            List<TranscribedWord> transcribedWords,
            List<AlignmentPair> alignment,
            float totalDuration)
        {
            AlignmentPair prevAlignment = null;
            AlignmentPair nextAlignment = null;

            for (int i = 0; i < alignment.Count; i++)
            {
                if (alignment[i].TranscribedIndex < 0) continue;
                if (alignment[i].ExpectedIndex < wordIndex)
                    prevAlignment = alignment[i];
                if (alignment[i].ExpectedIndex > wordIndex && nextAlignment == null)
                {
                    nextAlignment = alignment[i];
                    break;
                }
            }

            float startTime, endTime;
            float confidence = 0.3f;

            if (prevAlignment != null && nextAlignment != null)
            {
                var prevTw = transcribedWords[prevAlignment.TranscribedIndex];
                var nextTw = transcribedWords[nextAlignment.TranscribedIndex];

                int wordsBetween = nextAlignment.ExpectedIndex - prevAlignment.ExpectedIndex;
                int positionInGap = wordIndex - prevAlignment.ExpectedIndex;

                float gapStart = prevTw.EndTime;
                float gapEnd = nextTw.StartTime;
                float gapDuration = gapEnd - gapStart;

                if (gapDuration > 0)
                {
                    float wordDuration = gapDuration / wordsBetween;
                    startTime = gapStart + ((positionInGap - 0.5f) * wordDuration);
                    endTime = startTime + wordDuration;
                }
                else
                {
                    startTime = prevTw.EndTime;
                    endTime = startTime + 0.15f;
                }
                confidence = 0.4f;
            }
            else if (prevAlignment != null)
            {
                var prevTw = transcribedWords[prevAlignment.TranscribedIndex];
                float avgDuration = CalculateAverageWordDuration(transcribedWords, 0.2f);
                int wordsAfter = wordIndex - prevAlignment.ExpectedIndex;
                startTime = prevTw.EndTime + ((wordsAfter - 1) * avgDuration);
                endTime = startTime + avgDuration;
            }
            else if (nextAlignment != null)
            {
                var nextTw = transcribedWords[nextAlignment.TranscribedIndex];
                float avgDuration = CalculateAverageWordDuration(transcribedWords, 0.2f);
                int wordsBefore = nextAlignment.ExpectedIndex - wordIndex;
                endTime = nextTw.StartTime - ((wordsBefore - 1) * avgDuration);
                startTime = endTime - avgDuration;
            }
            else
            {
                float avgDuration = totalDuration / expectedWords.Length;
                startTime = wordIndex * avgDuration;
                endTime = (wordIndex + 1) * avgDuration;
                confidence = 0.2f;
            }

            return new WordTimestamp(expectedWords[wordIndex], startTime, endTime, confidence);
        }

        #endregion

        #region Word Cost & Similarity

        private float CalculateWordMatchCost(string expected, string transcribed)
        {
            expected = NormalizeWord(expected);
            transcribed = NormalizeWord(transcribed);

            if (expected == transcribed) return 0f;
            if (string.IsNullOrEmpty(expected) || string.IsNullOrEmpty(transcribed)) return 2.0f;

            float levSimilarity = LevenshteinSimilarity(expected, transcribed);
            float phoneticSimilarity = PhoneticSimilarity(expected, transcribed);
            float lengthSimilarity = LengthSimilarity(expected, transcribed);

            float combinedSimilarity = (levSimilarity * 0.5f) +
                                       (phoneticSimilarity * 0.3f) +
                                       (lengthSimilarity * 0.2f);

            return (1.0f - combinedSimilarity) * 2.0f;
        }

        private static float LevenshteinSimilarity(string s1, string s2)
        {
            if (string.IsNullOrEmpty(s1) && string.IsNullOrEmpty(s2)) return 1.0f;
            if (string.IsNullOrEmpty(s1) || string.IsNullOrEmpty(s2)) return 0.0f;
            int maxLen = Mathf.Max(s1.Length, s2.Length);
            return 1.0f - ((float)LevenshteinDistance(s1, s2) / maxLen);
        }

        private static int LevenshteinDistance(string s1, string s2)
        {
            int[,] d = new int[s1.Length + 1, s2.Length + 1];

            for (int i = 0; i <= s1.Length; i++) d[i, 0] = i;
            for (int j = 0; j <= s2.Length; j++) d[0, j] = j;

            for (int j = 1; j <= s2.Length; j++)
            {
                for (int i = 1; i <= s1.Length; i++)
                {
                    int cost = (s1[i - 1] == s2[j - 1]) ? 0 : 1;
                    d[i, j] = Mathf.Min(
                        Mathf.Min(d[i - 1, j] + 1, d[i, j - 1] + 1),
                        d[i - 1, j - 1] + cost);
                }
            }

            return d[s1.Length, s2.Length];
        }

        private static float PhoneticSimilarity(string word1, string word2)
        {
            string p1 = GetPhoneticCode(word1);
            string p2 = GetPhoneticCode(word2);
            if (p1 == p2) return 1.0f;
            return LevenshteinSimilarity(p1, p2) * 0.9f;
        }

        private static string GetPhoneticCode(string word)
        {
            if (string.IsNullOrEmpty(word)) return "";
            word = word.ToLower();
            var code = new StringBuilder();
            if (word.Length > 0) code.Append(word[0]);

            char prevCode = GetPhoneticChar(word[0]);
            for (int i = 1; i < word.Length; i++)
            {
                char currentCode = GetPhoneticChar(word[i]);
                if (currentCode != '0' && currentCode != prevCode)
                {
                    code.Append(currentCode);
                    prevCode = currentCode;
                }
            }
            return code.ToString();
        }

        private static char GetPhoneticChar(char c)
        {
            return c switch
            {
                'b' or 'f' or 'p' or 'v' => '1',
                'c' or 'g' or 'j' or 'k' or 'q' or 's' or 'x' or 'z' => '2',
                'd' or 't' => '3',
                'l' => '4',
                'm' or 'n' => '5',
                'r' => '6',
                'w' or 'h' or 'y' => '7',
                _ => '0'
            };
        }

        private static float LengthSimilarity(string s1, string s2)
        {
            if (s1.Length == 0 && s2.Length == 0) return 1.0f;
            int maxLen = Mathf.Max(s1.Length, s2.Length);
            int minLen = Mathf.Min(s1.Length, s2.Length);
            return (float)minLen / maxLen;
        }

        #endregion

        #region Text Utilities

        private static string[] NormalizeWords(string text)
        {
            if (string.IsNullOrEmpty(text)) return System.Array.Empty<string>();
            string[] words = text.Split(new[] { ' ', '\t', '\n', '\r' }, System.StringSplitOptions.RemoveEmptyEntries);
            for (int i = 0; i < words.Length; i++)
                words[i] = NormalizeWord(words[i]);
            return words;
        }

        private static string[] ExtractOriginalWords(string text)
        {
            if (string.IsNullOrEmpty(text)) return System.Array.Empty<string>();
            return text.Split(new[] { ' ', '\t', '\n', '\r' }, System.StringSplitOptions.RemoveEmptyEntries);
        }

        private static string NormalizeWord(string word)
        {
            if (string.IsNullOrEmpty(word)) return "";
            word = word.ToLower().Trim();
            word = word.Trim('.', ',', '!', '?', ';', ':', '"', '\'', '-', '(', ')', '[', ']');
            return word;
        }

        private static float CalculateAverageWordDuration(List<TranscribedWord> words, float fallback)
        {
            if (words.Count == 0) return fallback;
            float total = 0f;
            foreach (var w in words)
                total += (w.EndTime - w.StartTime);
            return total / words.Count;
        }

        #endregion

        #region Helpers

        private TranscriptionResult FallbackUniformDistribution(string[] words, string text, float totalDuration)
        {
            var result = new TranscriptionResult
            {
                Text = text,
                Duration = totalDuration
            };

            float wordDuration = totalDuration / words.Length;
            var timestamps = new List<WordTimestamp>(words.Length);

            for (int i = 0; i < words.Length; i++)
            {
                timestamps.Add(new WordTimestamp(words[i], i * wordDuration, (i + 1) * wordDuration, 0.2f));
            }

            result.WordTimestamps = timestamps;
            return result;
        }

        private void LogAlignmentQuality(List<WordTimestamp> timestamps)
        {
            if (timestamps.Count == 0) return;

            int high = 0, medium = 0, low = 0;
            foreach (var ts in timestamps)
            {
                if (ts.Probability >= 0.7f) high++;
                else if (ts.Probability >= 0.4f) medium++;
                else low++;
            }

            Debug.Log($"[DTW] Alignment Quality: High={high}, Medium={medium}, Low={low}");
        }

        #endregion
    }

    /// <summary>
    /// A transcribed word with timing information used for DTW alignment.
    /// </summary>
    public class TranscribedWord
    {
        public string Word { get; set; }
        public float StartTime { get; set; }
        public float EndTime { get; set; }
        public int SegmentIndex { get; set; }

        public override string ToString() => $"[{StartTime:F3}s - {EndTime:F3}s] {Word}";
    }

    /// <summary>
    /// Represents a pair in the DTW alignment path.
    /// </summary>
    public class AlignmentPair
    {
        public int ExpectedIndex { get; }
        public int TranscribedIndex { get; }
        public float Cost { get; }

        public AlignmentPair(int expectedIndex, int transcribedIndex, float cost)
        {
            ExpectedIndex = expectedIndex;
            TranscribedIndex = transcribedIndex;
            Cost = cost;
        }
    }
}

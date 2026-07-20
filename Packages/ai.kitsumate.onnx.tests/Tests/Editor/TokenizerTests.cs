using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using KitsuMate.Tokenizers;
using KitsuMate.Tokenizers.Core;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using UnityEngine;

namespace KitsuMate.Onnx.Tests
{
    public class TokenizerTests
    {
        private const string GroundTruthDir = "Packages/ai.kitsumate.onnx.embeddings/Tests/Python";
        private const string TestDataDir = "Packages/ai.kitsumate.onnx.tests/Tests/Editor/TestData";

        private static string LoadTestVocab()
        {
            var path = Path.GetFullPath(Path.Combine(TestDataDir, "vocab.txt"));
            if (!File.Exists(path))
                return null;

            return File.ReadAllText(path);
        }

        private static string LoadTestTokenizerJsonPath()
        {
            var path = Path.GetFullPath(Path.Combine(TestDataDir, "tokenizer.json"));
            return File.Exists(path) ? path : null;
        }

        private static JObject LoadTokenizerGroundTruth(string modelTag)
        {
            var path = Path.GetFullPath(Path.Combine(GroundTruthDir, $"tokenizer_ground_truth_{modelTag}.json"));
            if (!File.Exists(path))
                return null;

            return JObject.Parse(File.ReadAllText(path));
        }

        [Test]
        public void MiniLM_L6_v2_TokenizerMatchesGroundTruth()
        {
            ValidateTokenizerAgainstGroundTruth("all-MiniLM-L6-v2");
        }

        private void ValidateTokenizerAgainstGroundTruth(string modelTag)
        {
            var tokenizerJsonPath = LoadTestTokenizerJsonPath();
            if (tokenizerJsonPath == null)
            {
                Assert.Ignore($"tokenizer.json not available at {TestDataDir}");
                return;
            }

            var groundTruth = LoadTokenizerGroundTruth(modelTag);
            if (groundTruth == null)
            {
                Assert.Ignore($"Ground truth not found for {modelTag}. Run generate_test_data.py first.");
                return;
            }

            var sentences = groundTruth["sentences"] as JArray;
            if (sentences == null)
            {
                Assert.Fail($"Ground truth for {modelTag} is missing 'sentences'.");
            }

            var tokenizer = Tokenizer.FromTokenizerJson(File.ReadAllBytes(tokenizerJsonPath));

            int passed = 0;
            foreach (var sentenceToken in sentences.Children<JObject>())
            {
                var text = sentenceToken.Value<string>("text") ?? string.Empty;
                var expectedIds = GetIntArrayOrEmpty(sentenceToken, "input_ids");
                var expectedMask = GetIntArrayOrEmpty(sentenceToken, "attention_mask");
                var expectedTypeIds = GetIntArrayOrEmpty(sentenceToken, "token_type_ids");

                var result = tokenizer.Encode(text);

                Assert.AreEqual(expectedIds.Length, result.Ids.Count,
                    $"[{modelTag}] Token count mismatch for: \"{text}\"");

                CollectionAssert.AreEqual(expectedIds, result.Ids,
                    $"[{modelTag}] input_ids mismatch for: \"{text}\"");

                CollectionAssert.AreEqual(expectedMask, result.AttentionMask,
                    $"[{modelTag}] attention_mask mismatch for: \"{text}\"");

                CollectionAssert.AreEqual(expectedTypeIds, result.TypeIds,
                    $"[{modelTag}] token_type_ids mismatch for: \"{text}\"");

                passed++;
            }

            Debug.Log($"[{modelTag}] Tokenizer ground truth: {passed} sentences passed");
        }

        private static int[] GetIntArrayOrEmpty(JObject element, string propertyName)
        {
            if (element[propertyName] is not JArray values)
            {
                return Array.Empty<int>();
            }

            return values.Values<int>().ToArray();
        }

        [Test]
        public void BertTokenizer_EncodeReturnsValidResult()
        {
            var vocabText = LoadTestVocab();
            if (vocabText == null) { Assert.Ignore("Test vocab not available"); return; }

            var tokenizer = Tokenizer.CreateWordPiece(Encoding.UTF8.GetBytes(vocabText));
            var result = tokenizer.Encode("Hello world");

            Assert.IsNotNull(result);
            Assert.IsNotNull(result.Ids);
            Assert.IsNotNull(result.AttentionMask);
            Assert.IsNotNull(result.TypeIds);
            Assert.Greater(result.Ids.Count, 0);

            Assert.AreEqual(101, result.Ids[0], "First token should be [CLS]");
            Assert.AreEqual(102, result.Ids[result.Ids.Count - 1], "Last token should be [SEP]");
        }

        [Test]
        public void BertTokenizer_EncodeWithoutSpecialTokens()
        {
            var vocabText = LoadTestVocab();
            if (vocabText == null) { Assert.Ignore("Test vocab not available"); return; }

            var tokenizer = Tokenizer.CreateWordPiece(Encoding.UTF8.GetBytes(vocabText));
            var withSpecial = tokenizer.Encode("Hello world", addSpecialTokens: true);
            var withoutSpecial = tokenizer.Encode("Hello world", addSpecialTokens: false);

            Assert.AreEqual(withSpecial.Ids.Count - 2, withoutSpecial.Ids.Count);
        }

        [Test]
        public void BertTokenizer_TokenTypeIdsAreZeros()
        {
            var vocabText = LoadTestVocab();
            if (vocabText == null) { Assert.Ignore("Test vocab not available"); return; }

            var tokenizer = Tokenizer.CreateWordPiece(Encoding.UTF8.GetBytes(vocabText));
            var result = tokenizer.Encode("The quick brown fox");

            foreach (var typeId in result.TypeIds)
                Assert.AreEqual(0, typeId, "Single sentence should have all zero token type IDs");
        }

        [Test]
        public void BertTokenizer_MaxLengthTruncates()
        {
            var vocabText = LoadTestVocab();
            if (vocabText == null) { Assert.Ignore("Test vocab not available"); return; }

            var tokenizer = Tokenizer.CreateWordPiece(Encoding.UTF8.GetBytes(vocabText));
            var result = tokenizer.Encode("The quick brown fox jumps over the lazy dog", maxTokenCount: 5);

            Assert.AreEqual(5, result.Ids.Count);
        }

        [Test]
        public void BertTokenizer_DecodeRoundTrips()
        {
            var vocabText = LoadTestVocab();
            if (vocabText == null) { Assert.Ignore("Test vocab not available"); return; }

            var tokenizer = Tokenizer.CreateWordPiece(Encoding.UTF8.GetBytes(vocabText));
            var encoded = tokenizer.Encode("hello world");
            var decoded = tokenizer.Decode(encoded.Ids, skipSpecialTokens: true) ?? string.Empty;

            Assert.IsNotNull(decoded);
            Assert.IsTrue(decoded.Contains("hello"), $"Decoded text should contain 'hello', got: {decoded}");
            Assert.IsTrue(decoded.Contains("world"), $"Decoded text should contain 'world', got: {decoded}");
        }

        [Test]
        public void BertTokenizer_CountTokens()
        {
            var vocabText = LoadTestVocab();
            if (vocabText == null) { Assert.Ignore("Test vocab not available"); return; }

            var tokenizer = Tokenizer.CreateWordPiece(Encoding.UTF8.GetBytes(vocabText));
            var count = tokenizer.CountTokens("Hello world");

            Assert.Greater(count, 0);
        }

        [Test]
        public void EncodingResult_PadExtends()
        {
            var result = new EncodingResult
            {
                Ids = new List<int> { 1, 2, 3 },
                TypeIds = new List<int> { 0, 0, 0 },
                Tokens = new List<string> { "a", "b", "c" },
                AttentionMask = new List<int> { 1, 1, 1 },
                SpecialTokensMask = new List<int> { 0, 0, 0 },
                Words = new List<int?> { null, null, null },
                Offsets = new List<(int Start, int End)> { (0, 1), (1, 2), (2, 3) }
            };

            result.Pad(5);

            Assert.AreEqual(5, result.Ids.Count);
            Assert.AreEqual(0, result.Ids[3]);
            Assert.AreEqual(0, result.AttentionMask[3]);
            Assert.AreEqual(0, result.TypeIds[3]);
        }

        [Test]
        public void EncodingResult_TruncateShortens()
        {
            var result = new EncodingResult
            {
                Ids = new List<int> { 1, 2, 3, 4, 5 },
                TypeIds = new List<int> { 0, 0, 0, 0, 0 },
                Tokens = new List<string> { "a", "b", "c", "d", "e" },
                AttentionMask = new List<int> { 1, 1, 1, 1, 1 },
                SpecialTokensMask = new List<int> { 0, 0, 0, 0, 0 },
                Words = new List<int?> { null, null, null, null, null },
                Offsets = new List<(int Start, int End)> { (0, 1), (1, 2), (2, 3), (3, 4), (4, 5) }
            };

            result.Truncate(3);

            Assert.AreEqual(3, result.Ids.Count);
        }

        [Test]
        public void TokenizerFactory_CreateFromTokenizerJson_InvalidJson_Throws()
        {
            Assert.Catch<JsonReaderException>(() =>
            {
                Tokenizer.FromTokenizerJson(Encoding.UTF8.GetBytes("not json"));
            });
        }

        [Test]
        public void TokenizerFactory_CreateFromTokenizerJson_MissingModelType_Throws()
        {
            Assert.Throws<TokenizerNotSupportedException>(() =>
            {
                Tokenizer.FromTokenizerJson(Encoding.UTF8.GetBytes("{}"));
            });
        }
    }
}

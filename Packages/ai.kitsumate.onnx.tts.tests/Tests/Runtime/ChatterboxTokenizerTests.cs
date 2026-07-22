using System;
using System.IO;
using System.Linq;
using System.Text;
using KitsuMate.Tokenizers;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using UnityEngine;
using KitsuMate.Onnx.Tts.Chatterbox;

namespace KitsuMate.Onnx.Tts.Tests
{
    /// <summary>
    /// Chatterbox tokenizer tests comparing C# output against Python ground truth.
    /// Both use the same tokenizer.json with TemplateProcessing post-processor.
    /// </summary>
    public class ChatterboxTokenizerTests
    {
        // Paths relative to project root
        private const string TokenizerJsonPath = "KitsuMateOnnxFixtures/chatterbox/tokenizer.json";
        private const string GroundTruthPath = "Packages/ai.kitsumate.onnx.tts.tests/Tests/Runtime/tokenizer_ground_truth.json";

        private Tokenizer _tokenizer;
        private JObject _groundTruth;

        [OneTimeSetUp]
        public void LoadTokenizer()
        {
            var tokenizerPath = Path.GetFullPath(TokenizerJsonPath);
            Assert.That(File.Exists(tokenizerPath), Is.True,
                $"Required Chatterbox tokenizer fixture is missing: {tokenizerPath}");

            var groundTruthPath = Path.GetFullPath(GroundTruthPath);
            Assert.That(File.Exists(groundTruthPath), Is.True,
                $"Required Chatterbox tokenizer ground truth is missing: {groundTruthPath}");

            var tokenizerJson = File.ReadAllText(tokenizerPath);
            _tokenizer = Tokenizer.FromTokenizerJson(Encoding.UTF8.GetBytes(tokenizerJson));

            var groundTruthJson = File.ReadAllText(groundTruthPath);
            _groundTruth = JObject.Parse(groundTruthJson);
        }

        [OneTimeTearDown]
        public void Cleanup()
        {
        }

        // =================================================================
        // Template structure: all outputs should have the correct framing
        // =================================================================

        [Test]
        public void Encode_AllCases_HaveCorrectTemplatePrefix()
        {
            var prefix = (_groundTruth["template_prefix"] as JArray)?.Values<long>().ToArray();
            Assert.IsNotNull(prefix, "Ground truth is missing 'template_prefix'.");

            foreach (var tc in (_groundTruth["test_cases"] as JArray).Children<JObject>())
            {
                var text = GetEncodableText(tc);
                var ids = Encode(text);

                Assert.GreaterOrEqual(ids.Length, prefix.Length,
                    $"Token sequence too short for prefix: {tc.Value<string>("name")}");

                for (int i = 0; i < prefix.Length; i++)
                {
                    Assert.AreEqual(prefix[i], ids[i],
                        $"Prefix mismatch at position {i} for: {tc.Value<string>("name")}");
                }
            }
        }

        [Test]
        public void Encode_AllCases_HaveCorrectTemplateSuffix()
        {
            var suffix = (_groundTruth["template_suffix"] as JArray)?.Values<long>().ToArray();
            Assert.IsNotNull(suffix, "Ground truth is missing 'template_suffix'.");

            foreach (var tc in (_groundTruth["test_cases"] as JArray).Children<JObject>())
            {
                var text = GetEncodableText(tc);
                var ids = Encode(text);

                Assert.GreaterOrEqual(ids.Length, suffix.Length,
                    $"Token sequence too short for suffix: {tc.Value<string>("name")}");

                for (int i = 0; i < suffix.Length; i++)
                {
                    Assert.AreEqual(suffix[i], ids[ids.Length - suffix.Length + i],
                        $"Suffix mismatch at position -{suffix.Length - i} for: {tc.Value<string>("name")}");
                }
            }
        }

        // =================================================================
        // Exact match: C# token IDs == Python ground truth
        // =================================================================

        [Test]
        public void Encode_EnglishCases_MatchGroundTruth()
        {
            int passed = 0;
            foreach (var tc in (_groundTruth["test_cases"] as JArray).Children<JObject>())
            {
                if (tc.Value<bool>("is_multilingual"))
                    continue;

                var name = tc.Value<string>("name");
                var text = tc.Value<string>("text") ?? string.Empty;
                var expected = (tc["expected_ids"] as JArray)?.Values<long>().ToArray();
                Assert.IsNotNull(expected, $"Ground truth entry '{name}' is missing 'expected_ids'.");

                var actual = Encode(text);

                Assert.AreEqual(expected.Length, actual.Length,
                    $"Length mismatch for '{name}': expected {expected.Length}, got {actual.Length}\n" +
                    $"  expected: [{string.Join(", ", expected)}]\n" +
                    $"  actual:   [{string.Join(", ", actual)}]");

                CollectionAssert.AreEqual(expected, actual,
                    $"Token ID mismatch for '{name}': \"{text}\"");

                passed++;
                Debug.Log($"  [{passed}] PASS '{name}' ({actual.Length} tokens)");
            }

            Debug.Log($"[Chatterbox Tokenizer] English: {passed} cases passed");
            Assert.Greater(passed, 0, "No English test cases found in ground truth");
        }

        [Test]
        public void Encode_MultilingualCases_MatchGroundTruth()
        {
            int passed = 0;
            foreach (var tc in (_groundTruth["test_cases"] as JArray).Children<JObject>())
            {
                if (!tc.Value<bool>("is_multilingual"))
                    continue;

                var name = tc.Value<string>("name");
                var text = tc.Value<string>("text") ?? string.Empty;
                var langId = tc.Value<string>("language_id");
                var expected = (tc["expected_ids"] as JArray)?.Values<long>().ToArray();
                Assert.IsNotNull(expected, $"Ground truth entry '{name}' is missing 'expected_ids'.");

                // Multilingual text goes through LanguagePreprocessor + lang token prepend
                // before reaching the tokenizer. The ground truth already has the preprocessed
                // form, so we use preprocessed_text directly.
                var preprocessedText = tc.Value<string>("preprocessed_text") ?? string.Empty;
                var actual = Encode(preprocessedText);

                Assert.AreEqual(expected.Length, actual.Length,
                    $"Length mismatch for '{name}': expected {expected.Length}, got {actual.Length}\n" +
                    $"  expected: [{string.Join(", ", expected)}]\n" +
                    $"  actual:   [{string.Join(", ", actual)}]");

                CollectionAssert.AreEqual(expected, actual,
                    $"Token ID mismatch for '{name}': \"{text}\" (lang={langId})");

                passed++;
                Debug.Log($"  [{passed}] PASS '{name}' ({actual.Length} tokens)");
            }

            Debug.Log($"[Chatterbox Tokenizer] Multilingual: {passed} cases passed");
            Assert.Greater(passed, 0, "No multilingual test cases found in ground truth");
        }

        // =================================================================
        // Special token IDs match expected values
        // =================================================================

        [Test]
        public void SpecialTokenIds_MatchConstants()
        {
            // Verify the template IDs by encoding a minimal string and checking framing
            var ids = Encode("a");

            // [EXAGGERATION](6563) [START](255) a(...) [STOP](0) [START_SPEECH](6561) [START_SPEECH](6561)
            Assert.AreEqual(ChatterboxConstants.ExaggerationToken, ids[0],
                $"First token should be EXAGGERATION ({ChatterboxConstants.ExaggerationToken})");
            Assert.AreEqual(ChatterboxConstants.StartTextToken, ids[1],
                $"Second token should be START ({ChatterboxConstants.StartTextToken})");
            Assert.AreEqual(ChatterboxConstants.StopTextToken, ids[ids.Length - 3],
                $"Third-to-last token should be STOP ({ChatterboxConstants.StopTextToken})");
            Assert.AreEqual(ChatterboxConstants.StartSpeechToken, ids[ids.Length - 2],
                $"Second-to-last token should be START_SPEECH ({ChatterboxConstants.StartSpeechToken})");
            Assert.AreEqual(ChatterboxConstants.StartSpeechToken, ids[ids.Length - 1],
                $"Last token should be START_SPEECH ({ChatterboxConstants.StartSpeechToken})");
        }

        // =================================================================
        // Space handling: space char should map to a single token (ID 2)
        // =================================================================

        [Test]
        public void Encode_Space_MapsToSingleToken()
        {
            // "a b" should have exactly one more token than "ab" (the space token)
            var withSpace = Encode("a b");
            var noSpace = Encode("ab");

            // Both have the same 5 framing tokens (2 prefix + 3 suffix)
            // "a b" should have text tokens for: a, space, b
            // "ab" should have text tokens for: a, b (or merged)
            // The space should be a single token, not multiple sub-tokens
            Assert.Greater(withSpace.Length, noSpace.Length,
                "Adding a space should increase token count");

            // Find the space token (should be ID 2 based on ground truth)
            // Template: [6563, 255, ..text.., 0, 6561, 6561]
            // For "a b", text tokens are between index 2 and Length-3
            var textTokens = withSpace.Skip(2).Take(withSpace.Length - 5).ToArray();
            Assert.IsTrue(textTokens.Contains(2L),
                $"Space should map to token ID 2, text tokens: [{string.Join(", ", textTokens)}]");
        }

        // =================================================================
        // Edge cases
        // =================================================================

        [Test]
        public void Encode_EmptyString_ReturnsOnlyFramingTokens()
        {
            var ids = Encode("");

            // Should be just: [EXAGGERATION][START][STOP][START_SPEECH][START_SPEECH]
            Assert.AreEqual(5, ids.Length,
                $"Empty string should produce 5 framing tokens, got {ids.Length}: [{string.Join(", ", ids)}]");
        }

        [Test]
        public void Encode_SingleChar_ProducesValidOutput()
        {
            var ids = Encode("x");

            // At minimum: 2 prefix + 1 text token + 3 suffix = 6
            Assert.GreaterOrEqual(ids.Length, 6,
                $"Single char should produce at least 6 tokens, got {ids.Length}");
        }

        // =================================================================
        // Helpers
        // =================================================================

        /// <summary>
        /// Get the text to pass to Encode() for a test case.
        /// For multilingual cases, uses the preprocessed_text (already lowercased + NFKD + lang token).
        /// For English cases, uses the raw text.
        /// </summary>
        private static string GetEncodableText(JObject testCase)
        {
            if (testCase.Value<bool>("is_multilingual") &&
                testCase.TryGetValue("preprocessed_text", out var preprocessed))
            {
                return preprocessed.Value<string>() ?? string.Empty;
            }

            return testCase.Value<string>("text") ?? string.Empty;
        }

        private long[] Encode(string text)
        {
            var encoded = _tokenizer.Encode(text, addSpecialTokens: true);
            return encoded.Ids.Select(static id => (long)id).ToArray();
        }
    }
}

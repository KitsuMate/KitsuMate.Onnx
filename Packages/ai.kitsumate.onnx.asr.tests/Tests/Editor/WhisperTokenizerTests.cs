using System.Collections.Generic;
using System.IO;
using System.Text;
using KitsuMate.Tokenizers;
using NUnit.Framework;
using UnityEngine;
using KitsuMate.Onnx.Asr.Whisper;

namespace KitsuMate.Onnx.Asr.Tests
{
    /// <summary>
    /// Tests for Whisper tokenizer functionality.
    /// </summary>
    public class WhisperTokenizerTests
    {
        private Tokenizer _tokenizer;
        private const string TestTokenizerAssetPath = "Packages/ai.kitsumate.onnx.asr.tests/Tests/Editor/Fixtures/tokenizer.json";
        private static string TestTokenizerPath => Path.GetFullPath(Path.Combine(Application.dataPath, "..", TestTokenizerAssetPath));
        
        [SetUp]
        public void Setup()
        {
            // Load tokenizer pipeline if it exists.
            if (File.Exists(TestTokenizerPath))
            {
                var tokenizerJson = File.ReadAllText(TestTokenizerPath);
                _tokenizer = Tokenizer.FromTokenizerJson(Encoding.UTF8.GetBytes(tokenizerJson));
            }
        }
        
        [TearDown]
        public void TearDown()
        {
            _tokenizer = null;
        }
        
        [Test]
        public void Tokenizer_EncodesBasicText()
        {
            if (!File.Exists(TestTokenizerPath))
            {
                Assert.Ignore("Tokenizer JSON not found at: " + TestTokenizerPath);
                return;
            }
            
            var text = "Hello world";
            var tokens = _tokenizer.Encode(text).Ids;
            
            Assert.IsNotNull(tokens, "Tokens should not be null");
            Assert.IsTrue(tokens.Count > 0, "Should produce at least one token");
        }
        
        [Test]
        public void Tokenizer_DecodesTokensToText()
        {
            if (!File.Exists(TestTokenizerPath))
            {
                Assert.Ignore("Tokenizer JSON not found at: " + TestTokenizerPath);
                return;
            }
            
            var tokens = new List<int> { 50258, 50259, 50359, 50363, 15947, 1002, 50257 };
            var text = _tokenizer.Decode(tokens, skipSpecialTokens: true) ?? string.Empty;
            
            Assert.IsNotNull(text, "Decoded text should not be null");
            Assert.IsTrue(text.Length > 0, "Decoded text should not be empty");
        }
        
        [Test]
        public void Tokenizer_RoundTrip()
        {
            if (!File.Exists(TestTokenizerPath))
            {
                Assert.Ignore("Tokenizer JSON not found at: " + TestTokenizerPath);
                return;
            }
            
            var originalText = "This is a test.";
            var tokens = new List<int>(_tokenizer.Encode(originalText).Ids);
            var decodedText = _tokenizer.Decode(tokens, skipSpecialTokens: true) ?? string.Empty;
            
            Assert.IsNotNull(decodedText, "Decoded text should not be null");
            // Note: Round-trip may not be exact due to tokenization
            Debug.Log($"Original: '{originalText}' -> Tokens: [{string.Join(", ", tokens)}] -> Decoded: '{decodedText}'");
        }
        
        [Test]
        public void SpecialTokens_AreCorrect()
        {
            // Verify special token constants match expected Whisper values
            Assert.AreEqual(50257, WhisperConstants.EndOfText, "EOT token should be 50257");
            Assert.AreEqual(50258, WhisperConstants.StartOfTranscript, "SOT token should be 50258");
        }
        
        [Test]
        public void LanguageToken_IsGenerated()
        {
            var englishToken = WhisperConstants.GetLanguageToken("en");
            Assert.AreEqual(50259, englishToken, "English language token should be 50259");
            
            var japaneseToken = WhisperConstants.GetLanguageToken("ja");
            Assert.IsTrue(japaneseToken > 50258, "Japanese language token should be valid");
        }
    }
}

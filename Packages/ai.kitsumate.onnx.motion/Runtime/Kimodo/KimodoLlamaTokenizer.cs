using System;
using System.Collections.Generic;
using System.IO;
using KitsuMate.Tokenizers;

namespace KitsuMate.Onnx.Motion.Kimodo
{
    internal sealed class KimodoTokenizedPrompt
    {
        public string SanitizedPrompt { get; }
        public long[] InputIds { get; }
        public long[] AttentionMask { get; }
        public long[] PoolingMask { get; }

        public KimodoTokenizedPrompt(string sanitizedPrompt, long[] inputIds, long[] attentionMask, long[] poolingMask)
        {
            SanitizedPrompt = sanitizedPrompt;
            InputIds = inputIds;
            AttentionMask = attentionMask;
            PoolingMask = poolingMask;
        }
    }

    /// <summary>Exact Llama 3 formatting and instruction-skipping mask construction.</summary>
    internal sealed class KimodoLlamaTokenizer
    {
        private readonly Tokenizer _tokenizer;
        private readonly int _sequenceLength;

        public KimodoLlamaTokenizer(string tokenizerJsonPath, string tokenizerConfigPath = null, int sequenceLength = KimodoLlm2VecContract.SequenceLength)
        {
            if (string.IsNullOrWhiteSpace(tokenizerJsonPath))
                throw new ArgumentException("Tokenizer path is required.", nameof(tokenizerJsonPath));
            if (!Path.IsPathRooted(tokenizerJsonPath))
                throw new ArgumentException("Tokenizer path must be absolute.", nameof(tokenizerJsonPath));
            if (!File.Exists(tokenizerJsonPath))
                throw new FileNotFoundException("Llama tokenizer.json was not found.", tokenizerJsonPath);
            if (sequenceLength < 4) throw new ArgumentOutOfRangeException(nameof(sequenceLength));

            byte[] config = !string.IsNullOrWhiteSpace(tokenizerConfigPath) && File.Exists(tokenizerConfigPath)
                ? File.ReadAllBytes(tokenizerConfigPath)
                : null;
            _tokenizer = Tokenizer.FromTokenizerJson(File.ReadAllBytes(tokenizerJsonPath), null, config);
            _sequenceLength = sequenceLength;
        }

        public KimodoTokenizedPrompt Encode(string prompt)
        {
            string sanitized = KimodoTextPreprocessor.Sanitize(prompt ?? throw new ArgumentNullException(nameof(prompt)));
            var content = _tokenizer.Encode(sanitized, new TokenizerEncodeOptions
            {
                AddSpecialTokens = false,
                Padding = TokenizerPaddingMode.None,
                Truncation = TokenizerTruncationMode.None,
                ReturnAttentionMask = true,
            });

            // KitsuMate.Tokenizers currently segments the Llama header's two newlines
            // as [198,198], while the authoritative tokenizer merges them to token 271.
            // Construct the invariant instruction envelope explicitly and use the shared
            // BPE implementation only for sanitized prompt content.
            var fullIds = new List<long>(content.Ids.Count + 6)
            {
                KimodoLlm2VecContract.BeginOfTextTokenId,
                KimodoLlm2VecContract.StartHeaderTokenId,
                KimodoLlm2VecContract.UserTokenId,
                KimodoLlm2VecContract.EndHeaderTokenId,
                KimodoLlm2VecContract.DoubleNewlineTokenId,
            };
            for (int i = 0; i < content.Ids.Count; i++) fullIds.Add(content.Ids[i]);
            fullIds.Add(KimodoLlm2VecContract.EndOfTurnTokenId);

            int retained = Math.Min(fullIds.Count, _sequenceLength);
            int padding = _sequenceLength - retained;
            const int poolingStart = 5;
            var ids = new long[_sequenceLength];
            var attention = new long[_sequenceLength];
            var pooling = new long[_sequenceLength];
            for (int i = 0; i < padding; i++) ids[i] = KimodoLlm2VecContract.PadTokenId;
            for (int source = 0; source < retained; source++)
            {
                int destination = padding + source;
                ids[destination] = fullIds[source];
                attention[destination] = 1;
                pooling[destination] = source >= poolingStart ? 1 : 0;
            }
            bool hasPoolingToken = false;
            for (int i = 0; i < pooling.Length; i++) hasPoolingToken |= pooling[i] != 0;
            if (!hasPoolingToken)
                throw new ArgumentException($"Prompt is too long for the {_sequenceLength}-token LLM2Vec model.", nameof(prompt));

            return new KimodoTokenizedPrompt(sanitized, ids, attention, pooling);
        }
    }
}

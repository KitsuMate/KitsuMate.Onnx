using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using KitsuMate.Tokenizers;

namespace KitsuMate.Onnx.Tts.NeuTts
{
    internal static class NeuTtsPromptBuilder
    {
        internal static string Normalize(string text) => text.Replace('\u2018', '\'').Replace('\u2019', '\'')
            .Replace('\u201c', '"').Replace('\u201d', '"').Normalize(NormalizationForm.FormKC);

        internal static long[] Build(Tokenizer tokenizer, NeuTtsMetadata metadata, string text, NeuTtsGenerationConfig config)
        {
            config.Validate();
            var speaker = metadata.speakers.SingleOrDefault(s => s.name == config.Speaker)
                ?? throw new ArgumentException($"Unknown NeuTTS speaker '{config.Speaker}'.");
            var emotion = metadata.emotions.SingleOrDefault(e => e.name == config.Emotion)
                ?? throw new ArgumentException($"Unknown NeuTTS emotion '{config.Emotion}'.");
            var ids = new List<long> { metadata.textStart };
            void Encode(string value) => ids.AddRange(tokenizer.Encode(value, addSpecialTokens: false).Ids.Select(id => (long)id));
            if (emotion.name == "neutral") Encode(Normalize(speaker.text) + " " + Normalize(text));
            else
            {
                Encode(Normalize(speaker.text));
                ids.Add(emotion.token);
                Encode(Normalize(text));
            }
            ids.Add(metadata.textEnd);
            ids.Add(metadata.speechStart);
            ids.AddRange(speaker.codes.Select(code => (long)metadata.speechTokenIds[code]));
            return ids.ToArray();
        }
    }
}

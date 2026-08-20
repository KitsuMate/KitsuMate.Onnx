using System;
using System.Collections.Generic;
using System.Text.RegularExpressions;
using KitsuMate.Tokenizers;

namespace KitsuMate.Onnx.Tts.OmniVoice
{
    internal enum OmniVoiceVoiceMode { Auto, Design, Clone }

    internal static class OmniVoicePromptBuilder
    {
        private static readonly Regex NonVerbal = new(
            @"\[(laughter|sigh|confirmation-en|question-en|question-ah|question-oh|question-ei|question-yi|surprise-ah|surprise-oh|surprise-wa|surprise-yo|dissatisfaction-hnn)\]",
            RegexOptions.Compiled | RegexOptions.CultureInvariant);

        internal static string StyleText(string language, string instruction, bool denoise, bool hasReference)
        {
            string value = denoise && hasReference ? "<|denoise|>" : string.Empty;
            value += $"<|lang_start|>{ValueOrNone(language)}<|lang_end|>";
            value += $"<|instruct_start|>{ValueOrNone(instruction)}<|instruct_end|>";
            return value;
        }

        internal static OmniVoiceVoiceMode ResolveMode(TtsRequest request)
        {
            if (request == null) throw new ArgumentNullException(nameof(request));
            if (request.VoiceReference != null)
            {
                if (string.IsNullOrWhiteSpace(request.VoiceReferenceText))
                    throw new ArgumentException("OmniVoice cloning requires VoiceReferenceText.", nameof(request));
                return OmniVoiceVoiceMode.Clone;
            }
            if (!string.IsNullOrWhiteSpace(request.VoiceReferenceText))
                throw new ArgumentException("VoiceReferenceText requires VoiceReference.", nameof(request));
            return string.IsNullOrWhiteSpace(request.VoiceInstruction)
                ? OmniVoiceVoiceMode.Auto
                : OmniVoiceVoiceMode.Design;
        }

        internal static string WrappedText(string text, string referenceText)
        {
            if (string.IsNullOrWhiteSpace(text)) throw new ArgumentException("Text is required.", nameof(text));
            string combined = string.IsNullOrWhiteSpace(referenceText)
                ? text.Trim()
                : referenceText.Trim() + " " + text.Trim();
            combined = Regex.Replace(combined, "[\r\n]+", string.Empty)
                .Replace('（', '(').Replace('）', ')');
            combined = Regex.Replace(combined, "[ \t]+", " ");
            combined = Regex.Replace(combined, @"(?<=[\u4e00-\u9fff])\s+|\s+(?=[\u4e00-\u9fff])", string.Empty);
            return $"<|text_start|>{combined}<|text_end|>";
        }

        internal static long[] EncodeText(Tokenizer tokenizer, string wrappedText)
        {
            var ids = new List<long>();
            int end = 0;
            foreach (Match match in NonVerbal.Matches(wrappedText))
            {
                if (match.Index > end) Add(tokenizer, wrappedText.Substring(end, match.Index - end), ids);
                Add(tokenizer, match.Value, ids);
                end = match.Index + match.Length;
            }
            if (end < wrappedText.Length) Add(tokenizer, wrappedText.Substring(end), ids);
            if (ids.Count == 0) Add(tokenizer, wrappedText, ids);
            return ids.ToArray();
        }

        internal static long[] Encode(Tokenizer tokenizer, string text)
        {
            var result = new List<long>();
            Add(tokenizer, text, result);
            return result.ToArray();
        }

        private static void Add(Tokenizer tokenizer, string text, ICollection<long> destination)
        {
            foreach (uint id in tokenizer.Encode(text, addSpecialTokens: false).Ids) destination.Add(id);
        }

        private static string ValueOrNone(string value) => string.IsNullOrWhiteSpace(value) ? "None" : value.Trim();
    }
}

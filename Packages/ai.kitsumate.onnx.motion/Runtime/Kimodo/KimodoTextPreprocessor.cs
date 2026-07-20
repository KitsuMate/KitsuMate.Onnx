using System;
using System.Globalization;
using System.Linq;

namespace KitsuMate.Onnx.Motion.Kimodo
{
    /// <summary>Port of kimodo.sanitize.sanitize_text.</summary>
    internal static class KimodoTextPreprocessor
    {
        private const string AcceptedFinalPunctuation = ".!?\"])'";

        public static string Sanitize(string text)
        {
            if (text == null) throw new ArgumentNullException(nameof(text));
            text = string.Join(" ", text.Split((char[])null, StringSplitOptions.RemoveEmptyEntries));
            if (text.Length == 0) return string.Empty;

            int first = 0;
            while (first < text.Length && !char.IsLetterOrDigit(text[first])) first++;
            // Preserve the reference implementation's behavior for punctuation-only prompts.
            if (first == text.Length) first = text.Length - 1;
            text = text.Substring(first);
            text = CapitalizeLikePython(text);

            int last = text.Length - 1;
            while (last >= 0 && !char.IsLetterOrDigit(text[last]) &&
                   AcceptedFinalPunctuation.IndexOf(text[last]) < 0)
                last--;
            if (last < 0) last = 0;
            text = text.Substring(0, last + 1);
            if (".!?".IndexOf(text[text.Length - 1]) < 0) text += ".";

            foreach (char sentenceBreak in new[] { '.', '!', '?' })
            {
                string[] parts = text.Split(sentenceBreak);
                text = string.Join(sentenceBreak + " ", parts.Select(part =>
                {
                    string trimmed = part.Trim();
                    return trimmed.Length == 0 ? trimmed : CapitalizeFirstOnly(trimmed);
                })).Trim();
            }
            return text;
        }

        private static string CapitalizeLikePython(string value)
        {
            string lowered = value.ToLowerInvariant();
            return CapitalizeFirstOnly(lowered);
        }

        private static string CapitalizeFirstOnly(string value)
        {
            if (value.Length == 0) return value;
            string first = char.ToUpper(value[0], CultureInfo.InvariantCulture).ToString();
            return value.Length == 1 ? first : first + value.Substring(1);
        }
    }
}

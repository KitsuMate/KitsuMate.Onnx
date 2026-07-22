using System.Text.RegularExpressions;

namespace KitsuMate.Onnx.Editor.Download
{
    /// <summary>
    /// Converts quantization suffixes from file names into human-readable display names.
    /// </summary>
    public static class QuantizationNameHelper
    {
        /// <summary>
        /// Converts a quantization suffix (e.g., "_q4f16", "_int8", "_fp16") to a display name.
        /// </summary>
        public static string ToDisplayName(string suffix)
        {
            if (string.IsNullOrEmpty(suffix))
                return "Float32";

            // Remove leading underscore
            var s = suffix.TrimStart('_');
            if (string.IsNullOrEmpty(s))
                return "Float32";

            // q{N}f{N} -> Q{N} Float{N}
            var qfMatch = Regex.Match(s, @"^q(\d+)f(\d+)$", RegexOptions.IgnoreCase);
            if (qfMatch.Success)
                return $"Q{qfMatch.Groups[1].Value} Float{qfMatch.Groups[2].Value}";

            // q{N} -> Q{N}
            var qMatch = Regex.Match(s, @"^q(\d+)$", RegexOptions.IgnoreCase);
            if (qMatch.Success)
                return $"Q{qMatch.Groups[1].Value}";

            // int{N} -> Int{N}
            var intMatch = Regex.Match(s, @"^int(\d+)$", RegexOptions.IgnoreCase);
            if (intMatch.Success)
                return $"Int{intMatch.Groups[1].Value}";

            // uint{N} -> UInt{N}
            var uintMatch = Regex.Match(s, @"^uint(\d+)$", RegexOptions.IgnoreCase);
            if (uintMatch.Success)
                return $"UInt{uintMatch.Groups[1].Value}";

            // fp{N} -> Float{N}
            var fpMatch = Regex.Match(s, @"^fp(\d+)$", RegexOptions.IgnoreCase);
            if (fpMatch.Success)
                return $"Float{fpMatch.Groups[1].Value}";

            // bnb{N} -> BitsAndBytes{N}
            var bnbMatch = Regex.Match(s, @"^bnb(\d+)$", RegexOptions.IgnoreCase);
            if (bnbMatch.Success)
                return $"BitsAndBytes{bnbMatch.Groups[1].Value}";

            // quantized -> Quantized
            if (s.Equals("quantized", System.StringComparison.OrdinalIgnoreCase))
                return "Quantized";

            // Default: capitalize first letter
            return char.ToUpperInvariant(s[0]) + s.Substring(1).ToLowerInvariant();
        }

        /// <summary>
        /// Extracts the quantization suffix from a file name.
        /// E.g., "encoder_model_q4f16.onnx" with base "encoder_model" returns "_q4f16".
        /// Returns empty string if no suffix (base variant).
        /// </summary>
        public static string ExtractSuffix(string fileName, string baseName)
        {
            if (string.IsNullOrEmpty(fileName) || string.IsNullOrEmpty(baseName))
                return null;

            // Remove extension
            var nameWithoutExt = System.IO.Path.GetFileNameWithoutExtension(fileName);

            if (!nameWithoutExt.StartsWith(baseName, System.StringComparison.OrdinalIgnoreCase))
                return null;

            return nameWithoutExt.Substring(baseName.Length);
        }
    }
}

using System;
using System.IO;
using System.Linq;

namespace KitsuMate.Onnx.Editor
{
    /// <summary>
    /// Helpers for working with supported model file formats.
    /// </summary>
    public static class ModelFileUtility
    {
        public const string OnnxExtension = ".onnx";
        public const string OrtExtension = ".ort";

        public static readonly string[] SupportedModelExtensions =
        {
            OnnxExtension,
            OrtExtension,
        };

        public static bool IsSupportedModelPath(string path)
        {
            return HasSupportedExtension(Path.GetExtension(path));
        }

        public static bool IsOnnxModelPath(string path)
        {
            return string.Equals(Path.GetExtension(path), OnnxExtension, StringComparison.OrdinalIgnoreCase);
        }

        public static string[] GetModelFilePatterns(string baseName, string suffix = null)
        {
            var stem = (baseName ?? string.Empty) + (suffix ?? string.Empty);
            return SupportedModelExtensions
                .Select(extension => stem + extension)
                .ToArray();
        }

        private static bool HasSupportedExtension(string extension)
        {
            if (string.IsNullOrEmpty(extension))
                return false;

            return SupportedModelExtensions.Any(supported =>
                string.Equals(extension, supported, StringComparison.OrdinalIgnoreCase));
        }
    }
}
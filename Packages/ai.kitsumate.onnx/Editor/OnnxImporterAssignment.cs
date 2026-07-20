#if UNITY_EDITOR
using System;
using UnityEditor;

namespace KitsuMate.Onnx.Editor
{
    /// <summary>
    /// Explicitly assigns KitsuMate importers to framework-owned model files.
    /// Sentis remains the primary importer for unrelated ONNX assets when installed.
    /// </summary>
    public static class OnnxImporterAssignment
    {
        public static bool IsFrameworkModelPath(string assetPath)
        {
            return ModelFileUtility.IsSupportedModelPath(assetPath) ||
                   assetPath.EndsWith(".onnx_data", StringComparison.OrdinalIgnoreCase);
        }

        public static void AssignFrameworkImporter(string assetPath)
        {
            if (string.IsNullOrWhiteSpace(assetPath))
                throw new ArgumentException("An asset path is required.", nameof(assetPath));

            assetPath = assetPath.Replace('\\', '/');
            if (ModelFileUtility.IsSupportedModelPath(assetPath))
            {
                AssetDatabase.SetImporterOverride<OnnxModelImporter>(assetPath);
                return;
            }

            if (assetPath.EndsWith(".onnx_data", StringComparison.OrdinalIgnoreCase))
            {
                AssetDatabase.SetImporterOverride<OnnxDataImporter>(assetPath);
                return;
            }

            throw new ArgumentException($"'{assetPath}' is not a supported KitsuMate model asset.", nameof(assetPath));
        }
    }
}
#endif

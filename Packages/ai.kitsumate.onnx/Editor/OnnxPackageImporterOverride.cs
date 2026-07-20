#if UNITY_EDITOR && KITSUMATE_HAS_SENTIS
using System.Collections.Generic;
using UnityEditor;

namespace KitsuMate.Onnx.Editor
{
    /// <summary>
    /// Ensures supported model files inside any kitsumate.onnx package always use
    /// KitsuMate importers, even when Unity AI Inference is installed.
    /// Unity normalizes all package asset paths to "Packages/{name}/" regardless
    /// of disk location, so a prefix check is reliable and fast.
    /// .onnx_data is always primary since Unity AI Inference does not handle it.
    /// </summary>
    internal sealed class OnnxPackageImporterOverride : AssetPostprocessor
    {
        private const string PackageRoot = "Packages/ai.kitsumate.onnx";

        private static readonly HashSet<string> _pendingOverrides = new();

        private void OnPreprocessAsset()
        {
            if (!assetPath.StartsWith(PackageRoot))
                return;

            if (ModelFileUtility.IsSupportedModelPath(assetPath) && assetImporter is not OnnxModelImporter)
                _pendingOverrides.Add(assetPath);
        }

        private static void OnPostprocessAllAssets(
            string[] importedAssets, string[] deletedAssets,
            string[] movedAssets, string[] movedFromAssetPaths)
        {
            if (_pendingOverrides.Count == 0)
                return;

            // Copy and clear to avoid re-entrancy
            var paths = new List<string>(_pendingOverrides);
            _pendingOverrides.Clear();

            foreach (var path in paths)
                OnnxImporterAssignment.AssignFrameworkImporter(path);
        }
    }
}
#endif

#if UNITY_EDITOR
using System;
using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEditor.AssetImporters;
using UnityEngine;

namespace KitsuMate.Onnx.Editor
{
    /// <summary>
    /// Scripted importer for .onnx and .ort files.
    /// Creates OnnxModelAsset ScriptableObjects from supported ONNX model files.
    /// </summary>
#if KITSUMATE_HAS_UNITY_AI_INFERENCE
    [ScriptedImporter(4, null, new[] { "onnx", "ort" }, importQueueOffset: 100)]
#else
    [ScriptedImporter(4, new[] { "onnx", "ort" }, importQueueOffset: 100)]
#endif
    public class OnnxModelImporter : ScriptedImporter
    {
        private sealed class MetadataResult
        {
            public string ModelName { get; set; } = string.Empty;
            public string GraphName { get; set; } = string.Empty;
            public string Domain { get; set; } = string.Empty;
            public string Description { get; set; } = string.Empty;
            public string Producer { get; set; } = string.Empty;
            public long OpsetVersion { get; set; }
            public List<OnnxModelAsset.TensorInfo> Inputs { get; set; } = new();
            public List<OnnxModelAsset.TensorInfo> Outputs { get; set; } = new();
            public List<OnnxModelAsset.MetadataEntry> CustomMetadata { get; set; } = new();
        }

        public override void OnImportAsset(AssetImportContext ctx)
        {
            var modelAsset = ScriptableObject.CreateInstance<OnnxModelAsset>();
            OnnxModelData modelData = null;

            try
            {
                var fileInfo = new FileInfo(ctx.assetPath);
                string absoluteAssetPath = ResolveAbsoluteAssetPath(ctx.assetPath);
                string modelDirectory = Path.GetDirectoryName(absoluteAssetPath) ?? string.Empty;

                bool isOnnxModel = ModelFileUtility.IsOnnxModelPath(ctx.assetPath);
                bool hasExternalData = isOnnxModel && CheckAndRegisterExternalData(ctx, absoluteAssetPath);
                bool externalDataMerged = false;
                modelData = ScriptableObject.CreateInstance<OnnxModelData>();
                byte[] modelBytes = File.ReadAllBytes(ctx.assetPath);
                if (hasExternalData)
                {
                    byte[] mergedBytes = OnnxExternalDataMerger.MergeExternalData(modelBytes, modelDirectory);
                    if (mergedBytes == modelBytes)
                        throw new InvalidDataException("ONNX external data was not merged into the imported model.");
                    modelBytes = mergedBytes;
                    externalDataMerged = true;
                }
                modelData.SetData(modelBytes);
                modelData.name = "ModelData";
                modelData.hideFlags = HideFlags.HideInHierarchy;
                
                var metadataState = OnnxModelAsset.MetadataInspectionState.NotInspected;
                string metadataError = string.Empty;
                MetadataResult metadataResult = null;

                if (TryExtractMetadata(absoluteAssetPath, out metadataResult, out var inspectionError))
                {
                    metadataState = OnnxModelAsset.MetadataInspectionState.Succeeded;
                }
                else
                {
                    metadataState = OnnxModelAsset.MetadataInspectionState.Failed;
                    metadataError = inspectionError ?? string.Empty;
                    if (!string.IsNullOrEmpty(inspectionError))
                    {
                        ctx.LogImportWarning($"Imported model '{ctx.assetPath}' but failed to inspect metadata: {inspectionError}");
                    }
                }

                string graphName = metadataResult?.GraphName ?? string.Empty;
                string domain = metadataResult?.Domain ?? string.Empty;
                string description = metadataResult?.Description ?? string.Empty;
                string producer = metadataResult?.Producer ?? string.Empty;
                long opsetVersion = metadataResult?.OpsetVersion ?? 0;

                var inputs = metadataResult?.Inputs ?? new List<OnnxModelAsset.TensorInfo>();
                var outputs = metadataResult?.Outputs ?? new List<OnnxModelAsset.TensorInfo>();
                var metadata = metadataResult?.CustomMetadata ?? new List<OnnxModelAsset.MetadataEntry>();

                string fallbackName = Path.GetFileNameWithoutExtension(ctx.assetPath);
                string modelName = string.IsNullOrWhiteSpace(graphName) ? fallbackName : graphName;

                modelAsset.Populate(
                    modelName,
                    graphName,
                    domain,
                    description,
                    producer,
                    opsetVersion,
                    fileInfo.Exists ? fileInfo.Length : 0L,
                    DateTime.UtcNow,
                    ctx.assetPath,
                    inputs,
                    outputs,
                    metadata,
                    modelData,
                    metadataState: metadataState,
                    metadataError: metadataError,
                    externalDataMerged: externalDataMerged);

                var icon = LoadOnnxIcon();
                ctx.AddObjectToAsset("ModelData", modelData);
                ctx.AddObjectToAsset("OnnxModelAsset", modelAsset, icon);
                ctx.SetMainObject(modelAsset);
                if (icon != null)
                    EditorGUIUtility.SetIconForObject(modelAsset, icon);

                modelAsset.name = modelName;
            }
            catch (Exception ex)
            {
                ctx.LogImportError($"Failed to import model '{ctx.assetPath}': {ex.Message}");
                if (modelData != null) ScriptableObject.DestroyImmediate(modelData);
                ScriptableObject.DestroyImmediate(modelAsset);
            }
        }

        private static bool TryExtractMetadata(string modelPath, out MetadataResult result, out string failureReason)
        {
            result = null;
            failureReason = null;

            if (string.IsNullOrWhiteSpace(modelPath) || !File.Exists(modelPath))
            {
                failureReason = "Model contains no data.";
                return false;
            }

            try
            {
                OnnxLightweightMetadataReader.Result metadata = OnnxLightweightMetadataReader.Read(modelPath);

                result = new MetadataResult
                {
                    ModelName = metadata.GraphName,
                    GraphName = metadata.GraphName,
                    Domain = metadata.Domain,
                    Producer = metadata.Producer,
                    OpsetVersion = metadata.OpsetVersion,
                    Description = metadata.Description,
                    Inputs = metadata.Inputs,
                    Outputs = metadata.Outputs,
                    CustomMetadata = metadata.CustomMetadata
                };
                return true;
            }
            catch (Exception ex)
            {
                failureReason = ex.Message;
            }

            return false;
        }

        private static string ResolveAbsoluteAssetPath(string assetPath)
        {
            if (string.IsNullOrEmpty(assetPath))
                return null;

            if (Path.IsPathRooted(assetPath))
                return assetPath;

            var projectRoot = Directory.GetCurrentDirectory();
            assetPath = assetPath.TrimStart(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
            return Path.GetFullPath(Path.Combine(projectRoot, assetPath));
        }

        private static bool CheckAndRegisterExternalData(AssetImportContext ctx, string absoluteModelPath)
        {
            if (string.IsNullOrEmpty(absoluteModelPath))
                return false;

            var metadata = OnnxLightweightMetadataReader.Read(absoluteModelPath);
            foreach (var reference in metadata.ExternalData)
            {
                string path = OnnxLightweightMetadataReader.ResolveExternalDataPath(absoluteModelPath, reference);
                string projectPath = FileUtil.GetProjectRelativePath(path.Replace('\\', '/'));
                if (!projectPath.StartsWith("Assets/", StringComparison.Ordinal))
                    throw new InvalidDataException($"External data is outside Assets: '{reference.Location}'.");
                ctx.DependsOnSourceAsset(projectPath);
            }
            return metadata.ExternalData.Count > 0;
        }

        private static Texture2D LoadOnnxIcon()
        {
            const string iconGuid = "16c6d06ecc04b0444970869c7e0203e6";
            string iconPath = AssetDatabase.GUIDToAssetPath(iconGuid);
            if (string.IsNullOrEmpty(iconPath))
                return null;
            return AssetDatabase.LoadAssetAtPath<Texture2D>(iconPath);
        }
    }
}
#endif

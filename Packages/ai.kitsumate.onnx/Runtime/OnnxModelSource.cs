using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEngine;

namespace KitsuMate.Onnx
{
    public interface IOnnxModelSource
    {
        string SourceName { get; }
        bool IsAvailable { get; }
        bool HasInspectedMetadata { get; }
        bool IsMetadataStale { get; }
        IReadOnlyList<OnnxModelAsset.TensorInfo> Inputs { get; }
        IReadOnlyList<OnnxModelAsset.TensorInfo> Outputs { get; }
        OnnxModelAsset ImportedAsset { get; }
        string ResolveModelPath();
    }

    [Serializable]
    public sealed class OnnxModelReference : IOnnxModelSource
    {
        public enum SourceKind { Asset, File }

        [SerializeField] private SourceKind sourceKind;
        [SerializeField] private OnnxModelAsset asset;
        [SerializeField] private string relativePath;
        [SerializeField, HideInInspector] private string sha256;
        [SerializeField, HideInInspector] private long cachedFileSize;
        [SerializeField, HideInInspector] private long cachedWriteTimeUtcTicks;
        [SerializeField, HideInInspector] private bool metadataInspected;
        [SerializeField, HideInInspector] private List<OnnxModelAsset.TensorInfo> inputs = new();
        [SerializeField, HideInInspector] private List<OnnxModelAsset.TensorInfo> outputs = new();
        [NonSerialized] private string resolvedFilePath;

        public SourceKind Kind => sourceKind;
        public OnnxModelAsset Asset => asset;
        public string RelativePath => relativePath;
        public string Sha256 => sha256;
        public string SourceName => sourceKind == SourceKind.Asset
            ? (asset != null ? asset.name : "Unassigned ONNX asset")
            : (string.IsNullOrWhiteSpace(relativePath) ? "Unassigned ONNX file" : Path.GetFileName(relativePath));
        public bool IsAvailable => sourceKind == SourceKind.Asset
            ? asset != null && asset.HasResolvableData
            : TryResolveFile(out _);
        public bool HasInspectedMetadata => sourceKind == SourceKind.Asset
            ? asset != null && asset.MetadataState == OnnxModelAsset.MetadataInspectionState.Succeeded
            : metadataInspected;
        public bool IsMetadataStale
        {
            get
            {
                if (sourceKind == SourceKind.Asset) return false;
                if (!TryResolveFile(out string path) || !metadataInspected) return true;
                var file = new FileInfo(path);
                return file.Length != cachedFileSize || file.LastWriteTimeUtc.Ticks != cachedWriteTimeUtcTicks;
            }
        }
        public IReadOnlyList<OnnxModelAsset.TensorInfo> Inputs => sourceKind == SourceKind.Asset
            ? asset?.Inputs ?? Array.Empty<OnnxModelAsset.TensorInfo>()
            : inputs;
        public IReadOnlyList<OnnxModelAsset.TensorInfo> Outputs => sourceKind == SourceKind.Asset
            ? asset?.Outputs ?? Array.Empty<OnnxModelAsset.TensorInfo>()
            : outputs;
        public OnnxModelAsset ImportedAsset => sourceKind == SourceKind.Asset ? asset : null;

        public string ResolveModelPath()
        {
            if (sourceKind == SourceKind.Asset) return asset?.ResolveModelPath();
            return TryResolveFile(out string path) ? path : null;
        }

        private bool TryResolveFile(out string resolved)
        {
            resolved = null;
            if (string.IsNullOrWhiteSpace(relativePath) || Path.IsPathRooted(relativePath)) return false;
            string normalized = relativePath.Replace('\\', '/').TrimStart('/');
            if (normalized.Split('/').Contains("..")) return false;
            if (!string.IsNullOrWhiteSpace(resolvedFilePath))
            {
                resolved = resolvedFilePath;
                return File.Exists(resolved);
            }
#if UNITY_EDITOR
            OnnxSettings settings = OnnxSettings.Load();
            string root = settings != null ? settings.ModelStorageRoot : "Assets/StreamingAssets/KitsuMateModels";
            string absoluteRoot = Path.GetFullPath(Path.Combine(Directory.GetCurrentDirectory(), root));
#else
            string absoluteRoot = Path.GetFullPath(Path.Combine(Application.streamingAssetsPath, "KitsuMateModels"));
#endif
            string candidate = Path.GetFullPath(Path.Combine(absoluteRoot, normalized));
            string prefix = absoluteRoot.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar) + Path.DirectorySeparatorChar;
            if (!candidate.StartsWith(prefix, StringComparison.OrdinalIgnoreCase)) return false;
            if (!File.Exists(candidate)) return false;
            resolvedFilePath = candidate;
            resolved = candidate;
            return true;
        }

#if UNITY_EDITOR
        public void Clear()
        {
            sourceKind = SourceKind.Asset;
            asset = null;
            relativePath = string.Empty;
            resolvedFilePath = null;
            ClearFileCache();
        }

        public void ConfigureAsset(OnnxModelAsset value)
        {
            sourceKind = SourceKind.Asset;
            asset = value;
            relativePath = string.Empty;
            resolvedFilePath = null;
            ClearFileCache();
        }

        public void ConfigureFile(string modelRootRelativePath, string hash,
            IEnumerable<OnnxModelAsset.TensorInfo> inspectedInputs,
            IEnumerable<OnnxModelAsset.TensorInfo> inspectedOutputs)
        {
            if (string.IsNullOrWhiteSpace(modelRootRelativePath) || Path.IsPathRooted(modelRootRelativePath))
                throw new ArgumentException("ONNX file path must be relative to the configured model root.", nameof(modelRootRelativePath));
            string normalized = modelRootRelativePath.Replace('\\', '/').TrimStart('/');
            if (normalized.Split('/').Contains("..")) throw new ArgumentException("ONNX file path cannot contain '..'.", nameof(modelRootRelativePath));
            sourceKind = SourceKind.File;
            asset = null;
            relativePath = normalized;
            resolvedFilePath = null;
            sha256 = hash ?? string.Empty;
            inputs.Clear();
            outputs.Clear();
            if (inspectedInputs != null) inputs.AddRange(inspectedInputs);
            if (inspectedOutputs != null) outputs.AddRange(inspectedOutputs);
            metadataInspected = outputs.Count > 0;
            if (TryResolveFile(out string path))
            {
                var file = new FileInfo(path);
                cachedFileSize = file.Length;
                cachedWriteTimeUtcTicks = file.LastWriteTimeUtc.Ticks;
            }
        }

        private void ClearFileCache()
        {
            sha256 = string.Empty;
            cachedFileSize = 0;
            cachedWriteTimeUtcTicks = 0;
            metadataInspected = false;
            inputs.Clear();
            outputs.Clear();
        }
#endif
    }
}

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
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
        public enum FileRoot { Absolute, PersistentData }

        [SerializeField] private SourceKind sourceKind;
        [SerializeField] private OnnxModelAsset asset;
        [SerializeField] private string filePath;
        [SerializeField] private FileRoot fileRoot;
        [SerializeField, HideInInspector] private string sha256;
        [SerializeField, HideInInspector] private long cachedFileSize;
        [SerializeField, HideInInspector] private long cachedWriteTimeUtcTicks;
        [SerializeField, HideInInspector] private bool metadataInspected;
        [SerializeField, HideInInspector] private List<OnnxModelAsset.TensorInfo> inputs = new();
        [SerializeField, HideInInspector] private List<OnnxModelAsset.TensorInfo> outputs = new();
        [NonSerialized] private string preparedRuntimePath;


        public SourceKind Kind => sourceKind;
        public OnnxModelAsset Asset => asset;
        public string FilePath => filePath;
        public FileRoot Root => fileRoot;
        public string Sha256 => sha256;
        public string SourceName => sourceKind == SourceKind.Asset
            ? (asset != null ? asset.name : "Unassigned ONNX asset")
            : (string.IsNullOrWhiteSpace(filePath) ? "Unassigned ONNX file" : Path.GetFileName(filePath));
        public bool IsAvailable => sourceKind == SourceKind.Asset
            ? asset != null && asset.HasResolvableData
            : (!string.IsNullOrWhiteSpace(preparedRuntimePath) && File.Exists(preparedRuntimePath)) || TryResolveFile(out _);
        public bool HasInspectedMetadata => sourceKind == SourceKind.Asset
            ? asset != null && asset.MetadataState == OnnxModelAsset.MetadataInspectionState.Succeeded
            : metadataInspected;
        public bool IsMetadataStale
        {
            get
            {
                if (sourceKind == SourceKind.Asset) return false;
                string path = ResolveModelPath();
                if (string.IsNullOrWhiteSpace(path) || !metadataInspected) return true;
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
            if (!string.IsNullOrWhiteSpace(preparedRuntimePath) && File.Exists(preparedRuntimePath)) return preparedRuntimePath;
            return TryResolveFile(out string path) ? path : null;
        }

        internal Task PrepareForRuntimeAsync(CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (sourceKind == SourceKind.File && !TryResolveFile(out preparedRuntimePath))
                throw new OnnxModelPreparationException($"ONNX model '{filePath}' was not found.");
            return Task.CompletedTask;
        }

        private bool TryResolveFile(out string resolved)
        {
            resolved = null;
            if (string.IsNullOrWhiteSpace(filePath)) return false;
            try
            {
                resolved = fileRoot == FileRoot.PersistentData
                    ? Download.ModelDownloadPaths.Child(Application.persistentDataPath, filePath)
                    : Path.IsPathRooted(filePath) ? filePath : null;
            }
            catch (IOException) { return false; }
            catch (ArgumentException) { return false; }
            if (resolved == null || !File.Exists(resolved)) { resolved = null; return false; }
            return true;
        }

        public void Clear()
        {
            sourceKind = SourceKind.Asset;
            asset = null;
            filePath = string.Empty;
            fileRoot = FileRoot.Absolute;
            ClearFileCache();
        }

        public void ConfigureAsset(OnnxModelAsset value)
        {
            sourceKind = SourceKind.Asset;
            asset = value;
            filePath = string.Empty;
            ClearFileCache();
        }

        /// <summary>Uses a caller-owned downloaded file; never falls back to bundled model storage.</summary>
        public void ConfigureFile(string absolutePath, string hash,
            IEnumerable<OnnxModelAsset.TensorInfo> inspectedInputs,
            IEnumerable<OnnxModelAsset.TensorInfo> inspectedOutputs)
        {
            if (string.IsNullOrWhiteSpace(absolutePath) || !Path.IsPathRooted(absolutePath))
                throw new ArgumentException("Downloaded model path must be absolute.", nameof(absolutePath));
            Clear();
            sourceKind = SourceKind.File;
            filePath = Path.GetFullPath(absolutePath);
            string persistentRoot = Path.GetFullPath(Application.persistentDataPath).TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar) + Path.DirectorySeparatorChar;
            var comparison = Application.platform == RuntimePlatform.WindowsEditor || Application.platform == RuntimePlatform.WindowsPlayer
                ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal;
            if (filePath.StartsWith(persistentRoot, comparison))
            {
                fileRoot = FileRoot.PersistentData;
                filePath = filePath.Substring(persistentRoot.Length).Replace('\\', '/');
            }
            sha256 = hash ?? string.Empty;
            if (inspectedInputs != null) inputs.AddRange(inspectedInputs);
            if (inspectedOutputs != null) outputs.AddRange(inspectedOutputs);
            metadataInspected = outputs.Count > 0;
            if (File.Exists(absolutePath))
            {
                var file = new FileInfo(absolutePath);
                cachedFileSize = file.Length;
                cachedWriteTimeUtcTicks = file.LastWriteTimeUtc.Ticks;
            }
        }

        private void ClearFileCache()
        {
            preparedRuntimePath = null;
            sha256 = string.Empty;
            cachedFileSize = 0;
            cachedWriteTimeUtcTicks = 0;
            metadataInspected = false;
            inputs.Clear();
            outputs.Clear();
        }
    }

}

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;
#if UNITY_ANDROID && !UNITY_EDITOR
using UnityEngine.Networking;
#endif

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
        public enum SourceKind { Asset, File, DownloadedFile }

        [SerializeField] private SourceKind sourceKind;
        [SerializeField] private OnnxModelAsset asset;
        [SerializeField] private string relativePath;
        [SerializeField, HideInInspector] private string sha256;
        [SerializeField, HideInInspector] private long cachedFileSize;
        [SerializeField, HideInInspector] private long cachedWriteTimeUtcTicks;
        [SerializeField, HideInInspector] private bool metadataInspected;
        [SerializeField, HideInInspector] private List<OnnxModelAsset.TensorInfo> inputs = new();
        [SerializeField, HideInInspector] private List<OnnxModelAsset.TensorInfo> outputs = new();
        [NonSerialized] private string preparedRuntimePath;
#if UNITY_ANDROID && !UNITY_EDITOR
        private static readonly AndroidModelStager AndroidStager = new(new UnityWebRequestModelContentFetcher());
#endif

        public SourceKind Kind => sourceKind;
        public OnnxModelAsset Asset => asset;
        public string RelativePath => relativePath;
        public string Sha256 => sha256;
        public string SourceName => sourceKind == SourceKind.Asset
            ? (asset != null ? asset.name : "Unassigned ONNX asset")
            : (string.IsNullOrWhiteSpace(relativePath) ? "Unassigned ONNX file" : Path.GetFileName(relativePath));
        public bool IsAvailable => sourceKind == SourceKind.Asset
            ? asset != null && asset.HasResolvableData
            : (!string.IsNullOrWhiteSpace(preparedRuntimePath) && File.Exists(preparedRuntimePath)) || TryResolveFile(null, out _);
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
            return TryResolveFile(null, out string path) ? path : null;
        }

        internal async Task PrepareForRuntimeAsync(OnnxRuntimeEnvironment environment, CancellationToken cancellationToken)
        {
            if (environment == null) throw new ArgumentNullException(nameof(environment));
            if (sourceKind == SourceKind.DownloadedFile)
            {
                cancellationToken.ThrowIfCancellationRequested();
                if (!TryResolveFile(environment, out preparedRuntimePath))
                    throw new OnnxModelPreparationException($"Downloaded model '{relativePath}' is not installed.");
                return;
            }
            if (sourceKind != SourceKind.File) return;
#if UNITY_ANDROID && !UNITY_EDITOR
            preparedRuntimePath = await AndroidStager.StageAsync(relativePath, sha256, environment, cancellationToken);
#else
            cancellationToken.ThrowIfCancellationRequested();
            if (!TryResolveFile(environment, out preparedRuntimePath))
                throw new OnnxModelPreparationException($"ONNX model '{relativePath}' was not found in the configured model root.");
#endif
        }

        private bool TryResolveFile(OnnxRuntimeEnvironment environment, out string resolved)
        {
            resolved = null;
            if (sourceKind == SourceKind.DownloadedFile)
            {
                if (string.IsNullOrWhiteSpace(relativePath) || !Path.IsPathRooted(relativePath) || !File.Exists(relativePath)) return false;
                resolved = relativePath;
                return true;
            }
            if (string.IsNullOrWhiteSpace(relativePath) || Path.IsPathRooted(relativePath)) return false;
            string normalized = relativePath.Replace('\\', '/').TrimStart('/');
            if (normalized.Split('/').Contains("..")) return false;
#if UNITY_EDITOR
            string root = environment?.ModelStorageRoot ?? OnnxSettings.Load()?.ModelStorageRoot ?? OnnxSettings.DefaultModelStorageRoot;
            string absoluteRoot = Path.GetFullPath(Path.Combine(Directory.GetCurrentDirectory(), root));
#elif UNITY_ANDROID
            string persistentDataPath = environment?.PersistentDataPath;
            if (string.IsNullOrWhiteSpace(persistentDataPath)) return false;
            string absoluteRoot = Path.GetFullPath(Path.Combine(persistentDataPath, "KitsuMateModels"));
#else
            string streamingAssetsPath = environment?.StreamingAssetsPath;
            if (string.IsNullOrWhiteSpace(streamingAssetsPath)) return false;
            string absoluteRoot = Path.GetFullPath(Path.Combine(streamingAssetsPath, "KitsuMateModels"));
#endif
            string candidate = Path.GetFullPath(Path.Combine(absoluteRoot, normalized));
            string prefix = absoluteRoot.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar) + Path.DirectorySeparatorChar;
            if (!candidate.StartsWith(prefix, StringComparison.OrdinalIgnoreCase)) return false;
            if (!File.Exists(candidate)) return false;
            resolved = candidate;
            return true;
        }

        public void Clear()
        {
            sourceKind = SourceKind.Asset;
            asset = null;
            relativePath = string.Empty;
            ClearFileCache();
        }

        public void ConfigureAsset(OnnxModelAsset value)
        {
            sourceKind = SourceKind.Asset;
            asset = value;
            relativePath = string.Empty;
            ClearFileCache();
        }

        /// <summary>Uses a caller-owned downloaded file; never falls back to bundled model storage.</summary>
        public void ConfigureDownloadedFile(string absolutePath, string hash,
            IEnumerable<OnnxModelAsset.TensorInfo> inspectedInputs,
            IEnumerable<OnnxModelAsset.TensorInfo> inspectedOutputs)
        {
            if (string.IsNullOrWhiteSpace(absolutePath) || !Path.IsPathRooted(absolutePath))
                throw new ArgumentException("Downloaded model path must be absolute.", nameof(absolutePath));
            Clear();
            sourceKind = SourceKind.DownloadedFile;
            relativePath = Path.GetFullPath(absolutePath);
            sha256 = hash ?? string.Empty;
            if (inspectedInputs != null) inputs.AddRange(inspectedInputs);
            if (inspectedOutputs != null) outputs.AddRange(inspectedOutputs);
            metadataInspected = outputs.Count > 0;
            if (File.Exists(relativePath))
            {
                var file = new FileInfo(relativePath);
                cachedFileSize = file.Length;
                cachedWriteTimeUtcTicks = file.LastWriteTimeUtc.Ticks;
            }
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
            sha256 = hash ?? string.Empty;
            inputs.Clear();
            outputs.Clear();
            if (inspectedInputs != null) inputs.AddRange(inspectedInputs);
            if (inspectedOutputs != null) outputs.AddRange(inspectedOutputs);
            metadataInspected = outputs.Count > 0;
            if (TryResolveFile(null, out string path))
            {
                var file = new FileInfo(path);
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

#if UNITY_ANDROID && !UNITY_EDITOR
    internal sealed class UnityWebRequestModelContentFetcher : IOnnxModelContentFetcher
    {
        public async Task FetchAsync(string sourceUri, string destinationPath, CancellationToken cancellationToken)
        {
            using var request = UnityWebRequest.Get(sourceUri);
            request.downloadHandler = new DownloadHandlerFile(destinationPath) { removeFileOnAbort = true };
            UnityWebRequestAsyncOperation operation = request.SendWebRequest();
            while (!operation.isDone)
            {
                if (cancellationToken.IsCancellationRequested)
                {
                    request.Abort();
                    cancellationToken.ThrowIfCancellationRequested();
                }
                await Task.Yield();
            }
            if (request.result != UnityWebRequest.Result.Success)
                throw new IOException($"Failed to fetch Android model content: {request.error}");
        }
    }
#endif
}

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using KitsuMate.Onnx.Download;
using UnityEngine;
using UnityEngine.Networking;

namespace KitsuMate.Onnx
{
    /// <summary>Provides ordinary files to native backends when StreamingAssets is packaged in an archive.</summary>
    internal static class StreamingAssetFiles
    {
        private static readonly SemaphoreSlim CopyGate = new(1, 1);
        private static string cleanedBuild;

        internal static async Task<string> PrepareOnnxAsync(string relativePath, CancellationToken cancellationToken)
        {
            string graph = await PrepareAsync(relativePath, cancellationToken);
            var metadata = OnnxLightweightMetadataReader.Read(graph);
            var locations = new HashSet<string>(metadata.ExternalData.Select(item => item.Location), StringComparer.Ordinal);
            foreach (string location in locations)
            {
                string safeLocation = HuggingFaceModelRepository.SafeRelativePath(location);
                string relativeSidecar = Path.GetDirectoryName(relativePath)?.Replace('\\', '/') is string directory && directory.Length > 0
                    ? directory + "/" + safeLocation : safeLocation;
                await PrepareAsync(relativeSidecar, cancellationToken);
            }
            foreach (var reference in metadata.ExternalData)
                OnnxLightweightMetadataReader.ResolveExternalDataPath(graph, reference);
            return graph;
        }

        internal static async Task<string> PrepareAsync(string relativePath, CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();
            string safePath = HuggingFaceModelRepository.SafeRelativePath(relativePath);
            string sourceRoot = Application.streamingAssetsPath;
            if (!sourceRoot.Contains("://"))
            {
                string source = ModelDownloadPaths.Child(sourceRoot, safePath);
                if (!File.Exists(source)) throw new OnnxModelPreparationException($"Streaming asset '{safePath}' was not found.");
                return source;
            }
            if (Application.platform == RuntimePlatform.WebGLPlayer)
                throw new PlatformNotSupportedException("Native ONNX file sessions are unavailable on WebGL.");

            string build = string.IsNullOrWhiteSpace(Application.buildGUID) ? "unknown-build" : Application.buildGUID;
            string cacheBase = ModelDownloadPaths.Child(Application.persistentDataPath, "KitsuMateStreamingModels");
            string cacheRoot = ModelDownloadPaths.Child(cacheBase, build);
            string destination = ModelDownloadPaths.Child(cacheRoot, safePath);
            await CopyGate.WaitAsync(cancellationToken);
            try
            {
                if (!File.Exists(destination))
                {
                    string url = sourceRoot.TrimEnd('/') + "/" + string.Join("/", safePath.Split('/').Select(Uri.EscapeDataString));
                    await CopyFromUrlAsync(url, destination, cancellationToken);
                }
                if (cleanedBuild != build)
                {
                    DeleteOldBuildCaches(cacheBase, build);
                    cleanedBuild = build;
                }
                return destination;
            }
            finally { CopyGate.Release(); }
        }

        private static void DeleteOldBuildCaches(string cacheBase, string currentBuild)
        {
            if (!Directory.Exists(cacheBase)) return;
            foreach (string directory in Directory.GetDirectories(cacheBase))
            {
                if (string.Equals(Path.GetFileName(directory), currentBuild, StringComparison.Ordinal)) continue;
                try
                {
                    if ((File.GetAttributes(directory) & FileAttributes.ReparsePoint) != 0) continue;
                    Directory.Delete(directory, true);
                }
                catch (Exception exception) when (exception is IOException || exception is UnauthorizedAccessException)
                {
                    Debug.LogWarning($"Could not remove old bundled-model cache '{directory}': {exception.Message}");
                }
            }
        }

        internal static async Task CopyFromUrlAsync(string url, string destination, CancellationToken cancellationToken)
        {
            await Awaitable.MainThreadAsync();
            Directory.CreateDirectory(Path.GetDirectoryName(destination) ?? throw new InvalidDataException("Copy destination has no directory."));
            string partial = destination + ".partial";
            try
            {
                if (File.Exists(partial)) File.Delete(partial);
                using var request = new UnityWebRequest(url, UnityWebRequest.kHttpVerbGET,
                    new DownloadHandlerFile(partial), null);
                var operation = request.SendWebRequest();
                while (!operation.isDone)
                {
                    if (cancellationToken.IsCancellationRequested)
                    {
                        request.Abort();
                        cancellationToken.ThrowIfCancellationRequested();
                    }
                    await Awaitable.NextFrameAsync();
                }
                cancellationToken.ThrowIfCancellationRequested();
                if (request.result != UnityWebRequest.Result.Success)
                    throw new OnnxModelPreparationException($"Could not copy streaming asset: {request.error}");
                if (!File.Exists(partial))
                    throw new OnnxModelPreparationException("Streaming asset copy produced no file.");
                File.Move(partial, destination);
            }
            finally { if (File.Exists(partial)) File.Delete(partial); }
        }
    }
}

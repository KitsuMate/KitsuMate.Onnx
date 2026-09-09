using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using UnityEditor;
using KitsuMate.Onnx.Download;
using UnityEngine;

namespace KitsuMate.Onnx.Editor.Download
{
    public sealed class ModelDownloadResult
    {
        private readonly DiscoveredFile[] files;
        private readonly IReadOnlyDictionary<string, string> projectPaths;
        private readonly string modelRoot;
        public DownloadedModel Downloaded { get; }

        public ModelIdentity Identity { get; }
        public IReadOnlyDictionary<string, string> ProjectPaths => projectPaths;
        public IReadOnlyList<string> Capabilities { get; } = Array.Empty<string>();

        internal ModelDownloadResult(ModelIdentity identity, DiscoveredFile[] selectedFiles,
            IReadOnlyDictionary<string, string> paths, string root, DownloadedModel downloaded)
        {
            Identity = identity;
            files = selectedFiles;
            projectPaths = paths;
            modelRoot = root.TrimEnd('/', '\\');
            Downloaded = downloaded;
        }

        public string GetProjectPath(string role)
        {
            if (!projectPaths.TryGetValue(role, out string path))
                throw new KeyNotFoundException($"Downloaded model does not contain the required '{role}' file.");
            return path;
        }

        public T LoadAsset<T>(string role) where T : UnityEngine.Object
        {
            string path = GetProjectPath(role);
            T asset = AssetDatabase.LoadAssetAtPath<T>(path);
            if (asset == null)
                throw new InvalidOperationException(
                    $"Downloaded '{role}' file was not imported as {typeof(T).Name} at '{path}'.");
            return asset;
        }

        public void ConfigureModel(OnnxModelReference target, string role)
        {
            if (target == null) throw new ArgumentNullException(nameof(target));
            DiscoveredFile file = FindFile(role);
            string path = GetProjectPath(role);
            string prefix = modelRoot + "/";
            if (!path.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
                throw new InvalidOperationException($"Downloaded model '{path}' is outside '{modelRoot}'.");
            target.ConfigureFile(path.Substring(prefix.Length), file.Sha256, file.Inputs, file.Outputs);
        }

        public void ApplyMetadata(StandardModelSet target)
        {
            if (target == null) throw new ArgumentNullException(nameof(target));
            target.SetDownloadMetadata(Identity, Capabilities);
        }

        public IReadOnlyList<string> GetInputNames(string role)
        {
            return FindFile(role).Inputs.Select(input => input.Name).ToArray();
        }

        public IReadOnlyList<string> GetOutputNames(string role)
        {
            return FindFile(role).Outputs.Select(output => output.Name).ToArray();
        }

        public void RequireGraph(string role, IEnumerable<string> inputs = null,
            IEnumerable<string> outputs = null)
        {
            IReadOnlyList<string> availableInputs = GetInputNames(role);
            IReadOnlyList<string> availableOutputs = GetOutputNames(role);
            foreach (string input in inputs ?? Array.Empty<string>())
                if (!availableInputs.Contains(input))
                    throw new InvalidDataException(
                        $"Downloaded '{role}' graph is missing input '{input}'.");
            foreach (string output in outputs ?? Array.Empty<string>())
                if (!availableOutputs.Contains(output))
                    throw new InvalidDataException(
                        $"Downloaded '{role}' graph is missing output '{output}'.");
        }

        private DiscoveredFile FindFile(string role)
        {
            DiscoveredFile file = files.FirstOrDefault(candidate =>
                string.Equals(candidate.Role, role, StringComparison.Ordinal));
            if (file == null)
                throw new KeyNotFoundException($"Downloaded model does not contain the required '{role}' file.");
            return file;
        }
    }

    /// <summary>Imports files produced by the runtime downloader into a Unity project.</summary>
    public static class EditorModelDownloader
    {
        public static Task<ModelDownloadResult> DownloadAsync(ModelDownloadRequest request,
            IReadOnlyDictionary<string, string> selection, string token = null, IProgress<float> progress = null,
            CancellationToken cancellationToken = default) => DownloadAsync(request, selection, false, token, progress, cancellationToken);

        public static Task<ModelDownloadResult> DownloadForSentisAsync(ModelDownloadRequest request,
            IReadOnlyDictionary<string, string> selection, string token = null, IProgress<float> progress = null,
            CancellationToken cancellationToken = default) => DownloadAsync(request, selection, true, token, progress, cancellationToken);

        private static async Task<ModelDownloadResult> DownloadAsync(ModelDownloadRequest request,
            IReadOnlyDictionary<string, string> selection, bool importWithSentis, string token,
            IProgress<float> progress, CancellationToken cancellationToken)
        {
            var repository = await Task.Run(() => HuggingFaceModelRepository.ScanAsync(request, token, cancellationToken), cancellationToken);
            var artifacts = HuggingFaceModelRepository.SelectArtifacts(repository, selection);
            string modelRoot = ModelRoot();
            string directory = Path.Combine(modelRoot, repository.Owner, repository.Name).Replace('\\', '/');
            string configuredCache = Environment.GetEnvironmentVariable("KITSUMATE_MODEL_CACHE");
            string cache = string.IsNullOrWhiteSpace(configuredCache)
                ? Path.GetFullPath("Library/KitsuMateModelCache") : Path.GetFullPath(configuredCache);
            var result = await Task.Run(() => ModelDownloader.DownloadAsync(repository, selection, Path.GetFullPath(directory), cache,
                token, progress, cancellationToken), cancellationToken);
            var paths = result.Files.ToDictionary(file => file.Role,
                file => (directory + "/" + file.Path).Replace('\\', '/'), StringComparer.Ordinal);
            foreach (var file in result.Files)
                SetImporter(paths[file.Role], file, importWithSentis || artifacts.Any(artifact =>
                    artifact.Model.Path == file.Path && HuggingFaceModelRepository.IsSentisArtifact(artifact.Type)));
            AssetDatabase.Refresh(ImportAssetOptions.ForceSynchronousImport);
            if (!importWithSentis) ValidateWithOnnxRuntime(artifacts, paths);
            return new ModelDownloadResult(result.Identity, result.Files.ToArray(), paths, modelRoot, result);
        }

        internal static string ModelRoot() => OnnxSettings.Load()?.ModelStorageRoot ?? "Assets/StreamingAssets/KitsuMateModels";
        private static void SetImporter(string projectPath, DiscoveredFile file, bool importWithSentis)
        {
            if (file.Path.EndsWith(".onnx", StringComparison.OrdinalIgnoreCase))
            {
                if (importWithSentis) AssetDatabase.ClearImporterOverride(projectPath);
                else OnnxImporterAssignment.AssignFrameworkImporter(projectPath);
                return;
            }

            if (file.Path.EndsWith(".onnx_data", StringComparison.OrdinalIgnoreCase))
                OnnxImporterAssignment.AssignFrameworkImporter(projectPath);
        }

        private static void ValidateWithOnnxRuntime(IEnumerable<DiscoveredArtifact> artifacts,
            IReadOnlyDictionary<string, string> installed)
        {
            OnnxBackend backend = CreateOnnxRuntimeBackend();
            try
            {
                foreach (DiscoveredArtifact artifact in artifacts)
                {
                    string path = Path.GetFullPath(Path.Combine(Directory.GetCurrentDirectory(),
                        installed[artifact.Role]));
                    using IOnnxSession session = backend.CreateSession(path);
                }
            }
            finally
            {
                backend.Dispose();
                UnityEngine.Object.DestroyImmediate(backend);
            }
        }

        private static OnnxBackend CreateOnnxRuntimeBackend()
        {
            OnnxBackend configured = OnnxSettings.Load()?.DefaultBackend;
            if (configured != null &&
                string.Equals(configured.BackendId, "onnxruntime", StringComparison.OrdinalIgnoreCase))
                return UnityEngine.Object.Instantiate(configured);

            foreach (Type type in TypeCache.GetTypesDerivedFrom<OnnxBackend>())
            {
                if (type.IsAbstract) continue;
                var candidate = ScriptableObject.CreateInstance(type) as OnnxBackend;
                if (candidate != null &&
                    string.Equals(candidate.BackendId, "onnxruntime", StringComparison.OrdinalIgnoreCase))
                    return candidate;
                if (candidate != null) UnityEngine.Object.DestroyImmediate(candidate);
            }
            throw new InvalidOperationException(
                "ONNX Runtime is required to validate downloaded ONNX artifacts before assignment.");
        }

    }
}

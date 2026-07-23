using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Security.Cryptography;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using UnityEditor;
using UnityEngine;

namespace KitsuMate.Onnx.Editor.Download
{
    public sealed class ModelDownloadResult
    {
        private readonly DiscoveredFile[] files;
        private readonly IReadOnlyDictionary<string, string> projectPaths;
        private readonly string modelRoot;

        public ModelIdentity Identity { get; }
        public IReadOnlyDictionary<string, string> ProjectPaths => projectPaths;
        public IReadOnlyList<string> Capabilities { get; } = Array.Empty<string>();

        internal ModelDownloadResult(ModelIdentity identity, DiscoveredFile[] selectedFiles,
            IReadOnlyDictionary<string, string> paths, string root)
        {
            Identity = identity;
            files = selectedFiles;
            projectPaths = paths;
            modelRoot = root.TrimEnd('/', '\\');
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

    public static class ModelDownloader
    {
        private static readonly HttpClient Client = new();

        public static async Task<(string Revision,
            IReadOnlyDictionary<string, IReadOnlyList<string>> Artifacts)> GetArtifactsAsync(
            ModelDownloadRequest request, string token = null, CancellationToken cancellationToken = default)
        {
            DiscoveredRepository repository = await ScanAsync(request, token, cancellationToken);
            var artifacts = repository.Artifacts.ToDictionary(
                pair => pair.Key,
                pair => (IReadOnlyList<string>)pair.Value.Select(artifact => artifact.Model.Path).ToArray(),
                StringComparer.Ordinal);
            return (repository.Revision, artifacts);
        }

        public static Task<ModelDownloadResult> DownloadAsync(ModelDownloadRequest request,
            IReadOnlyDictionary<string, string> artifacts, string token = null, IProgress<float> progress = null,
            CancellationToken cancellationToken = default)
        {
            return DownloadAsync(request, artifacts, false, token, progress, cancellationToken);
        }

        public static Task<ModelDownloadResult> DownloadForSentisAsync(ModelDownloadRequest request,
            IReadOnlyDictionary<string, string> artifacts, string token = null, IProgress<float> progress = null,
            CancellationToken cancellationToken = default)
        {
            return DownloadAsync(request, artifacts, true, token, progress, cancellationToken);
        }

        private static async Task<ModelDownloadResult> DownloadAsync(ModelDownloadRequest request,
            IReadOnlyDictionary<string, string> selection, bool importWithSentis, string token,
            IProgress<float> progress, CancellationToken cancellationToken)
        {
            if (request == null) throw new ArgumentNullException(nameof(request));
            if (selection == null) throw new ArgumentNullException(nameof(selection));

            DiscoveredRepository repository = await ScanAsync(request, token, cancellationToken);
            DiscoveredArtifact[] artifacts = SelectArtifacts(repository, selection);
            DiscoveredFile[] files = artifacts.SelectMany(artifact => artifact.Files)
                .Concat(repository.CommonFiles)
                .GroupBy(file => file.Path, StringComparer.OrdinalIgnoreCase)
                .Select(group => group.First())
                .ToArray();
            var sentisEligible = new HashSet<string>(
                artifacts.Where(artifact => IsSentisArtifact(artifact.Type))
                    .Select(artifact => artifact.Model.Path),
                StringComparer.OrdinalIgnoreCase);

            string modelRoot = ModelRoot();
            SafeProjectDirectory(modelRoot);
            string projectRoot = Path.GetFullPath(Directory.GetCurrentDirectory());
            string cacheRoot = CacheRoot(projectRoot);
            Directory.CreateDirectory(cacheRoot);

            var installed = new Dictionary<string, string>(StringComparer.Ordinal);
            long total = files.Sum(file => Math.Max(0, file.Size));
            long completed = 0;

            foreach (DiscoveredFile file in files)
            {
                cancellationToken.ThrowIfCancellationRequested();
                ValidateFile(file);
                string relative = SafeRelativePath(file.Path);
                string projectPath = Path.Combine(modelRoot, repository.Owner, repository.Name, relative)
                    .Replace('\\', '/');
                string destination = SafeChildPath(projectRoot, projectPath);
                string cache = Path.Combine(cacheRoot, file.Sha256.ToLowerInvariant());

                if (!await HasExpectedHashAsync(destination, file.Sha256, cancellationToken))
                {
                    if (!await HasExpectedHashAsync(cache, file.Sha256, cancellationToken))
                        await DownloadFileAsync(FileUrl(request, repository.Revision, relative), cache, file, token,
                            cancellationToken);
                    CopyVerifiedFile(cache, destination);
                }

                SetImporter(projectPath, file, importWithSentis || sentisEligible.Contains(file.Path));
                completed += Math.Max(0, file.Size);
                progress?.Report(total > 0
                    ? Math.Min(1f, (float)completed / total)
                    : (float)(installed.Count + 1) / files.Length);
                installed[file.Role] = projectPath;
            }

            AssetDatabase.Refresh(ImportAssetOptions.ForceSynchronousImport);
            InspectModels(artifacts, installed);
            if (!importWithSentis) ValidateWithOnnxRuntime(artifacts, installed);
            var identity = new ModelIdentity(repository.Family, repository.Name, repository.Revision,
                ContentHash(files));
            return new ModelDownloadResult(identity, files, installed, modelRoot);
        }

        internal static async Task<DiscoveredRepository> ScanAsync(ModelDownloadRequest request, string token,
            CancellationToken cancellationToken)
        {
            if (request == null) throw new ArgumentNullException(nameof(request));
            string[] repository = RepositoryParts(request.Repository);
            string url =
                $"https://huggingface.co/api/models/{Uri.EscapeDataString(repository[0])}/{Uri.EscapeDataString(repository[1])}/revision/{Uri.EscapeDataString(request.Revision)}?blobs=true";
            using var message = new HttpRequestMessage(HttpMethod.Get, url);
            AddToken(message, token);
            using HttpResponseMessage response = await Client.SendAsync(message, cancellationToken);
            response.EnsureSuccessStatusCode();
            HfModelInfo info = JsonUtility.FromJson<HfModelInfo>(await response.Content.ReadAsStringAsync());
            if (info == null || string.IsNullOrWhiteSpace(info.sha))
                throw new InvalidDataException($"Could not resolve '{request.Repository}@{request.Revision}'.");

            var byPath = (info.siblings ?? Array.Empty<HfSibling>())
                .Where(file => file != null && !string.IsNullOrWhiteSpace(file.rfilename))
                .ToDictionary(file => file.rfilename.Replace('\\', '/'), StringComparer.OrdinalIgnoreCase);
            HfSibling[] onnxFiles = byPath.Values
                .Where(file => file.rfilename.StartsWith("onnx/", StringComparison.OrdinalIgnoreCase) &&
                    file.rfilename.EndsWith(".onnx", StringComparison.OrdinalIgnoreCase))
                .OrderBy(file => file.rfilename, StringComparer.OrdinalIgnoreCase)
                .ToArray();
            if (onnxFiles.Length == 0)
                throw new InvalidDataException($"Repository '{request.Repository}' has no ONNX files under onnx/.");

            Dictionary<string, DiscoveredArtifact[]> artifacts;
            string[] requiredRoles;
            if (IsChatterbox(request, onnxFiles))
            {
                artifacts = await DiscoverChatterboxAsync(request, info.sha, byPath, onnxFiles, token,
                    cancellationToken);
                requiredRoles = new[] { "speech-encoder", "embed-tokens", "language-model", "conditional-decoder" };
            }
            else if (onnxFiles.Any(file =>
                FileStem(file).StartsWith("encoder_model", StringComparison.OrdinalIgnoreCase)))
            {
                artifacts = await DiscoverWhisperAsync(request, info.sha, byPath, onnxFiles, token,
                    cancellationToken);
                requiredRoles = new[] { "encoder", "decoder" };
            }
            else
            {
                artifacts = await DiscoverModelsAsync(request, info.sha, byPath, onnxFiles, token,
                    cancellationToken);
                requiredRoles = artifacts.Keys.ToArray();
            }

            if (requiredRoles.Any(role => !artifacts.TryGetValue(role, out DiscoveredArtifact[] choices) ||
                    choices.Length == 0))
                throw new InvalidDataException(
                    $"Repository '{request.Repository}' does not use a complete recognized ONNX layout.");

            DiscoveredFile[] common = await DiscoverCommonFilesAsync(request, info.sha, byPath, token,
                cancellationToken);
            return new DiscoveredRepository(repository[0], repository[1],
                string.IsNullOrWhiteSpace(request.ExpectedFamily) ? "onnx" : request.ExpectedFamily,
                info.sha, artifacts, common, requiredRoles);
        }

        internal static string ArtifactType(string path, string baseStem)
        {
            string stem = Path.GetFileNameWithoutExtension(path);
            if (!stem.StartsWith(baseStem, StringComparison.OrdinalIgnoreCase)) return null;
            string suffix = stem.Substring(baseStem.Length).TrimStart('_', '-');
            return string.IsNullOrWhiteSpace(suffix) ? "default" : suffix;
        }

        internal static string ArtifactLabel(DiscoveredArtifact artifact)
        {
            string type = string.Equals(artifact.Type, "default", StringComparison.OrdinalIgnoreCase)
                ? "Default"
                : artifact.Type;
            return $"{type} — {Path.GetFileName(artifact.Model.Path)}";
        }

        internal static bool IsSentisArtifact(string type)
        {
            return string.Equals(type, "default", StringComparison.OrdinalIgnoreCase) ||
                string.Equals(type, "fp32", StringComparison.OrdinalIgnoreCase);
        }

        internal static string ModelRoot()
        {
            OnnxSettings settings = OnnxSettings.Load();
            return settings != null ? settings.ModelStorageRoot : "Assets/StreamingAssets/KitsuMateModels";
        }

        internal static string SafeRelativePath(string path)
        {
            if (string.IsNullOrWhiteSpace(path) || Path.IsPathRooted(path))
                throw new InvalidDataException("Repository file paths must be relative.");
            string normalized = path.Replace('\\', '/').TrimStart('/');
            if (normalized.Split('/').Any(part => part == ".." || part.Length == 0))
                throw new InvalidDataException($"Repository contains an unsafe file path: '{path}'.");
            return normalized;
        }

        private static async Task<Dictionary<string, DiscoveredArtifact[]>> DiscoverWhisperAsync(
            ModelDownloadRequest request, string revision, Dictionary<string, HfSibling> byPath,
            HfSibling[] models, string token, CancellationToken cancellationToken)
        {
            var result = new Dictionary<string, List<DiscoveredArtifact>>(StringComparer.Ordinal);
            foreach (HfSibling model in models)
            {
                string stem = FileStem(model);
                string role;
                string baseStem;
                bool merged = false;
                if (stem.StartsWith("decoder_with_past_model", StringComparison.OrdinalIgnoreCase))
                    (role, baseStem) = ("decoder-with-past", "decoder_with_past_model");
                else if (stem.StartsWith("decoder_model_merged", StringComparison.OrdinalIgnoreCase))
                {
                    (role, baseStem) = ("decoder", "decoder_model_merged");
                    merged = true;
                }
                else if (stem.StartsWith("decoder_model", StringComparison.OrdinalIgnoreCase))
                    (role, baseStem) = ("decoder", "decoder_model");
                else if (stem.StartsWith("encoder_model", StringComparison.OrdinalIgnoreCase))
                    (role, baseStem) = ("encoder", "encoder_model");
                else if (stem.StartsWith("mel", StringComparison.OrdinalIgnoreCase))
                    (role, baseStem) = ("mel", "mel");
                else
                    continue;

                await AddArtifactAsync(result, request, revision, byPath, model, role, baseStem, merged, token,
                    cancellationToken);
            }
            return Finish(result);
        }

        private static async Task<Dictionary<string, DiscoveredArtifact[]>> DiscoverChatterboxAsync(
            ModelDownloadRequest request, string revision, Dictionary<string, HfSibling> byPath,
            HfSibling[] models, string token, CancellationToken cancellationToken)
        {
            var result = new Dictionary<string, List<DiscoveredArtifact>>(StringComparer.Ordinal);
            var roles = new[]
            {
                (Stem: "speech_encoder", Role: "speech-encoder"),
                (Stem: "embed_tokens", Role: "embed-tokens"),
                (Stem: "language_model", Role: "language-model"),
                (Stem: "conditional_decoder", Role: "conditional-decoder")
            };
            foreach (HfSibling model in models)
            {
                var match = roles.FirstOrDefault(candidate =>
                    FileStem(model).StartsWith(candidate.Stem, StringComparison.OrdinalIgnoreCase));
                if (match.Stem == null) continue;
                await AddArtifactAsync(result, request, revision, byPath, model, match.Role, match.Stem, false,
                    token, cancellationToken);
            }
            return Finish(result);
        }

        private static async Task<Dictionary<string, DiscoveredArtifact[]>> DiscoverModelsAsync(
            ModelDownloadRequest request, string revision, Dictionary<string, HfSibling> byPath,
            HfSibling[] models, string token, CancellationToken cancellationToken)
        {
            var result = new Dictionary<string, List<DiscoveredArtifact>>(StringComparer.Ordinal);
            bool llm2vec = string.Equals(request.ExpectedFamily, "llm2vec", StringComparison.OrdinalIgnoreCase);
            foreach (HfSibling model in models)
            {
                string stem = FileStem(model);
                string baseStem;
                if (llm2vec && stem.StartsWith("encoder", StringComparison.OrdinalIgnoreCase))
                    baseStem = "encoder";
                else if (stem.StartsWith("model", StringComparison.OrdinalIgnoreCase))
                    baseStem = "model";
                else
                    continue;
                await AddArtifactAsync(result, request, revision, byPath, model, llm2vec ? "encoder" : "model",
                    baseStem, false, token, cancellationToken);
            }
            return Finish(result);
        }

        private static async Task AddArtifactAsync(
            Dictionary<string, List<DiscoveredArtifact>> result, ModelDownloadRequest request, string revision,
            Dictionary<string, HfSibling> byPath, HfSibling source, string role, string baseStem, bool merged,
            string token, CancellationToken cancellationToken)
        {
            var files = new List<DiscoveredFile>
            {
                await DiscoverFileAsync(request, revision, source, role, token, cancellationToken)
            };
            await AddExternalDataAsync(request, revision, byPath, source, role + "-data", files, token,
                cancellationToken);
            if (!result.TryGetValue(role, out List<DiscoveredArtifact> artifacts))
                result.Add(role, artifacts = new List<DiscoveredArtifact>());
            artifacts.Add(new DiscoveredArtifact(role, ArtifactType(source.rfilename, baseStem), merged,
                files.ToArray()));
        }

        private static Dictionary<string, DiscoveredArtifact[]> Finish(
            Dictionary<string, List<DiscoveredArtifact>> source)
        {
            return source.ToDictionary(
                pair => pair.Key,
                pair => pair.Value
                    .OrderBy(artifact => string.Equals(artifact.Type, "default", StringComparison.OrdinalIgnoreCase)
                        ? 0
                        : 1)
                    .ThenBy(artifact => artifact.Type, StringComparer.OrdinalIgnoreCase)
                    .ThenBy(artifact => artifact.Model.Path, StringComparer.OrdinalIgnoreCase)
                    .ToArray(),
                StringComparer.Ordinal);
        }

        private static async Task<DiscoveredFile[]> DiscoverCommonFilesAsync(ModelDownloadRequest request,
            string revision, Dictionary<string, HfSibling> byPath, string token,
            CancellationToken cancellationToken)
        {
            var files = new List<DiscoveredFile>();
            foreach (var item in new[]
            {
                (Path: "tokenizer.json", Role: "tokenizer"),
                (Path: "tokenizer_config.json", Role: "tokenizer-config"),
                (Path: "config.json", Role: "config"),
                (Path: "generation_config.json", Role: "generation-config"),
                (Path: "preprocessor_config.json", Role: "preprocessor-config"),
                (Path: "special_tokens_map.json", Role: "special-tokens"),
                (Path: "added_tokens.json", Role: "added-tokens"),
                (Path: "vocab.json", Role: "vocabulary"),
                (Path: "vocab.txt", Role: "vocabulary"),
                (Path: "merges.txt", Role: "merges"),
                (Path: "default_voice.wav", Role: "voice"),
                (Path: "cangjie.json", Role: "cangjie"),
                (Path: "cangjie_mapping.json", Role: "cangjie")
            })
            {
                if (!byPath.TryGetValue(item.Path, out HfSibling sibling)) continue;
                if (files.Any(file => string.Equals(file.Role, item.Role, StringComparison.Ordinal))) continue;
                files.Add(await DiscoverFileAsync(request, revision, sibling, item.Role, token, cancellationToken));
            }
            return files.ToArray();
        }

        private static async Task AddExternalDataAsync(ModelDownloadRequest request, string revision,
            Dictionary<string, HfSibling> byPath, HfSibling model, string role, List<DiscoveredFile> files,
            string token, CancellationToken cancellationToken)
        {
            foreach (string candidate in new[] { model.rfilename + "_data", model.rfilename + ".data" })
            {
                if (!byPath.TryGetValue(candidate, out HfSibling data)) continue;
                files.Add(await DiscoverFileAsync(request, revision, data, role, token, cancellationToken));
                return;
            }
        }

        private static async Task<DiscoveredFile> DiscoverFileAsync(ModelDownloadRequest request, string revision,
            HfSibling source, string role, string token, CancellationToken cancellationToken)
        {
            string hash = source.lfs?.sha256;
            long size = source.lfs != null && source.lfs.size > 0 ? source.lfs.size : source.size;
            if (string.IsNullOrWhiteSpace(hash))
            {
                using var message =
                    new HttpRequestMessage(HttpMethod.Get, FileUrl(request, revision, source.rfilename));
                AddToken(message, token);
                using HttpResponseMessage response = await Client.SendAsync(message, cancellationToken);
                response.EnsureSuccessStatusCode();
                byte[] bytes = await response.Content.ReadAsByteArrayAsync();
                using var sha = SHA256.Create();
                hash = ToHex(sha.ComputeHash(bytes));
                size = bytes.LongLength;
            }
            return new DiscoveredFile(role, source.rfilename, hash, size);
        }

        internal static DiscoveredArtifact[] SelectArtifacts(DiscoveredRepository repository,
            IReadOnlyDictionary<string, string> selection)
        {
            var selected = new Dictionary<string, DiscoveredArtifact>(StringComparer.Ordinal);
            foreach (string role in repository.RequiredRoles)
                selected[role] = SelectArtifact(repository, selection, role);

            if (string.Equals(repository.Family, "whisper", StringComparison.OrdinalIgnoreCase))
            {
                DiscoveredArtifact decoder = selected["decoder"];
                if (!decoder.IsMerged)
                    selected["decoder-with-past"] = SelectArtifact(repository, selection, "decoder-with-past");
            }

            foreach (string role in selection.Keys)
                if (!repository.Artifacts.ContainsKey(role))
                    throw new InvalidDataException($"Repository does not contain model role '{role}'.");
            return selected.Values.ToArray();
        }

        private static DiscoveredArtifact SelectArtifact(DiscoveredRepository repository,
            IReadOnlyDictionary<string, string> selection, string role)
        {
            if (!repository.Artifacts.TryGetValue(role, out DiscoveredArtifact[] choices) || choices.Length == 0)
                throw new InvalidDataException($"Repository does not contain model role '{role}'.");
            if (!selection.TryGetValue(role, out string path) || string.IsNullOrWhiteSpace(path))
                throw new InvalidDataException($"Select an artifact for '{role}'.");
            DiscoveredArtifact selected = choices.FirstOrDefault(candidate =>
                string.Equals(candidate.Model.Path, path, StringComparison.OrdinalIgnoreCase));
            return selected ?? throw new InvalidDataException(
                $"Selected artifact '{path}' is not available for role '{role}'.");
        }

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

        private static void InspectModels(IEnumerable<DiscoveredArtifact> artifacts,
            IReadOnlyDictionary<string, string> installed)
        {
            foreach (DiscoveredArtifact artifact in artifacts)
            {
                string path = Path.GetFullPath(Path.Combine(Directory.GetCurrentDirectory(),
                    installed[artifact.Role]));
                ExternalOnnxModelAssetUtility.InspectionResult metadata =
                    ExternalOnnxModelAssetUtility.Inspect(path);
                artifact.Model.SetMetadata(metadata.Inputs, metadata.Outputs);
            }
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

        private static bool IsChatterbox(ModelDownloadRequest request, IEnumerable<HfSibling> files)
        {
            return string.Equals(request.ExpectedFamily, "chatterbox", StringComparison.OrdinalIgnoreCase) ||
                files.Any(file => FileStem(file).StartsWith("speech_encoder", StringComparison.OrdinalIgnoreCase));
        }

        private static string FileStem(HfSibling file)
        {
            return Path.GetFileNameWithoutExtension(file.rfilename);
        }

        private static void ValidateFile(DiscoveredFile file)
        {
            if (file == null || string.IsNullOrWhiteSpace(file.Role) || string.IsNullOrWhiteSpace(file.Path) ||
                string.IsNullOrWhiteSpace(file.Sha256) || file.Sha256.Length != 64 ||
                file.Sha256.Any(character => !Uri.IsHexDigit(character)) || file.Size < 0)
                throw new InvalidDataException("Repository contains an invalid file entry.");
        }

        private static string FileUrl(ModelDownloadRequest request, string revision, string path)
        {
            string[] parts = RepositoryParts(request.Repository);
            return
                $"https://huggingface.co/{Uri.EscapeDataString(parts[0])}/{Uri.EscapeDataString(parts[1])}/resolve/{Uri.EscapeDataString(revision)}/{EscapePath(path)}";
        }

        private static string[] RepositoryParts(string value)
        {
            string repository = value.Trim().TrimEnd('/');
            if (Uri.TryCreate(repository, UriKind.Absolute, out Uri uri))
            {
                if (!string.Equals(uri.Host, "huggingface.co", StringComparison.OrdinalIgnoreCase))
                    throw new NotSupportedException("Model repositories must be hosted on huggingface.co.");
                repository = uri.AbsolutePath.Trim('/');
            }
            string[] parts = repository.Split('/');
            if (parts.Length != 2 || parts.Any(string.IsNullOrWhiteSpace))
                throw new ArgumentException("Repository must use 'owner/name' or a huggingface.co model URL.");
            return parts;
        }

        private static string EscapePath(string path)
        {
            return string.Join("/", SafeRelativePath(path).Split('/').Select(Uri.EscapeDataString));
        }

        private static async Task DownloadFileAsync(string url, string cache, DiscoveredFile file, string token,
            CancellationToken cancellationToken)
        {
            Directory.CreateDirectory(Path.GetDirectoryName(cache));
            string partial = cache + ".partial";
            using var message = new HttpRequestMessage(HttpMethod.Get, url);
            AddToken(message, token);
            using HttpResponseMessage response = await Client.SendAsync(message,
                HttpCompletionOption.ResponseHeadersRead, cancellationToken);
            response.EnsureSuccessStatusCode();
            using Stream source = await response.Content.ReadAsStreamAsync();
            using (var destination = new FileStream(partial, FileMode.Create, FileAccess.Write, FileShare.None,
                1024 * 1024, true))
                await source.CopyToAsync(destination, 1024 * 1024, cancellationToken);
            if (file.Size > 0 && new FileInfo(partial).Length != file.Size)
            {
                File.Delete(partial);
                throw new InvalidDataException(
                    $"Downloaded size does not match repository metadata for '{file.Path}'.");
            }
            if (!await HasExpectedHashAsync(partial, file.Sha256, cancellationToken))
            {
                File.Delete(partial);
                throw new InvalidDataException($"SHA-256 verification failed for '{file.Path}'.");
            }
            if (File.Exists(cache)) File.Delete(cache);
            File.Move(partial, cache);
        }

        private static void CopyVerifiedFile(string source, string destination)
        {
            Directory.CreateDirectory(Path.GetDirectoryName(destination));
            string partial = destination + ".partial";
            File.Copy(source, partial, true);
            if (File.Exists(destination)) File.Delete(destination);
            File.Move(partial, destination);
        }

        private static async Task<bool> HasExpectedHashAsync(string path, string expected,
            CancellationToken cancellationToken)
        {
            if (!File.Exists(path)) return false;
            using var sha = SHA256.Create();
            using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 1024 * 1024,
                true);
            byte[] hash = await Task.Run(() =>
            {
                cancellationToken.ThrowIfCancellationRequested();
                return sha.ComputeHash(stream);
            }, cancellationToken);
            return string.Equals(ToHex(hash), expected, StringComparison.OrdinalIgnoreCase);
        }

        private static string SafeProjectDirectory(string relative)
        {
            string projectRoot = Path.GetFullPath(Directory.GetCurrentDirectory());
            string full = SafeChildPath(projectRoot, relative);
            Directory.CreateDirectory(full);
            return full;
        }

        private static string SafeChildPath(string root, string relative)
        {
            string safe = SafeRelativePath(relative);
            string full = Path.GetFullPath(Path.Combine(root, safe));
            string prefix = root.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar) +
                Path.DirectorySeparatorChar;
            if (!full.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
                throw new InvalidDataException($"Path escapes the project: '{relative}'.");
            return full;
        }

        private static string CacheRoot(string projectRoot)
        {
            string configured = Environment.GetEnvironmentVariable("KITSUMATE_MODEL_CACHE");
            return string.IsNullOrWhiteSpace(configured)
                ? Path.Combine(projectRoot, "Library", "KitsuMateModelCache")
                : Path.GetFullPath(configured);
        }

        private static void AddToken(HttpRequestMessage request, string token)
        {
            if (!string.IsNullOrWhiteSpace(token))
                request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", token.Trim());
        }

        private static string ContentHash(IEnumerable<DiscoveredFile> files)
        {
            string value = string.Join("\n", files.OrderBy(file => file.Path, StringComparer.Ordinal)
                .Select(file => file.Role + ":" + file.Path + ":" + file.Sha256));
            using var sha = SHA256.Create();
            return ToHex(sha.ComputeHash(Encoding.UTF8.GetBytes(value)));
        }

        private static string ToHex(byte[] bytes)
        {
            return BitConverter.ToString(bytes).Replace("-", string.Empty).ToLowerInvariant();
        }
    }

    internal sealed class DiscoveredRepository
    {
        public string Owner { get; }
        public string Name { get; }
        public string Family { get; }
        public string Revision { get; }
        public Dictionary<string, DiscoveredArtifact[]> Artifacts { get; }
        public DiscoveredFile[] CommonFiles { get; }
        public string[] RequiredRoles { get; }

        public DiscoveredRepository(string owner, string name, string family, string revision,
            Dictionary<string, DiscoveredArtifact[]> artifacts, DiscoveredFile[] commonFiles,
            string[] requiredRoles)
        {
            Owner = owner;
            Name = name;
            Family = family;
            Revision = revision;
            Artifacts = artifacts;
            CommonFiles = commonFiles;
            RequiredRoles = requiredRoles;
        }
    }

    internal sealed class DiscoveredArtifact
    {
        public string Role { get; }
        public string Type { get; }
        public bool IsMerged { get; }
        public DiscoveredFile[] Files { get; }
        public DiscoveredFile Model => Files[0];

        public DiscoveredArtifact(string role, string type, bool isMerged, DiscoveredFile[] files)
        {
            Role = role;
            Type = type;
            IsMerged = isMerged;
            Files = files;
        }
    }

    internal sealed class DiscoveredFile
    {
        public string Role { get; }
        public string Path { get; }
        public string Sha256 { get; }
        public long Size { get; }
        public OnnxModelAsset.TensorInfo[] Inputs { get; private set; } =
            Array.Empty<OnnxModelAsset.TensorInfo>();
        public OnnxModelAsset.TensorInfo[] Outputs { get; private set; } =
            Array.Empty<OnnxModelAsset.TensorInfo>();

        public DiscoveredFile(string role, string path, string sha256, long size)
        {
            Role = role;
            Path = path;
            Sha256 = sha256;
            Size = size;
        }

        public void SetMetadata(OnnxModelAsset.TensorInfo[] inputs, OnnxModelAsset.TensorInfo[] outputs)
        {
            Inputs = inputs ?? Array.Empty<OnnxModelAsset.TensorInfo>();
            Outputs = outputs ?? Array.Empty<OnnxModelAsset.TensorInfo>();
        }
    }

    [Serializable]
    internal sealed class HfModelInfo
    {
        public string sha;
        public HfSibling[] siblings = Array.Empty<HfSibling>();
    }

    [Serializable]
    internal sealed class HfSibling
    {
        public string rfilename;
        public long size;
        public HfLfs lfs;
    }

    [Serializable]
    internal sealed class HfLfs
    {
        public string sha256;
        public long size;
    }
}

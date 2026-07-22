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
        public IReadOnlyList<ModelProviderCompatibility> ProviderCompatibility { get; } =
            Array.Empty<ModelProviderCompatibility>();

        internal ModelDownloadResult(ModelIdentity identity, DiscoveredVariant variant,
            IReadOnlyDictionary<string, string> paths, string root)
        {
            Identity = identity;
            files = variant.Files;
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
                throw new InvalidOperationException($"Downloaded '{role}' file was not imported as {typeof(T).Name} at '{path}'.");
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
            target.SetDownloadMetadata(Identity, Capabilities, ProviderCompatibility);
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
        private static readonly HttpClient Client = new HttpClient();

        public static async Task<IReadOnlyList<string>> GetVariantsAsync(ModelDownloadRequest request,
            string token = null, CancellationToken cancellationToken = default)
        {
            DiscoveredRepository repository = await ScanAsync(request, token, cancellationToken);
            return repository.Variants.Select(variant => variant.Name).ToArray();
        }

        public static async Task<ModelDownloadResult> DownloadAsync(ModelDownloadRequest request,
            string token = null, IProgress<float> progress = null, CancellationToken cancellationToken = default)
        {
            if (request == null) throw new ArgumentNullException(nameof(request));
            DiscoveredRepository repository = await ScanAsync(request, token, cancellationToken);
            DiscoveredVariant variant = SelectVariant(repository, request.Variant);
            SafeProjectDirectory(request.Destination);
            string projectRoot = Path.GetFullPath(Directory.GetCurrentDirectory());
            string cacheRoot = CacheRoot(projectRoot);
            Directory.CreateDirectory(cacheRoot);

            var installed = new Dictionary<string, string>(StringComparer.Ordinal);
            long total = variant.Files.Sum(file => Math.Max(0, file.Size));
            long completed = 0;

            foreach (DiscoveredFile file in variant.Files)
            {
                cancellationToken.ThrowIfCancellationRequested();
                ValidateFile(file);
                string relative = SafeRelativePath(file.Path);
                string projectPath = Path.Combine(request.Destination, repository.Family, repository.ModelId,
                    variant.Name, relative).Replace('\\', '/');
                string destination = SafeChildPath(projectRoot, projectPath);
                string cache = Path.Combine(cacheRoot, file.Sha256.ToLowerInvariant());

                if (!await HasExpectedHashAsync(destination, file.Sha256, cancellationToken))
                {
                    if (!await HasExpectedHashAsync(cache, file.Sha256, cancellationToken))
                        await DownloadFileAsync(FileUrl(request, repository.Revision, relative), cache, file, token,
                            cancellationToken);
                    CopyVerifiedFile(cache, destination);
                }

                completed += Math.Max(0, file.Size);
                progress?.Report(total > 0 ? Math.Min(1f, (float)completed / total) :
                    (float)(installed.Count + 1) / variant.Files.Length);
                installed.Add(file.Role, projectPath);
            }

            AssetDatabase.Refresh(ImportAssetOptions.ForceSynchronousImport);
            var identity = new ModelIdentity(repository.Family, repository.ModelId, repository.Revision,
                variant.Name, ContentHash(variant), 1);
            return new ModelDownloadResult(identity, variant, installed, request.Destination);
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

        private static async Task<DiscoveredRepository> ScanAsync(ModelDownloadRequest request, string token,
            CancellationToken cancellationToken)
        {
            if (request == null) throw new ArgumentNullException(nameof(request));
            string[] repository = RepositoryParts(request.Repository);
            string url = $"https://huggingface.co/api/models/{Uri.EscapeDataString(repository[0])}/{Uri.EscapeDataString(repository[1])}/revision/{Uri.EscapeDataString(request.Revision)}?blobs=true";
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

            DiscoveredFile[] common = await DiscoverCommonFilesAsync(request, info.sha, byPath, token,
                cancellationToken);
            List<DiscoveredVariant> variants;
            if (IsChatterbox(request, onnxFiles))
                variants = await DiscoverChatterboxAsync(request, info.sha, byPath, onnxFiles, common, token,
                    cancellationToken);
            else if (onnxFiles.Any(file => FileStem(file).StartsWith("encoder_model", StringComparison.OrdinalIgnoreCase)))
                variants = await DiscoverWhisperAsync(request, info.sha, byPath, onnxFiles, common, token,
                    cancellationToken);
            else
                variants = await DiscoverModelsAsync(request, info.sha, byPath, onnxFiles, common, token,
                    cancellationToken);

            if (variants.Count == 0)
                throw new InvalidDataException($"Repository '{request.Repository}' does not use a recognized ONNX layout.");
            return new DiscoveredRepository(
                string.IsNullOrWhiteSpace(request.ExpectedFamily) ? "onnx" : request.ExpectedFamily,
                repository[1], info.sha,
                variants.OrderBy(variant => variant.Name, StringComparer.OrdinalIgnoreCase).ToArray());
        }

        private static async Task<List<DiscoveredVariant>> DiscoverWhisperAsync(ModelDownloadRequest request,
            string revision, Dictionary<string, HfSibling> byPath, HfSibling[] onnxFiles, DiscoveredFile[] common,
            string token, CancellationToken cancellationToken)
        {
            var variants = new List<DiscoveredVariant>();
            foreach (HfSibling encoder in onnxFiles.Where(file =>
                FileStem(file).StartsWith("encoder_model", StringComparison.OrdinalIgnoreCase)))
            {
                string suffix = FileStem(encoder).Substring("encoder_model".Length);
                string directory = PathPrefix(encoder.rfilename);
                if (!byPath.TryGetValue(directory + "decoder_model_merged" + suffix + ".onnx", out HfSibling decoder) &&
                    !byPath.TryGetValue(directory + "decoder_model" + suffix + ".onnx", out decoder))
                    continue;

                var files = new List<DiscoveredFile>
                {
                    await DiscoverFileAsync(request, revision, encoder, "encoder", token, cancellationToken),
                    await DiscoverFileAsync(request, revision, decoder, "decoder", token, cancellationToken)
                };
                await AddExternalDataAsync(request, revision, byPath, encoder, "encoder-data", files, token,
                    cancellationToken);
                await AddExternalDataAsync(request, revision, byPath, decoder, "decoder-data", files, token,
                    cancellationToken);
                files.AddRange(common);
                variants.Add(new DiscoveredVariant(VariantName(suffix, "fp32"), files.ToArray()));
            }
            return variants;
        }

        private static async Task<List<DiscoveredVariant>> DiscoverChatterboxAsync(ModelDownloadRequest request,
            string revision, Dictionary<string, HfSibling> byPath, HfSibling[] onnxFiles, DiscoveredFile[] common,
            string token, CancellationToken cancellationToken)
        {
            var components = new[]
            {
                (Prefix: "speech_encoder", Role: "speech-encoder"),
                (Prefix: "embed_tokens", Role: "embed-tokens"),
                (Prefix: "language_model", Role: "language-model"),
                (Prefix: "conditional_decoder", Role: "conditional-decoder")
            };
            var groups = new Dictionary<string, Dictionary<string, HfSibling>>(StringComparer.OrdinalIgnoreCase);
            foreach (HfSibling model in onnxFiles)
            {
                var component = components.FirstOrDefault(item =>
                    FileStem(model).StartsWith(item.Prefix, StringComparison.OrdinalIgnoreCase));
                if (component.Prefix == null) continue;
                string suffix = FileStem(model).Substring(component.Prefix.Length);
                string name = VariantName(suffix, "default");
                if (!groups.TryGetValue(name, out Dictionary<string, HfSibling> files))
                    groups.Add(name, files = new Dictionary<string, HfSibling>(StringComparer.Ordinal));
                files[component.Role] = model;
            }

            var variants = new List<DiscoveredVariant>();
            foreach (var group in groups)
            {
                if (components.Any(component => !group.Value.ContainsKey(component.Role))) continue;
                var files = new List<DiscoveredFile>();
                foreach (var component in components)
                {
                    HfSibling model = group.Value[component.Role];
                    files.Add(await DiscoverFileAsync(request, revision, model, component.Role, token,
                        cancellationToken));
                    await AddExternalDataAsync(request, revision, byPath, model, component.Role + "-data", files,
                        token, cancellationToken);
                }
                files.AddRange(common);
                variants.Add(new DiscoveredVariant(group.Key, files.ToArray()));
            }
            return variants;
        }

        private static async Task<List<DiscoveredVariant>> DiscoverModelsAsync(ModelDownloadRequest request,
            string revision, Dictionary<string, HfSibling> byPath, HfSibling[] onnxFiles, DiscoveredFile[] common,
            string token, CancellationToken cancellationToken)
        {
            var variants = new List<DiscoveredVariant>();
            var names = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            string role = string.Equals(request.ExpectedFamily, "llm2vec", StringComparison.OrdinalIgnoreCase)
                ? "encoder"
                : "model";
            foreach (HfSibling model in onnxFiles.Where(file =>
                !string.Equals(Path.GetFileName(file.rfilename), "mel.onnx", StringComparison.OrdinalIgnoreCase)))
            {
                string name = ModelVariantName(model.rfilename);
                if (!names.Add(name)) continue;
                var files = new List<DiscoveredFile>
                {
                    await DiscoverFileAsync(request, revision, model, role, token, cancellationToken)
                };
                await AddExternalDataAsync(request, revision, byPath, model, role + "-data", files, token,
                    cancellationToken);
                files.AddRange(common);
                variants.Add(new DiscoveredVariant(name, files.ToArray()));
            }
            return variants;
        }

        private static async Task<DiscoveredFile[]> DiscoverCommonFilesAsync(ModelDownloadRequest request,
            string revision, Dictionary<string, HfSibling> byPath, string token,
            CancellationToken cancellationToken)
        {
            var files = new List<DiscoveredFile>();
            foreach (var item in new[]
            {
                (Path: "onnx/mel.onnx", Role: "mel"),
                (Path: "tokenizer.json", Role: "tokenizer"),
                (Path: "tokenizer_config.json", Role: "tokenizer-config"),
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
                using var message = new HttpRequestMessage(HttpMethod.Get, FileUrl(request, revision, source.rfilename));
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

        private static bool IsChatterbox(ModelDownloadRequest request, IEnumerable<HfSibling> files)
        {
            return string.Equals(request.ExpectedFamily, "chatterbox", StringComparison.OrdinalIgnoreCase) ||
                files.Any(file => FileStem(file).StartsWith("speech_encoder", StringComparison.OrdinalIgnoreCase));
        }

        private static string FileStem(HfSibling file)
        {
            return Path.GetFileNameWithoutExtension(file.rfilename);
        }

        private static string PathPrefix(string path)
        {
            int slash = path.LastIndexOf('/');
            return slash < 0 ? string.Empty : path.Substring(0, slash + 1);
        }

        private static string ModelVariantName(string path)
        {
            string stem = Path.GetFileNameWithoutExtension(path);
            if (string.Equals(stem, "model", StringComparison.OrdinalIgnoreCase)) return "default";
            return stem.StartsWith("model_", StringComparison.OrdinalIgnoreCase)
                ? VariantName(stem.Substring("model".Length), "default")
                : VariantName(stem, "default");
        }

        private static string VariantName(string suffix, string fallback)
        {
            string value = suffix?.Trim().TrimStart('_', '-') ?? string.Empty;
            return string.IsNullOrWhiteSpace(value) ? fallback : value.Replace('_', '-').ToLowerInvariant();
        }

        private static DiscoveredVariant SelectVariant(DiscoveredRepository repository, string name)
        {
            DiscoveredVariant selected = string.IsNullOrWhiteSpace(name)
                ? repository.Variants.FirstOrDefault()
                : repository.Variants.FirstOrDefault(variant =>
                    string.Equals(variant.Name, name, StringComparison.OrdinalIgnoreCase));
            if (selected == null) throw new InvalidDataException($"Model variant '{name}' was not found.");
            return selected;
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
            return $"https://huggingface.co/{Uri.EscapeDataString(parts[0])}/{Uri.EscapeDataString(parts[1])}/resolve/{Uri.EscapeDataString(revision)}/{EscapePath(path)}";
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
                throw new InvalidDataException($"Downloaded size does not match repository metadata for '{file.Path}'.");
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
            using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 1024 * 1024, true);
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

        private static string ContentHash(DiscoveredVariant variant)
        {
            string value = string.Join("\n", variant.Files.OrderBy(file => file.Path, StringComparer.Ordinal)
                .Select(file => file.Path + ":" + file.Sha256));
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
        public string Family { get; }
        public string ModelId { get; }
        public string Revision { get; }
        public DiscoveredVariant[] Variants { get; }

        public DiscoveredRepository(string family, string modelId, string revision, DiscoveredVariant[] variants)
        {
            Family = family;
            ModelId = modelId;
            Revision = revision;
            Variants = variants;
        }
    }

    internal sealed class DiscoveredVariant
    {
        public string Name { get; }
        public DiscoveredFile[] Files { get; }

        public DiscoveredVariant(string name, DiscoveredFile[] files)
        {
            Name = name;
            Files = files;
        }
    }

    internal sealed class DiscoveredFile
    {
        public string Role { get; }
        public string Path { get; }
        public string Sha256 { get; }
        public long Size { get; }
        public OnnxModelAsset.TensorInfo[] Inputs { get; } = Array.Empty<OnnxModelAsset.TensorInfo>();
        public OnnxModelAsset.TensorInfo[] Outputs { get; } = Array.Empty<OnnxModelAsset.TensorInfo>();

        public DiscoveredFile(string role, string path, string sha256, long size)
        {
            Role = role;
            Path = path;
            Sha256 = sha256;
            Size = size;
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

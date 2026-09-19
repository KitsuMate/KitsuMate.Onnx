using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

namespace KitsuMate.Onnx.Download
{
    /// <summary>Discovers supported Hugging Face layouts without importing Unity assets.</summary>
    public static class HuggingFaceModelRepository
    {
        private static readonly HttpClient Client = new();
        public static async Task<string[]> GetBranchesAsync(string repository, string token = null,
            CancellationToken cancellationToken = default)
        {
            string[] parts = RepositoryParts(repository);
            using var request = new HttpRequestMessage(HttpMethod.Get,
                $"https://huggingface.co/api/models/{Uri.EscapeDataString(parts[0])}/{Uri.EscapeDataString(parts[1])}/refs");
            AddToken(request, token);
            using var response = await Client.SendAsync(request, cancellationToken);
            response.EnsureSuccessStatusCode();
            var refs = JsonUtility.FromJson<RepositoryRefs>(await response.Content.ReadAsStringAsync());
            return (refs?.branches ?? Array.Empty<RepositoryBranch>()).Select(branch => branch.name)
                .Where(name => !string.IsNullOrWhiteSpace(name)).OrderBy(name => name, StringComparer.Ordinal).ToArray();
        }

        [Serializable] private sealed class RepositoryRefs { public RepositoryBranch[] branches; }
        [Serializable] private sealed class RepositoryBranch { public string name; }

        public static async Task<(string Revision, IReadOnlyDictionary<string, IReadOnlyList<string>> Artifacts)> GetArtifactsAsync(
            ModelDownloadRequest request, string token = null, CancellationToken cancellationToken = default)
        {
            var repository = await ScanAsync(request, token, cancellationToken);
            return (repository.Revision, repository.Artifacts.ToDictionary(pair => pair.Key,
                pair => (IReadOnlyList<string>)pair.Value.Select(artifact => artifact.Model.Path).ToArray()));
        }
        public static async Task<DiscoveredRepository> ScanAsync(ModelDownloadRequest request, string token = null,
            CancellationToken cancellationToken = default)
        {
            if (request == null) throw new ArgumentNullException(nameof(request));
            string[] repository = RepositoryParts(request.Repository);
            string url =
                $"https://huggingface.co/api/models/{Uri.EscapeDataString(repository[0])}/{Uri.EscapeDataString(repository[1])}" +
                (string.IsNullOrEmpty(request.Revision) ? "" : $"/revision/{Uri.EscapeDataString(request.Revision)}") + "?blobs=true";
            using var message = new HttpRequestMessage(HttpMethod.Get, url);
            AddToken(message, token);
            using HttpResponseMessage response = await Client.SendAsync(message, cancellationToken);
            response.EnsureSuccessStatusCode();
            HfModelInfo info = JsonUtility.FromJson<HfModelInfo>(await response.Content.ReadAsStringAsync());
            return await ScanSnapshotAsync(request, info, token, cancellationToken);
        }

        internal static async Task<DiscoveredRepository> ScanSnapshotAsync(ModelDownloadRequest request,
            HfModelInfo info, string token = null, CancellationToken cancellationToken = default)
        {
            if (request == null) throw new ArgumentNullException(nameof(request));
            string[] repository = RepositoryParts(request.Repository);
            if (info == null || string.IsNullOrWhiteSpace(info.sha))
                throw new InvalidDataException($"Could not resolve '{request.Repository}@{request.Revision}'.");

            var filesByPath = (info.siblings ?? Array.Empty<HfSibling>())
                .Where(file => file != null && !string.IsNullOrWhiteSpace(file.rfilename))
                .ToArray();
            var byPath = new Dictionary<string, HfSibling>(StringComparer.OrdinalIgnoreCase);
            foreach (var file in filesByPath)
            {
                string path = SafeRelativePath(file.rfilename);
                if (byPath.ContainsKey(path))
                    throw new InvalidDataException($"Repository contains colliding file paths: '{path}'.");
                byPath.Add(path, file);
            }
            bool omniVoice = IsOmniVoice(request);
            bool neuTts = string.Equals(request.ExpectedFamily, "neutts", StringComparison.OrdinalIgnoreCase) ||
                string.Equals(request.Repository, "KitsuMate/neutts-2e-onnx", StringComparison.OrdinalIgnoreCase);
            HfSibling[] onnxFiles = byPath.Values
                .Where(file => (omniVoice || file.rfilename.StartsWith("onnx/", StringComparison.OrdinalIgnoreCase)) &&
                    file.rfilename.EndsWith(".onnx", StringComparison.OrdinalIgnoreCase))
                .OrderBy(file => file.rfilename, StringComparer.OrdinalIgnoreCase)
                .ToArray();
            if (onnxFiles.Length == 0)
                throw new InvalidDataException($"Repository '{request.Repository}' has no ONNX files under onnx/.");

            Dictionary<string, DiscoveredArtifact[]> artifacts;
            string[] requiredRoles;
            if (request.GraphRoles != null && request.GraphRoles.Count > 0)
            {
                artifacts = await DiscoverFixedGraphsAsync(request, info.sha, byPath, onnxFiles, token,
                    cancellationToken);
                requiredRoles = request.GraphRoles.Select(role => role.role).ToArray();
            }
            else if (neuTts)
            {
                var choices = new Dictionary<string, List<DiscoveredArtifact>>(StringComparer.Ordinal);
                foreach (var model in onnxFiles)
                {
                    string stem = FileStem(model);
                    string role = stem.StartsWith("backbone_", StringComparison.Ordinal) ? "backbone" : stem == "codec_decoder" ? "codec-decoder" : null;
                    if (role == null) continue;
                    await AddArtifactAsync(choices, request, info.sha, byPath, model, role,
                        role == "backbone" ? "backbone" : "codec_decoder", false, token, cancellationToken);
                }
                artifacts = Finish(choices);
                requiredRoles = new[] { "backbone", "codec-decoder" };
                if (!byPath.ContainsKey("neutts.json") || !byPath.ContainsKey("tokenizer.json"))
                    throw new InvalidDataException("NeuTTS requires tokenizer.json and neutts.json speaker metadata.");
            }
            else if (omniVoice)
            {
                artifacts = await DiscoverOmniVoiceAsync(request, info.sha, byPath, onnxFiles, token,
                    cancellationToken);
                bool merged = artifacts.ContainsKey("merged-backbone");
                requiredRoles = merged
                    ? new[] { "merged-backbone", "acoustic-encoder", "semantic-encoder", "quantizer-encoder", "higgs-decoder" }
                    : new[] { "audio-embeddings", "language-decoder", "audio-heads", "acoustic-encoder", "semantic-encoder", "quantizer-encoder", "higgs-decoder" };
            }
            else if (IsChatterbox(request, onnxFiles))
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

            string[] missingRoles = requiredRoles.Where(role =>
                !artifacts.TryGetValue(role, out DiscoveredArtifact[] choices) || choices.Length == 0).ToArray();
            if (missingRoles.Length > 0)
                throw new InvalidDataException(
                    $"Repository '{request.Repository}' is missing the required ONNX graph roles: {string.Join(", ", missingRoles)}.");

            DiscoveredFile[] common = await DiscoverCommonFilesAsync(request, info.sha, byPath, token,
                cancellationToken);
            string[] missingCompanions = (request.RequiredCompanionRoles ?? Array.Empty<string>())
                .Where(role => !common.Any(file => file.Role == role)).ToArray();
            if (missingCompanions.Length > 0)
                throw new InvalidDataException($"Repository '{request.Repository}' is missing required companion files: {string.Join(", ", missingCompanions)}.");
            return new DiscoveredRepository(repository[0], repository[1],
                neuTts ? "neutts" : omniVoice ? "omnivoice" : string.IsNullOrWhiteSpace(request.ExpectedFamily) ? "onnx" : request.ExpectedFamily,
                info.sha, artifacts, common, requiredRoles);
        }

        public static string ArtifactType(string path, string baseStem)
        {
            string stem = Path.GetFileNameWithoutExtension(path);
            if (!stem.StartsWith(baseStem, StringComparison.OrdinalIgnoreCase)) return null;
            string suffix = stem.Substring(baseStem.Length).TrimStart('_', '-');
            return string.IsNullOrWhiteSpace(suffix) ? "default" : suffix;
        }

        public static string ArtifactLabel(DiscoveredArtifact artifact)
        {
            string type = string.Equals(artifact.Type, "default", StringComparison.OrdinalIgnoreCase)
                ? "Default"
                : artifact.Type;
            return $"{type} - {artifact.Model.Path}";
        }

        public static bool IsSentisArtifact(string type)
        {
            return string.Equals(type, "default", StringComparison.OrdinalIgnoreCase) ||
                string.Equals(type, "fp32", StringComparison.OrdinalIgnoreCase);
        }

        public static string SafeRelativePath(string path)
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

        private static async Task<Dictionary<string, DiscoveredArtifact[]>> DiscoverFixedGraphsAsync(
            ModelDownloadRequest request, string revision, Dictionary<string, HfSibling> byPath,
            HfSibling[] models, string token, CancellationToken cancellationToken)
        {
            var result = new Dictionary<string, List<DiscoveredArtifact>>(StringComparer.Ordinal);
            var roles = request.GraphRoles;
            if (roles.Any(item => item == null || string.IsNullOrWhiteSpace(item.role) ||
                    string.IsNullOrWhiteSpace(item.fileStem)) ||
                roles.Select(item => item.role).Distinct(StringComparer.Ordinal).Count() != roles.Count)
                throw new ArgumentException("The model graph contract has invalid or duplicate roles.");
            foreach (HfSibling model in models)
            {
                string stem = FileStem(model);
                var match = roles.FirstOrDefault(candidate =>
                    stem.Equals(candidate.fileStem, StringComparison.OrdinalIgnoreCase) ||
                    stem.StartsWith(candidate.fileStem + "_", StringComparison.OrdinalIgnoreCase) ||
                    stem.StartsWith(candidate.fileStem + ".", StringComparison.OrdinalIgnoreCase));
                if (match == null) continue;
                await AddArtifactAsync(result, request, revision, byPath, model, match.role, match.fileStem, false,
                    token, cancellationToken);
            }
            return Finish(result);
        }

        private static Task<Dictionary<string, DiscoveredArtifact[]>> DiscoverChatterboxAsync(
            ModelDownloadRequest request, string revision, Dictionary<string, HfSibling> byPath,
            HfSibling[] models, string token, CancellationToken cancellationToken) =>
            DiscoverFixedGraphsAsync(new ModelDownloadRequest(request.Repository, request.Revision,
                request.ExpectedFamily, new[]
                {
                    new ModelGraphRole("speech-encoder", "speech_encoder"),
                    new ModelGraphRole("embed-tokens", "embed_tokens"),
                    new ModelGraphRole("language-model", "language_model"),
                    new ModelGraphRole("conditional-decoder", "conditional_decoder")
                }), revision, byPath, models, token, cancellationToken);

        private static async Task<Dictionary<string, DiscoveredArtifact[]>> DiscoverOmniVoiceAsync(
            ModelDownloadRequest request, string revision, Dictionary<string, HfSibling> byPath,
            HfSibling[] models, string token, CancellationToken cancellationToken)
        {
            IEnumerable<HfSibling> selected = models;
            var result = new Dictionary<string, List<DiscoveredArtifact>>(StringComparer.Ordinal);
            var roles = new[]
            {
                (Stem: "audio_embeddings_encoder", Role: "audio-embeddings"),
                (Stem: "llm_decoder", Role: "language-decoder"),
                (Stem: "audio_heads_decoder", Role: "audio-heads"),
                (Stem: "acoustic_encoder", Role: "acoustic-encoder"),
                (Stem: "semantic_encoder", Role: "semantic-encoder"),
                (Stem: "quantizer_encoder", Role: "quantizer-encoder"),
                (Stem: "higgs_decoder", Role: "higgs-decoder"),
                (Stem: "omnivoice", Role: "merged-backbone")
            };
            foreach (HfSibling model in selected)
            {
                string stem = FileStem(model);
                var match = roles.FirstOrDefault(candidate =>
                    stem.Equals(candidate.Stem, StringComparison.OrdinalIgnoreCase) ||
                    stem.StartsWith(candidate.Stem + ".", StringComparison.OrdinalIgnoreCase) ||
                    stem.StartsWith(candidate.Stem + "_", StringComparison.OrdinalIgnoreCase));
                if (match.Stem == null) continue;
                await AddArtifactAsync(result, request, revision, byPath, model, match.Role, match.Stem,
                    match.Role == "merged-backbone", token, cancellationToken);
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
                (Path: "neutts.json", Role: "neutts-metadata"),
                (Path: "LICENSE", Role: "model-license"),
                (Path: "CODEC_LICENSE", Role: "codec-license"),
                (Path: "ATTRIBUTION.md", Role: "attribution"),
                (Path: "tokenizer.model", Role: "tokenizer-model"),
                (Path: "GEMMA_TERMS.txt", Role: "license"),
                (Path: "GEMMA_USE_POLICY.txt", Role: "use-policy"),
                (Path: "NOTICE.txt", Role: "notice"),
                (Path: "int4/tokenizer.json", Role: "tokenizer"),
                (Path: "tokenizer_config.json", Role: "tokenizer-config"),
                (Path: "int4/tokenizer_config.json", Role: "tokenizer-config"),
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
                (Path: "cangjie_mapping.json", Role: "cangjie"),
                (Path: "Cangjie5_TC.json", Role: "cangjie")
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

        private static Task<DiscoveredFile> DiscoverFileAsync(ModelDownloadRequest request, string revision,
            HfSibling source, string role, string token, CancellationToken cancellationToken)
        {
            string hash = source.lfs?.sha256;
            long size = source.lfs != null && source.lfs.size > 0 ? source.lfs.size : source.size;
            return Task.FromResult(new DiscoveredFile(role, source.rfilename, hash, size));
        }

        public static DiscoveredArtifact[] SelectArtifacts(DiscoveredRepository repository,
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
            {
                if (!repository.Artifacts.ContainsKey(role))
                    throw new InvalidDataException($"Repository does not contain model role '{role}'.");
                if (role == "decoder-with-past" && selected.TryGetValue("decoder", out var decoder) && decoder.IsMerged)
                    continue;
                if (!selected.ContainsKey(role)) selected[role] = SelectArtifact(repository, selection, role);
            }
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

        private static bool IsChatterbox(ModelDownloadRequest request, IEnumerable<HfSibling> files)
        {
            return string.Equals(request.ExpectedFamily, "chatterbox", StringComparison.OrdinalIgnoreCase) ||
                files.Any(file => FileStem(file).StartsWith("speech_encoder", StringComparison.OrdinalIgnoreCase));
        }

        private static bool IsOmniVoice(ModelDownloadRequest request) =>
            string.Equals(request.ExpectedFamily, "omnivoice", StringComparison.OrdinalIgnoreCase);

        private static string FileStem(HfSibling file)
        {
            return Path.GetFileNameWithoutExtension(file.rfilename);
        }

        private static void ValidateFile(DiscoveredFile file)
        {
            if (file == null || string.IsNullOrWhiteSpace(file.Role) || string.IsNullOrWhiteSpace(file.Path) ||
                (!string.IsNullOrEmpty(file.Sha256) && (file.Sha256.Length != 64 ||
                file.Sha256.Any(character => !Uri.IsHexDigit(character)))) || file.Size < 0)
                throw new InvalidDataException("Repository contains an invalid file entry.");
        }

        internal static string FileUrl(ModelDownloadRequest request, string revision, string path)
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

        private static void AddToken(HttpRequestMessage request, string token)
        {
            if (!string.IsNullOrWhiteSpace(token))
                request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", token.Trim());
        }

    }
    public sealed class DiscoveredRepository
    {
        public string Repository => Owner + "/" + Name;
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

    public sealed class DiscoveredArtifact
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

    public sealed class DiscoveredFile
    {
        public string Role { get; }
        public string Path { get; }
        public string Sha256 { get; internal set; }
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

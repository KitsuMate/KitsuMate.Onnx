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

namespace KitsuMate.Onnx.Download
{
    public sealed class DownloadedModel
    {
        public string DirectoryPath { get; }
        public ModelIdentity Identity { get; }
        public IReadOnlyList<DiscoveredFile> Files { get; }

        public DownloadedModel(string directory, ModelIdentity identity, IReadOnlyList<DiscoveredFile> files)
        { DirectoryPath = Path.GetFullPath(directory); Identity = identity; Files = files; }

        public DiscoveredFile GetFile(string role) => Files.FirstOrDefault(file => file.Role == role)
            ?? throw new InvalidDataException($"Downloaded model is missing '{role}'.");
        public string GetPath(string role) => ModelDownloadPaths.Child(DirectoryPath, GetFile(role).Path);

        public void ConfigureModel(OnnxModelReference reference, string role)
        {
            var file = GetFile(role);
            reference.ConfigureDownloadedFile(GetPath(role), file.Sha256, file.Inputs, file.Outputs);
        }
    }

    /// <summary>Downloads a resolved repository snapshot without Unity asset imports.</summary>
    public static class ModelDownloader
    {
        private static readonly HttpClient Client = new() { Timeout = Timeout.InfiniteTimeSpan };

        public static async Task<DownloadedModel> DownloadAsync(DiscoveredRepository repository,
            IReadOnlyDictionary<string, string> selection, string destination, string cacheDirectory = null,
            string token = null, IProgress<float> progress = null, CancellationToken cancellationToken = default)
        {
            var artifacts = HuggingFaceModelRepository.SelectArtifacts(repository, selection);
            var files = artifacts.SelectMany(artifact => artifact.Files).Concat(repository.CommonFiles)
                .GroupBy(file => file.Path, StringComparer.Ordinal).Select(group => group.First()).ToArray();
            long total = files.Sum(file => Math.Max(0, file.Size)), completed = 0;
            var request = new ModelDownloadRequest(repository.Repository, repository.Revision, repository.Family);
            foreach (var file in files)
            {
                cancellationToken.ThrowIfCancellationRequested();
                string path = ModelDownloadPaths.Child(destination, file.Path);
                Directory.CreateDirectory(Path.GetDirectoryName(path));
                string cache = !string.IsNullOrEmpty(cacheDirectory) && IsHash(file.Sha256)
                    ? ModelDownloadPaths.Child(cacheDirectory, file.Sha256.ToLowerInvariant()) : null;
                if (!await HasHashAsync(path, file.Sha256, cancellationToken))
                {
                    if (cache != null && await HasHashAsync(cache, file.Sha256, cancellationToken))
                        File.Copy(cache, path, true);
                    else
                    {
                        await DownloadFileAsync(HuggingFaceModelRepository.FileUrl(request, repository.Revision, file.Path),
                            path, file, token, bytes => progress?.Report(total > 0
                                ? Math.Min(.99f, (float)(completed + bytes) / total) : 0), cancellationToken);
                        if (cache != null)
                        {
                            Directory.CreateDirectory(Path.GetDirectoryName(cache));
                            File.Copy(path, cache, true);
                        }
                    }
                }
                if (file.Path.EndsWith(".onnx", StringComparison.OrdinalIgnoreCase))
                {
                    var metadata = await Task.Run(() => OnnxLightweightMetadataReader.Read(path), cancellationToken);
                    file.SetMetadata(metadata.Inputs.ToArray(), metadata.Outputs.ToArray());
                }
                completed += Math.Max(0, file.Size);
                progress?.Report(total > 0 ? Math.Min(.99f, (float)completed / total) : 0);
            }
            string content = string.Join("\n", files.OrderBy(file => file.Path, StringComparer.Ordinal)
                .Select(file => file.Role + ":" + file.Path + ":" + file.Sha256));
            using var hash = SHA256.Create();
            var identity = new ModelIdentity(repository.Family, repository.Name, repository.Revision,
                Hex(hash.ComputeHash(Encoding.UTF8.GetBytes(content))));
            progress?.Report(1);
            return new DownloadedModel(destination, identity, files);
        }

        private static async Task DownloadFileAsync(string url, string path, DiscoveredFile file, string token,
            Action<long> progress, CancellationToken ct)
        {
            string partial = path + ".partial";
            try
            {
                using var request = new HttpRequestMessage(HttpMethod.Get, url);
                if (!string.IsNullOrWhiteSpace(token)) request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", token);
                using var response = await Client.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, ct);
                response.EnsureSuccessStatusCode();
                using var source = await response.Content.ReadAsStreamAsync();
                using var hash = SHA256.Create();
                long bytes = 0;
                using (var output = new FileStream(partial, FileMode.Create, FileAccess.Write, FileShare.None, 65536, true))
                {
                    var buffer = new byte[65536];
                    int count;
                    while ((count = await source.ReadAsync(buffer, 0, buffer.Length, ct)) != 0)
                    {
                        await output.WriteAsync(buffer, 0, count, ct);
                        hash.TransformBlock(buffer, 0, count, buffer, 0);
                        bytes += count;
                        progress(bytes);
                    }
                    hash.TransformFinalBlock(Array.Empty<byte>(), 0, 0);
                }
                ct.ThrowIfCancellationRequested();
                string actual = Hex(hash.Hash);
                if (file.Size > 0 && bytes != file.Size || !string.IsNullOrEmpty(file.Sha256) &&
                    !string.Equals(actual, file.Sha256, StringComparison.OrdinalIgnoreCase))
                    throw new InvalidDataException($"Download verification failed for '{file.Path}'.");
                file.Sha256 = actual;
                if (File.Exists(path)) File.Delete(path);
                File.Move(partial, path);
            }
            finally { if (File.Exists(partial)) File.Delete(partial); }
        }

        private static async Task<bool> HasHashAsync(string path, string expected, CancellationToken ct)
        {
            if (!IsHash(expected) || !File.Exists(path)) return false;
            return await Task.Run(() =>
            {
                ct.ThrowIfCancellationRequested();
                using var stream = File.OpenRead(path);
                using var hash = SHA256.Create();
                return string.Equals(Hex(hash.ComputeHash(stream)), expected, StringComparison.OrdinalIgnoreCase);
            }, ct);
        }
        private static bool IsHash(string value) => value?.Length == 64 && value.All(Uri.IsHexDigit);
        private static string Hex(byte[] bytes) => BitConverter.ToString(bytes).Replace("-", "").ToLowerInvariant();
    }

    internal static class ModelDownloadPaths
    {
        internal static string Child(string root, string relative)
        {
            string safe = HuggingFaceModelRepository.SafeRelativePath(relative);
            string fullRoot = Path.GetFullPath(root).TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
            string full = Path.GetFullPath(Path.Combine(fullRoot, safe));
            if (!full.StartsWith(fullRoot + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase))
                throw new InvalidDataException("Model path escapes its storage directory.");
            return full;
        }
    }
}

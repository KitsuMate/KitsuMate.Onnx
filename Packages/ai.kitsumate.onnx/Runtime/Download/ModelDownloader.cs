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
        public string Repository { get; }

        public DownloadedModel(string directory, ModelIdentity identity, IReadOnlyList<DiscoveredFile> files, string repository = null)
        { DirectoryPath = Path.GetFullPath(directory); Identity = identity; Files = files; Repository = repository; }

        public DiscoveredFile GetFile(string role) => Files.FirstOrDefault(file => file.Role == role)
            ?? throw new InvalidDataException($"Downloaded model is missing '{role}'.");
        public string GetPath(string role) => ModelDownloadPaths.Child(DirectoryPath, GetFile(role).Path);

        public void ConfigureModel(OnnxModelReference reference, string role)
        {
            var file = GetFile(role);
            string path = GetPath(role);
            reference.ConfigureFile(path, file.Sha256, file.Inputs, file.Outputs);
        }
    }

    /// <summary>Downloads a resolved repository snapshot without Unity asset imports.</summary>
    public static class ModelDownloader
    {
        private static readonly HttpClient Client = new() { Timeout = Timeout.InfiniteTimeSpan };

        public static Task DownloadFileAsync(DiscoveredRepository repository, DiscoveredFile file,
            string destination, string token = null, IProgress<float> progress = null,
            CancellationToken cancellationToken = default)
        {
            string path = ModelDownloadPaths.Child(destination, file.Path);
            Directory.CreateDirectory(Path.GetDirectoryName(path));
            var request = new ModelDownloadRequest(repository.Repository, repository.Revision, repository.Family);
            return DownloadFileAsync(HuggingFaceModelRepository.FileUrl(request, repository.Revision, file.Path),
                path, file, token, bytes => progress?.Report(file.Size > 0 ? (float)bytes / file.Size : 0), cancellationToken);
        }

        public static async Task<DownloadedModel> DownloadAsync(DiscoveredRepository repository,
            IReadOnlyDictionary<string, string> selection, string destination, string cacheDirectory = null,
            string token = null, IProgress<float> progress = null, CancellationToken cancellationToken = default,
            Action<string, long, long> fileProgress = null)
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
                if (!await Task.Run(() => HasVerifiedContent(path, file), cancellationToken).ConfigureAwait(false))
                {
                    if (cache != null && await Task.Run(() => HasVerifiedContent(cache, file), cancellationToken).ConfigureAwait(false))
                        File.Copy(cache, path, true);
                    else
                    {
                        await DownloadFileAsync(HuggingFaceModelRepository.FileUrl(request, repository.Revision, file.Path),
                            path, file, token, bytes =>
                            {
                                fileProgress?.Invoke(file.Path, bytes, file.Size);
                                progress?.Report(total > 0 ? Math.Min(.99f, (float)(completed + bytes) / total) : 0);
                            }, cancellationToken).ConfigureAwait(false);
                        if (cache != null)
                        {
                            Directory.CreateDirectory(Path.GetDirectoryName(cache));
                            File.Copy(path, cache, true);
                        }
                    }
                }
                if (file.Path.EndsWith(".onnx", StringComparison.OrdinalIgnoreCase))
                {
                    var metadata = await Task.Run(() => OnnxLightweightMetadataReader.Read(path), cancellationToken).ConfigureAwait(false);
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
            return new DownloadedModel(destination, identity, files, repository.Repository);
        }

        private static async Task DownloadFileAsync(string url, string path, DiscoveredFile file, string token,
            Action<long> progress, CancellationToken ct)
        {
            string partial = path + ".partial";
            try
            {
                using var request = new HttpRequestMessage(HttpMethod.Get, url);
                if (!string.IsNullOrWhiteSpace(token)) request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", token);
                // Streaming must not resume on Unity's Editor synchronization context per chunk.
                using var response = await Client.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, ct).ConfigureAwait(false);
                using var cancelResponse = ct.Register(response.Dispose);
                response.EnsureSuccessStatusCode();
                using var source = await response.Content.ReadAsStreamAsync().ConfigureAwait(false);
                long bytes = 0;
                var progressTimer = System.Diagnostics.Stopwatch.StartNew();
                using (var output = new FileStream(partial, FileMode.Create, FileAccess.Write, FileShare.None, 65536, true))
                {
                    var buffer = new byte[65536];
                    int count;
                    while ((count = await source.ReadAsync(buffer, 0, buffer.Length, ct).ConfigureAwait(false)) != 0)
                    {
                        await output.WriteAsync(buffer, 0, count, ct).ConfigureAwait(false);
                        bytes += count;
                        if (bytes == count || progressTimer.ElapsedMilliseconds >= 100 || bytes == file.Size)
                        {
                            progress(bytes);
                            progressTimer.Restart();
                        }
                    }
                }
                ct.ThrowIfCancellationRequested();
                if (file.Size > 0 && bytes != file.Size)
                    throw new InvalidDataException($"Download size does not match for '{file.Path}'.");
                if (!HasVerifiedContent(partial, file))
                    throw new InvalidDataException($"Downloaded file '{file.Path}' does not match its repository hash.");
                if (File.Exists(path)) File.Delete(path);
                File.Move(partial, path);
            }
            catch (Exception) when (ct.IsCancellationRequested) { throw new OperationCanceledException(ct); }
            finally { if (File.Exists(partial)) File.Delete(partial); }
        }

        private static bool HasVerifiedContent(string path, DiscoveredFile file)
        {
            if (!File.Exists(path) || new FileInfo(path).Length != file.Size) return false;
            if (!IsHash(file.Sha256)) return true;
            using var stream = File.OpenRead(path);
            using var sha = SHA256.Create();
            return string.Equals(Hex(sha.ComputeHash(stream)), file.Sha256, StringComparison.OrdinalIgnoreCase);
        }

        internal static bool HasSize(string path, long expected) =>
            expected > 0 && File.Exists(path) && new FileInfo(path).Length == expected;
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

using System;
using System.Collections.Concurrent;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Threading;
using System.Threading.Tasks;

namespace KitsuMate.Onnx
{
    internal sealed class OnnxRuntimeEnvironment
    {
        public OnnxRuntimeEnvironment(string modelStorageRoot, string streamingAssetsPath, string persistentDataPath)
        {
            ModelStorageRoot = modelStorageRoot;
            StreamingAssetsPath = streamingAssetsPath;
            PersistentDataPath = persistentDataPath;
        }

        public string ModelStorageRoot { get; }
        public string StreamingAssetsPath { get; }
        public string PersistentDataPath { get; }
    }

    internal interface IOnnxModelContentFetcher
    {
        Task FetchAsync(string sourceUri, string destinationPath, CancellationToken cancellationToken);
    }

    internal sealed class AndroidModelStager
    {
        private readonly IOnnxModelContentFetcher fetcher;
        private readonly ConcurrentDictionary<string, SemaphoreSlim> gates = new(StringComparer.Ordinal);

        public AndroidModelStager(IOnnxModelContentFetcher fetcher) =>
            this.fetcher = fetcher ?? throw new ArgumentNullException(nameof(fetcher));

        public async Task<string> StageAsync(
            string relativePath,
            string sha256,
            OnnxRuntimeEnvironment environment,
            CancellationToken cancellationToken)
        {
            string normalized = NormalizeRelativePath(relativePath);
            if (!IsSha256(sha256))
                throw new OnnxModelPreparationException(
                    $"Android filesystem model '{relativePath}' requires a valid SHA-256 checksum.");

            string root = Path.GetFullPath(Path.Combine(environment.PersistentDataPath, "KitsuMateModels"));
            string target = CombineContained(root, normalized);
            SemaphoreSlim gate = gates.GetOrAdd(target, _ => new SemaphoreSlim(1, 1));
            await gate.WaitAsync(cancellationToken).ConfigureAwait(false);
            try
            {
                cancellationToken.ThrowIfCancellationRequested();
                if (File.Exists(target) && await HasHashAsync(target, sha256, cancellationToken).ConfigureAwait(false))
                    return target;

                Directory.CreateDirectory(Path.GetDirectoryName(target));
                foreach (string interrupted in Directory.EnumerateFiles(
                             Path.GetDirectoryName(target), Path.GetFileName(target) + ".partial.*"))
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    try { File.Delete(interrupted); }
                    catch (IOException) { /* Another process may still own a stale candidate; unique names avoid collision. */ }
                }
                string partial = target + ".partial." + Guid.NewGuid().ToString("N");
                try
                {
                    string escaped = string.Join("/", normalized.Split('/').Select(Uri.EscapeDataString));
                    string uri = environment.StreamingAssetsPath.TrimEnd('/') + "/KitsuMateModels/" + escaped;
                    await fetcher.FetchAsync(uri, partial, cancellationToken).ConfigureAwait(false);
                    cancellationToken.ThrowIfCancellationRequested();
                    if (!await HasHashAsync(partial, sha256, cancellationToken).ConfigureAwait(false))
                        throw new OnnxModelPreparationException($"Checksum mismatch while staging Android model '{relativePath}'.");

                    string backup = target + ".backup." + Guid.NewGuid().ToString("N");
                    if (File.Exists(target))
                    {
                        try { File.Replace(partial, target, backup); }
                        catch (PlatformNotSupportedException) { File.Delete(target); File.Move(partial, target); }
                        finally { if (File.Exists(backup)) File.Delete(backup); }
                    }
                    else File.Move(partial, target);
                    return target;
                }
                finally
                {
                    if (File.Exists(partial)) File.Delete(partial);
                }
            }
            catch (OperationCanceledException) { throw; }
            catch (OnnxModelPreparationException) { throw; }
            catch (Exception exception)
            {
                throw new OnnxModelPreparationException($"Failed to stage Android model '{relativePath}'.", exception);
            }
            finally
            {
                gate.Release();
            }
        }

        internal static string NormalizeRelativePath(string path)
        {
            if (string.IsNullOrWhiteSpace(path) || Path.IsPathRooted(path))
                throw new OnnxModelPreparationException("Model path must be relative to the model root.");
            string normalized = path.Replace('\\', '/').TrimStart('/');
            if (normalized.Split('/').Any(part => part is ".." or "." or ""))
                throw new OnnxModelPreparationException($"Model path '{path}' is not a normalized contained path.");
            return normalized;
        }

        internal static string CombineContained(string root, string relativePath)
        {
            string absoluteRoot = Path.GetFullPath(root);
            string candidate = Path.GetFullPath(Path.Combine(absoluteRoot, relativePath));
            string prefix = absoluteRoot.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar) + Path.DirectorySeparatorChar;
            if (!candidate.StartsWith(prefix, StringComparison.Ordinal))
                throw new OnnxModelPreparationException($"Model path '{relativePath}' escapes the model root.");
            return candidate;
        }

        internal static bool IsSha256(string value) =>
            value?.Length == 64 && value.All(character => Uri.IsHexDigit(character));

        private static async Task<bool> HasHashAsync(string path, string expected, CancellationToken cancellationToken)
        {
            string actual = await Task.Run(() =>
            {
                using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
                using var hash = SHA256.Create();
                return string.Concat(hash.ComputeHash(stream).Select(value => value.ToString("x2")));
            }, cancellationToken).ConfigureAwait(false);
            cancellationToken.ThrowIfCancellationRequested();
            return string.Equals(actual, expected, StringComparison.OrdinalIgnoreCase);
        }
    }
}

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

namespace KitsuMate.Onnx.Download
{
    /// <summary>One managed installation. Callers coordinate its use with loaded inference sessions.</summary>
    public sealed class ModelInstallationStore
    {
        private const string RecordName = "installation.json";
        public string DirectoryPath { get; }
        public bool IsInstalled => Read() != null;
        /// <summary>Whether an update was promoted but its Editor binding was not finalized.</summary>
        public bool HasPendingBinding => Directory.Exists(DirectoryPath + ".previous");
        public ModelInstallationStore(string root, string installationId) => DirectoryPath = ModelDownloadPaths.Child(root, installationId);

        public DownloadedModel Read(bool allowIncomplete = false)
        {
            string recordPath = Path.Combine(DirectoryPath, RecordName);
            if (!File.Exists(recordPath)) return null;
            try
            {
                var record = JsonUtility.FromJson<Record>(File.ReadAllText(recordPath));
                if (record?.files == null || record.files.Length == 0) return null;
                var files = record.files.Where(file =>
                    !allowIncomplete || ModelDownloader.HasSize(ModelDownloadPaths.Child(DirectoryPath, file.path), file.size)).Select(file =>
                {
                    string path = ModelDownloadPaths.Child(DirectoryPath, file.path);
                    if (!File.Exists(path) || new FileInfo(path).Length != file.size) throw new InvalidDataException("Incomplete installation.");
                    var result = new DiscoveredFile(file.role, file.path, file.hash, file.size);
                    result.SetMetadata(file.inputs, file.outputs);
                    return result;
                }).ToArray();
                return new DownloadedModel(DirectoryPath,
                    new ModelIdentity(record.family, record.model, record.revision, record.hash), files, record.repository);
            }
            catch (Exception ex) when (ex is IOException || ex is InvalidDataException || ex is ArgumentException) { return null; }
        }

        public async Task InstallAsync(ModelDownloadProfile profile,
            IProgress<float> progress = null, CancellationToken cancellationToken = default, string token = null)
        {
            var request = profile.CreateRequest();
            var selection = profile.CreateSelection();
            var repository = await HuggingFaceModelRepository.ScanAsync(request, token, cancellationToken).ConfigureAwait(false);
            await InstallAsync(repository, selection, progress, cancellationToken, token).ConfigureAwait(false);
        }

        public string DownloadDirectory(DiscoveredRepository repository) =>
            ModelDownloadPaths.Child(DirectoryPath + ".downloading", repository.Repository + "/" + repository.Revision);

        public static DiscoveredFile[] RequiredFiles(DiscoveredRepository repository, IReadOnlyDictionary<string, string> selection) =>
            HuggingFaceModelRepository.SelectArtifacts(repository, selection).SelectMany(artifact => artifact.Files)
                .Concat(repository.CommonFiles).GroupBy(file => file.Path, StringComparer.Ordinal).Select(group => group.First()).ToArray();

        /// <summary>Finds completed files without reading model contents or computing hashes.</summary>
        public IReadOnlyDictionary<string, string> AvailableFiles(DiscoveredRepository repository,
            IEnumerable<DiscoveredFile> required, IReadOnlyDictionary<string, string> supplied = null)
        {
            var available = new Dictionary<string, string>(StringComparer.Ordinal);
            var installed = Read(allowIncomplete: true);
            string staging = DownloadDirectory(repository);
            foreach (var file in required)
            {
                string path = ModelDownloadPaths.Child(staging, file.Path);
                if (supplied != null && supplied.TryGetValue(file.Path, out string local) && ModelDownloader.HasSize(local, file.Size))
                    available[file.Path] = local;
                else if (ModelDownloader.HasSize(path, file.Size)) available[file.Path] = path;
                else if (installed != null && installed.Identity.Revision == repository.Revision &&
                    installed.Identity.ModelId == repository.Name && installed.Identity.Family == repository.Family &&
                    (string.IsNullOrEmpty(installed.Repository) || installed.Repository == repository.Repository) &&
                    installed.Files.Any(existing => existing.Path == file.Path && existing.Size == file.Size && existing.Sha256 == file.Sha256))
                    available[file.Path] = ModelDownloadPaths.Child(DirectoryPath, file.Path);
            }
            return available;
        }

        public async Task InstallAsync(DiscoveredRepository repository, IReadOnlyDictionary<string, string> selection,
            IProgress<float> progress = null, CancellationToken cancellationToken = default, string token = null,
            IReadOnlyDictionary<string, string> suppliedFiles = null, Action<string, long, long> fileProgress = null,
            bool retainPreviousUntilBinding = false)
        {
            cancellationToken.ThrowIfCancellationRequested();
            using var installationLock = AcquireLock();
            string staging = DownloadDirectory(repository);
            var required = RequiredFiles(repository, selection);
            PruneOtherStaging(staging);
            if (Directory.Exists(staging)) PruneUnselectedFiles(staging, required);
            var available = AvailableFiles(repository, required, suppliedFiles);
            foreach (var file in required)
            {
                cancellationToken.ThrowIfCancellationRequested();
                if (!available.TryGetValue(file.Path, out string source)) continue;
                string destination = ModelDownloadPaths.Child(staging, file.Path);
                if (Path.GetFullPath(source) == destination) continue;
                Directory.CreateDirectory(Path.GetDirectoryName(destination));
                string copying = destination + ".copying";
                try
                {
                    using (var input = new FileStream(source, FileMode.Open, FileAccess.Read, FileShare.Read, 65536, true))
                    using (var output = new FileStream(copying, FileMode.Create, FileAccess.Write, FileShare.None, 65536, true))
                        await input.CopyToAsync(output, 65536, cancellationToken).ConfigureAwait(false);
                    cancellationToken.ThrowIfCancellationRequested();
                    if (File.Exists(destination)) File.Delete(destination);
                    File.Move(copying, destination);
                    File.SetLastWriteTimeUtc(destination, File.GetLastWriteTimeUtc(source));
                }
                finally { if (File.Exists(copying)) File.Delete(copying); }
            }
            // Completed files survive cancellation. The interrupted file is restarted on retry.
            var result = await ModelDownloader.DownloadAsync(repository, selection, staging, token: token, progress: progress,
                cancellationToken: cancellationToken, fileProgress: fileProgress).ConfigureAwait(false);
            WriteRecord(result);
            cancellationToken.ThrowIfCancellationRequested();
            string previous = DirectoryPath + ".previous";
            if (Directory.Exists(previous)) throw new IOException($"Remove or recover the previous installation at {previous} before replacing it.");
            bool replacing = Directory.Exists(DirectoryPath);
            if (replacing) Directory.Move(DirectoryPath, previous);
            try
            {
                Directory.Move(staging, DirectoryPath);
            }
            catch
            {
                if (replacing) Directory.Move(previous, DirectoryPath);
                throw;
            }
            if (replacing && !retainPreviousUntilBinding) Directory.Delete(previous, true);
        }

        internal static void PruneUnselectedFiles(string staging, IReadOnlyList<DiscoveredFile> selectedFiles)
        {
            var selectedPaths = new HashSet<string>(selectedFiles.Select(file =>
                ModelDownloadPaths.Child(staging, file.Path)), StringComparer.OrdinalIgnoreCase);
            foreach (string path in Directory.GetFiles(staging, "*", SearchOption.AllDirectories))
            {
                if (!selectedPaths.Contains(path) &&
                    !string.Equals(path, Path.Combine(staging, RecordName), StringComparison.OrdinalIgnoreCase))
                    File.Delete(path);
            }
            foreach (string path in Directory.GetDirectories(staging, "*", SearchOption.AllDirectories)
                         .OrderByDescending(path => path.Length))
            {
                if (!Directory.EnumerateFileSystemEntries(path).Any()) Directory.Delete(path);
            }
        }

        private void PruneOtherStaging(string staging)
        {
            string root = DirectoryPath + ".downloading";
            if (!Directory.Exists(root)) return;
            string current = Path.GetFullPath(staging).TrimEnd(Path.DirectorySeparatorChar) + Path.DirectorySeparatorChar;
            foreach (string path in Directory.GetFiles(root, "*", SearchOption.AllDirectories))
            {
                if (!path.StartsWith(current, StringComparison.OrdinalIgnoreCase)) File.Delete(path);
            }
            foreach (string path in Directory.GetDirectories(root, "*", SearchOption.AllDirectories)
                         .OrderByDescending(path => path.Length))
            {
                if (!Directory.EnumerateFileSystemEntries(path).Any()) Directory.Delete(path);
            }
        }

        /// <summary>Called by an Editor binding operation after the new references are saved.</summary>
        public void CompleteBinding()
        {
            using var installationLock = AcquireLock();
            string previous = DirectoryPath + ".previous";
            if (Directory.Exists(previous)) Directory.Delete(previous, true);
        }

        /// <summary>Restores the last working installation, or removes a failed first install.</summary>
        public void RestorePrevious()
        {
            using var installationLock = AcquireLock();
            string previous = DirectoryPath + ".previous";
            if (Directory.Exists(DirectoryPath)) Directory.Delete(DirectoryPath, true);
            if (Directory.Exists(previous)) Directory.Move(previous, DirectoryPath);
        }

        /// <summary>Records a validated local installation for the application and Editor to share.</summary>
        internal static void WriteRecord(DownloadedModel result)
        {
            var record = new Record
            {
                family = result.Identity.Family, model = result.Identity.ModelId,
                repository = result.Repository,
                revision = result.Identity.Revision, hash = result.Identity.ContentHash,
                files = result.Files.Select(file => new FileRecord
                {
                    role = file.Role, path = file.Path, hash = file.Sha256,
                    size = new FileInfo(ModelDownloadPaths.Child(result.DirectoryPath, file.Path)).Length,
                    inputs = file.Inputs, outputs = file.Outputs
                }).ToArray()
            };
            File.WriteAllText(Path.Combine(result.DirectoryPath, RecordName), JsonUtility.ToJson(record));
        }

        public async Task DownloadFileAsync(DiscoveredRepository repository, DiscoveredFile file,
            string token = null, IProgress<float> progress = null, CancellationToken cancellationToken = default)
        {
            using var installationLock = AcquireLock();
            await ModelDownloader.DownloadFileAsync(repository, file, DownloadDirectory(repository), token,
                progress, cancellationToken).ConfigureAwait(false);
        }

        internal FileStream AcquireLock()
        {
            Directory.CreateDirectory(Path.GetDirectoryName(DirectoryPath));
            try
            {
                return new FileStream(DirectoryPath + ".lock", FileMode.OpenOrCreate, FileAccess.ReadWrite,
                    FileShare.None, 1, FileOptions.DeleteOnClose);
            }
            catch (IOException exception)
            {
                throw new IOException("This model installation is already in use. Cancel or finish the other download before retrying.", exception);
            }
        }

        public void Uninstall()
        {
            using var installationLock = AcquireLock();
            if (Directory.Exists(DirectoryPath)) Directory.Delete(DirectoryPath, true);
            if (Directory.Exists(DirectoryPath + ".downloading")) Directory.Delete(DirectoryPath + ".downloading", true);
        }

        [Serializable] private sealed class Record
        {
            public string repository, family, model, revision, hash;
            public FileRecord[] files;
        }
        [Serializable] private sealed class FileRecord
        {
            public string role, path, hash;
            public long size;
            public OnnxModelAsset.TensorInfo[] inputs, outputs;
        }
    }
}

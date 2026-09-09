using System;
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
        public ModelInstallationStore(string root, string installationId) => DirectoryPath = ModelDownloadPaths.Child(root, installationId);

        public DownloadedModel Read()
        {
            string recordPath = Path.Combine(DirectoryPath, RecordName);
            if (!File.Exists(recordPath)) return null;
            try
            {
                var record = JsonUtility.FromJson<Record>(File.ReadAllText(recordPath));
                if (record?.files == null || record.files.Length == 0) return null;
                var files = record.files.Select(file =>
                {
                    string path = ModelDownloadPaths.Child(DirectoryPath, file.path);
                    if (!File.Exists(path) || new FileInfo(path).Length != file.size) throw new InvalidDataException("Incomplete installation.");
                    var result = new DiscoveredFile(file.role, file.path, file.hash, file.size);
                    result.SetMetadata(file.inputs, file.outputs);
                    return result;
                }).ToArray();
                return new DownloadedModel(DirectoryPath,
                    new ModelIdentity(record.family, record.model, record.revision, record.hash), files);
            }
            catch (Exception ex) when (ex is IOException || ex is ArgumentException) { return null; }
        }

        public async Task InstallAsync(ModelDownloadProfile profile, Action<DownloadedModel> validate,
            IProgress<float> progress = null, CancellationToken cancellationToken = default)
        {
            var request = profile.CreateRequest();
            var selection = profile.CreateSelection();
            string staging = DirectoryPath + ".installing";
            try
            {
                if (Directory.Exists(staging)) Directory.Delete(staging, true);
                var repository = await HuggingFaceModelRepository.ScanAsync(request, cancellationToken: cancellationToken);
                var result = await ModelDownloader.DownloadAsync(repository, selection, staging, progress: progress,
                    cancellationToken: cancellationToken);
                validate?.Invoke(result);
                var record = new Record
                {
                    family = result.Identity.Family, model = result.Identity.ModelId,
                    revision = result.Identity.Revision, hash = result.Identity.ContentHash,
                    files = result.Files.Select(file => new FileRecord
                    {
                        role = file.Role, path = file.Path, hash = file.Sha256,
                        size = new FileInfo(ModelDownloadPaths.Child(result.DirectoryPath, file.Path)).Length,
                        inputs = file.Inputs, outputs = file.Outputs
                    }).ToArray()
                };
                File.WriteAllText(Path.Combine(staging, RecordName), JsonUtility.ToJson(record));
                cancellationToken.ThrowIfCancellationRequested();
                if (Directory.Exists(DirectoryPath)) Directory.Delete(DirectoryPath, true);
                Directory.Move(staging, DirectoryPath);
            }
            finally { if (Directory.Exists(staging)) Directory.Delete(staging, true); }
        }

        public void Uninstall() { if (Directory.Exists(DirectoryPath)) Directory.Delete(DirectoryPath, true); }

        [Serializable] private sealed class Record
        {
            public string family, model, revision, hash;
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

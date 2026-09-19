using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;
using KitsuMate.Onnx.Download;

namespace KitsuMate.Onnx
{
    [Serializable]
    public sealed class TextFileReference
    {
        [SerializeField] private OnnxModelReference.SourceKind sourceKind;
        [SerializeField] private TextAsset asset;
        [SerializeField] private OnnxModelReference.FileRoot fileRoot;
        [SerializeField] private string filePath;
        [NonSerialized] private string preparedRuntimePath;

        public OnnxModelReference.SourceKind Kind => sourceKind;
        public TextAsset Asset => asset;
        public bool IsAvailable
        {
            get
            {
                try { return sourceKind == OnnxModelReference.SourceKind.Asset ? asset != null : File.Exists(ResolveFile()); }
                catch (ArgumentException) { return false; }
                catch (IOException) { return false; }
            }
        }
        public byte[] bytes => sourceKind == OnnxModelReference.SourceKind.Asset ? asset.bytes : File.ReadAllBytes(ResolveFile());
        public string text => sourceKind == OnnxModelReference.SourceKind.Asset ? asset.text : File.ReadAllText(ResolveFile());
        public string name => sourceKind == OnnxModelReference.SourceKind.Asset ? asset?.name : Path.GetFileName(filePath);
        public string ResolveFile()
        {
            if (string.IsNullOrWhiteSpace(filePath)) return null;
            if (!string.IsNullOrWhiteSpace(preparedRuntimePath) && File.Exists(preparedRuntimePath))
                return preparedRuntimePath;
            if (fileRoot == OnnxModelReference.FileRoot.PersistentData)
                return ModelDownloadPaths.Child(Application.persistentDataPath, filePath);
            if (fileRoot == OnnxModelReference.FileRoot.StreamingAssets)
            {
                string root = Application.streamingAssetsPath;
                return root.Contains("://") ? null : ModelDownloadPaths.Child(root, filePath);
            }
            return Path.GetFullPath(filePath);
        }

        internal async Task PrepareForRuntimeAsync(CancellationToken cancellationToken)
        {
            if (sourceKind != OnnxModelReference.SourceKind.File) return;
            preparedRuntimePath = fileRoot == OnnxModelReference.FileRoot.StreamingAssets
                ? await StreamingAssetFiles.PrepareAsync(filePath, cancellationToken)
                : ResolveFile();
            if (string.IsNullOrWhiteSpace(preparedRuntimePath) || !File.Exists(preparedRuntimePath))
                throw new OnnxModelPreparationException($"Text file '{filePath}' was not found.");
        }

        public void ConfigureAsset(TextAsset value) { sourceKind = OnnxModelReference.SourceKind.Asset; asset = value; filePath = string.Empty; preparedRuntimePath = null; }
        public static implicit operator TextFileReference(TextAsset value)
        {
            var reference = new TextFileReference();
            reference.ConfigureAsset(value);
            return reference;
        }
        public void ConfigureFile(string path)
        {
            // Use the same portable path representation as ONNX graph references.
            var model = new OnnxModelReference();
            model.ConfigureFile(path, null, null, null);
            sourceKind = OnnxModelReference.SourceKind.File;
            filePath = model.FilePath;
            fileRoot = model.Root;
            asset = null;
            preparedRuntimePath = null;
        }
        public static TextFileReference FromFile(string path)
        {
            var reference = new TextFileReference();
            reference.ConfigureFile(path);
            return reference;
        }
    }
}

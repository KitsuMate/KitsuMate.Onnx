using System;
using System.IO;
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
        public string ResolveFile() => string.IsNullOrWhiteSpace(filePath) ? null : fileRoot == OnnxModelReference.FileRoot.PersistentData
            ? ModelDownloadPaths.Child(Application.persistentDataPath, filePath) : Path.GetFullPath(filePath);

        public void ConfigureAsset(TextAsset value) { sourceKind = OnnxModelReference.SourceKind.Asset; asset = value; filePath = string.Empty; }
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
        }
        public static TextFileReference FromFile(string path)
        {
            var reference = new TextFileReference();
            reference.ConfigureFile(path);
            return reference;
        }
    }
}

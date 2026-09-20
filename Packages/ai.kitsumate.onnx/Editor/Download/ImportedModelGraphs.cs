using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using KitsuMate.Onnx.Download;
using UnityEditor;
using UnityEngine;

namespace KitsuMate.Onnx.Editor.Download
{
    /// <summary>Explicit build preparation for graphs and their external weights.</summary>
    public static class ImportedModelGraphs
    {
        public static void MoveAssignedFilesToData(ModelSet set)
        {
            var settings = OnnxSettings.Load() ??
                throw new InvalidOperationException("Configure OnnxSettings before moving model files.");
            MoveAssignedFilesToData(set, settings.InstallationRoot);
        }

        internal static void MoveAssignedFilesToData(ModelSet set, string installationRoot)
        {
            string assetPath = AssetDatabase.GetAssetPath(set);
            if (string.IsNullOrEmpty(assetPath)) throw new InvalidOperationException("Save the model set first.");
            string directory = Path.Combine(installationRoot, "Local", AssetDatabase.AssetPathToGUID(assetPath));
            var destinations = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
            try
            {
                int index = 0;
                foreach (var source in set.GetAllModels())
                {
                    string folder = Path.Combine(directory, "model-" + index++);
                    if (source is not OnnxModelReference reference || reference.Kind != OnnxModelReference.SourceKind.Asset || reference.Asset == null) continue;
                    string original = AssetDatabase.GetAssetPath(reference.Asset);
                    if (!original.EndsWith(".onnx", StringComparison.OrdinalIgnoreCase) && !original.EndsWith(".ort", StringComparison.OrdinalIgnoreCase))
                        throw new InvalidOperationException($"{reference.SourceName} has no movable ONNX source file.");
                    if (!destinations.TryGetValue(original, out string destination))
                    {
                        destination = Path.Combine(folder, Path.GetFileName(original));
                        MoveGraphFiles(original, destination);
                        destinations.Add(original, destination);
                    }
                    reference.ConfigureFile(destination, reference.Sha256, reference.Inputs.ToArray(), reference.Outputs.ToArray());
                }
                index = 0;
                foreach (var reference in set.GetAllTextFiles())
                {
                    string folder = Path.Combine(directory, "text-" + index++);
                    if (reference == null || reference.Kind != OnnxModelReference.SourceKind.Asset || reference.Asset == null) continue;
                    string original = AssetDatabase.GetAssetPath(reference.Asset);
                    if (string.IsNullOrEmpty(original)) throw new InvalidOperationException("Save the text asset before moving it.");
                    if (original.EndsWith(".asset", StringComparison.OrdinalIgnoreCase))
                        throw new InvalidOperationException("This serialized TextAsset has no original text source file. Assign its text file before moving it.");
                    if (!destinations.TryGetValue(original, out string destination))
                    {
                        destination = Path.Combine(folder, Path.GetFileName(original));
                        ModelBuildFiles.Move(original, destination);
                        destinations.Add(original, destination);
                    }
                    reference.ConfigureFile(destination);
                }
            }
            finally
            {
                EditorUtility.SetDirty(set);
                AssetDatabase.SaveAssetIfDirty(set);
                AssetDatabase.Refresh(ImportAssetOptions.ForceSynchronousImport);
            }
        }

        private static void MoveGraphFiles(string source, string destination)
        {
            var copied = new List<string>();
            try
            {
                var references = ModelFileUtility.IsOnnxModelPath(source)
                    ? OnnxLightweightMetadataReader.Read(source).ExternalData
                    : new List<OnnxLightweightMetadataReader.ExternalDataReference>();
                foreach (var reference in references)
                    OnnxLightweightMetadataReader.ResolveExternalDataPath(source, reference);
                foreach (var reference in references.GroupBy(item => item.Location, StringComparer.OrdinalIgnoreCase)
                             .Select(group => group.First()))
                {
                    string data = OnnxLightweightMetadataReader.ResolveExternalDataPath(source, reference);
                    string target = Path.GetFullPath(Path.Combine(Path.GetDirectoryName(destination), reference.Location));
                    if (string.Equals(data, target, StringComparison.OrdinalIgnoreCase)) continue;
                    if (File.Exists(target)) throw new IOException($"External data already exists at {target}.");
                    ModelBuildFiles.Copy(data, target);
                    copied.Add(target);
                }
                ModelBuildFiles.Move(source, destination);
            }
            catch
            {
                foreach (string path in copied)
                {
                    if (File.Exists(path)) File.Delete(path);
                    if (File.Exists(path + ".meta")) File.Delete(path + ".meta");
                }
                throw;
            }
        }

        public static void MoveAssignedFiles(ModelSet set)
        {
            string directory = ModelBuildFiles.DirectoryFor(set);
            if (string.IsNullOrEmpty(directory)) throw new InvalidOperationException("Save the model set first.");
            try
            {
                int index = 0;
                foreach (var source in set.GetAllModels())
                {
                    string folder = directory + "/model-" + index++;
                    if (source is not OnnxModelReference reference || reference.Kind != OnnxModelReference.SourceKind.File) continue;
                    string sourcePath = reference.ResolveModelPath() ?? throw new FileNotFoundException(reference.SourceName);
                    string projectPath = FileUtil.GetProjectRelativePath(sourcePath.Replace('\\', '/'));
                    string path = projectPath.StartsWith("Assets/", StringComparison.Ordinal) ? projectPath : folder + "/" + Path.GetFileName(sourcePath);
                    folder = Path.GetDirectoryName(path).Replace('\\', '/');
                    MoveGraphFiles(sourcePath, path);
                    reference.ConfigureFile(Path.GetFullPath(path), reference.Sha256, reference.Inputs.ToArray(), reference.Outputs.ToArray());
                    EditorUtility.SetDirty(set);
                    AssetDatabase.SaveAssetIfDirty(set);
                    AssetDatabase.Refresh(ImportAssetOptions.ForceSynchronousImport);
                    var asset = AssetDatabase.LoadAssetAtPath<OnnxModelAsset>(path);
                    if (asset == null || !asset.HasResolvableData)
                    {
                        AssetDatabase.SetImporterOverride<OnnxModelImporter>(path);
                        AssetDatabase.ImportAsset(path, ImportAssetOptions.ForceSynchronousImport);
                        asset = AssetDatabase.LoadAssetAtPath<OnnxModelAsset>(path);
                    }
                    if (asset == null || !asset.HasResolvableData) throw new InvalidDataException($"Could not import {path}.");
                    reference.ConfigureAsset(asset);
                }
                index = 0;
                foreach (var reference in set.GetAllTextFiles())
                {
                    string folder = directory + "/text-" + index++;
                    if (reference == null || reference.Kind != OnnxModelReference.SourceKind.File) continue;
                    string sourcePath = reference.ResolveFile();
                    string projectPath = FileUtil.GetProjectRelativePath(sourcePath.Replace('\\', '/'));
                    if (projectPath.StartsWith("Assets/", StringComparison.Ordinal) && AssetDatabase.LoadAssetAtPath<TextAsset>(projectPath) is TextAsset existingText)
                    { reference.ConfigureAsset(existingText); continue; }
                    string path = folder + "/" + Path.GetFileName(sourcePath);
                    // Unity imports arbitrary tokenizer extensions as TextAsset when suffixed with .bytes.
                    if (!path.EndsWith(".bytes", StringComparison.OrdinalIgnoreCase)) path += ".bytes";
                    ModelBuildFiles.Move(sourcePath, path);
                    reference.ConfigureFile(Path.GetFullPath(path));
                    EditorUtility.SetDirty(set);
                    AssetDatabase.SaveAssetIfDirty(set);
                    AssetDatabase.Refresh(ImportAssetOptions.ForceSynchronousImport);
                    var asset = AssetDatabase.LoadAssetAtPath<TextAsset>(path);
                    if (asset == null) throw new InvalidDataException($"Could not import {path}.");
                    reference.ConfigureAsset(asset);
                }
            }
            finally
            {
                // Each successful reference survives an interrupted or failed migration.
                EditorUtility.SetDirty(set);
                AssetDatabase.SaveAssetIfDirty(set);
            }
        }

        public static void Apply(ModelSet set, DownloadedModel installation,
            IReadOnlyDictionary<string, string> properties, Func<UnityEngine.Object, ScriptableObject> wrap)
        {
            string directory = MoveGraphs(set, installation);
            var serialized = new SerializedObject(set);
            foreach (var item in properties)
            {
                var property = serialized.FindProperty(item.Value);
                if (!System.Linq.Enumerable.Any(installation.Files, file => file.Role == item.Key)) { property.objectReferenceValue = null; continue; }
                string path = directory + "/" + installation.GetFile(item.Key).Path;
                AssetDatabase.ClearImporterOverride(path);
                AssetDatabase.ImportAsset(path, ImportAssetOptions.ForceSynchronousImport | ImportAssetOptions.ForceUpdate);
                ScriptableObject source = wrap(AssetDatabase.LoadMainAssetAtPath(path));
                string sourcePath = directory + "/" + item.Key + ".asset";
                var existing = AssetDatabase.LoadMainAssetAtPath(sourcePath);
                if (existing != null) { EditorUtility.CopySerialized(source, existing); UnityEngine.Object.DestroyImmediate(source); }
                else { AssetDatabase.CreateAsset(source, sourcePath); existing = source; }
                property.objectReferenceValue = existing;
            }
            serialized.ApplyModifiedProperties();
            AssetDatabase.SaveAssets();
        }

        public static DownloadedModel MigratedInstallation(ModelSet set)
        {
            string directory = ModelBuildFiles.DirectoryFor(set);
            return new ModelInstallationStore(Path.GetDirectoryName(directory), Path.GetFileName(directory)).Read();
        }

        private static string MoveGraphs(ModelSet set, DownloadedModel installation)
        {
            string directory = ModelBuildFiles.DirectoryFor(set);
            if (string.IsNullOrEmpty(directory)) throw new InvalidOperationException("Save the model set first.");
            var moved = new List<(string Source, string Destination)>();
            try
            {
                foreach (string file in installation.Files.Select(file => file.Path).Append("installation.json"))
                {
                    string source = Path.Combine(installation.DirectoryPath, file);
                    string destination = directory + "/" + file;
                    ModelBuildFiles.Move(source, destination);
                    moved.Add((source, destination));
                }
            }
            catch
            {
                foreach (var item in moved.AsEnumerable().Reverse()) ModelBuildFiles.Move(item.Destination, item.Source);
                throw;
            }
            foreach (var reference in set.GetAllTextFiles())
            {
                if (reference == null || reference.Kind != OnnxModelReference.SourceKind.File) continue;
                string source = reference.ResolveFile();
                var file = installation.Files.FirstOrDefault(file => Path.GetFullPath(Path.Combine(installation.DirectoryPath, file.Path)) == source);
                if (file != null) reference.ConfigureFile(Path.GetFullPath(directory + "/" + file.Path));
            }
            EditorUtility.SetDirty(set);
            AssetDatabase.SaveAssetIfDirty(set);
            AssetDatabase.Refresh(ImportAssetOptions.ForceSynchronousImport);
            return directory;
        }
    }
}

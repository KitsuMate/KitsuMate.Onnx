using System;
using System.Collections.Generic;
using System.IO;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Security.Cryptography;
using System.Threading;
using System.Threading.Tasks;
using System.Text;
using UnityEditor;
using UnityEngine;

namespace KitsuMate.Onnx.Editor.Download
{
    [Serializable]
    public sealed class ModelFileManifest
    {
        public string Key;
        public string Url;
        public string RelativePath;
        public string Sha256;
        public long SizeBytes;
        public bool Optional;
        public bool ExternalModel;
        public string ModelSetProperty;
        public OnnxModelAsset.TensorInfo[] Inputs = Array.Empty<OnnxModelAsset.TensorInfo>();
        public OnnxModelAsset.TensorInfo[] Outputs = Array.Empty<OnnxModelAsset.TensorInfo>();
    }

    [Serializable]
    public sealed class ModelVariantManifest
    {
        public string Family;
        public string ModelId;
        public string Revision;
        public string Variant;
        public int ContractVersion = 1;
        public string[] Capabilities = Array.Empty<string>();
        public ModelProviderCompatibility[] ProviderCompatibility = Array.Empty<ModelProviderCompatibility>();
        public ModelFileManifest[] Files = Array.Empty<ModelFileManifest>();
    }

    public readonly struct ModelInstallProgress
    {
        public readonly string File; public readonly long Downloaded; public readonly long Total;
        public ModelInstallProgress(string file, long downloaded, long total) { File = file; Downloaded = downloaded; Total = total; }
    }

    public sealed class ModelInstallResult
    {
        public ModelIdentity Identity { get; internal set; }
        public IReadOnlyDictionary<string, string> ProjectPaths { get; internal set; }
    }

    [CreateAssetMenu(fileName = "OnnxModelCatalog", menuName = "KitsuMate/ONNX/Model Catalog")]
    public sealed class ModelCatalog : ScriptableObject
    {
        public ModelVariantManifest[] Variants = Array.Empty<ModelVariantManifest>();
    }

    public static class ModelManifestInstaller
    {
        private static readonly HttpClient Client = new HttpClient();

        public static async Task<ModelInstallResult> InstallAsync(ModelVariantManifest manifest, string projectRoot,
            IProgress<ModelInstallProgress> progress = null, CancellationToken cancellationToken = default)
        {
            if (manifest == null) throw new ArgumentNullException(nameof(manifest));
            if (string.IsNullOrWhiteSpace(projectRoot) || Path.IsPathRooted(projectRoot)) throw new ArgumentException("Model root must be project-relative.", nameof(projectRoot));
            var installed = new Dictionary<string, string>();
            foreach (ModelFileManifest file in manifest.Files ?? Array.Empty<ModelFileManifest>())
            {
                cancellationToken.ThrowIfCancellationRequested();
                if (file == null || string.IsNullOrWhiteSpace(file.Url) || string.IsNullOrWhiteSpace(file.RelativePath))
                { if (file?.Optional == true) continue; throw new InvalidOperationException("Manifest contains an invalid required file."); }
                string relative = Path.Combine(projectRoot, manifest.Family, manifest.ModelId, manifest.Revision, manifest.Variant, file.RelativePath).Replace('\\', '/');
                string destination = Path.GetFullPath(relative);
                string partial = destination + ".partial";
                Directory.CreateDirectory(Path.GetDirectoryName(destination));
                long existing = File.Exists(partial) ? new FileInfo(partial).Length : 0;
                using var request = new HttpRequestMessage(HttpMethod.Get, file.Url);
                if (existing > 0) request.Headers.Range = new RangeHeaderValue(existing, null);
                using HttpResponseMessage response = await Client.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, cancellationToken);
                if (existing > 0 && response.StatusCode != System.Net.HttpStatusCode.PartialContent) { File.Delete(partial); existing = 0; }
                response.EnsureSuccessStatusCode();
                using (Stream source = await response.Content.ReadAsStreamAsync())
                using (var target = new FileStream(partial, existing > 0 ? FileMode.Append : FileMode.Create, FileAccess.Write, FileShare.None, 1024 * 1024, true))
                {
                    var buffer = new byte[1024 * 1024]; long downloaded = existing; int read;
                    while ((read = await source.ReadAsync(buffer, 0, buffer.Length, cancellationToken)) > 0)
                    { await target.WriteAsync(buffer, 0, read, cancellationToken); downloaded += read; progress?.Report(new ModelInstallProgress(file.RelativePath, downloaded, file.SizeBytes)); }
                }
                if (!string.IsNullOrWhiteSpace(file.Sha256) && !string.Equals(await ComputeSha256Async(partial, cancellationToken), file.Sha256, StringComparison.OrdinalIgnoreCase))
                {
                    File.Delete(partial);
                    throw new InvalidDataException($"SHA-256 verification failed for {file.RelativePath}.");
                }
                if (File.Exists(destination)) File.Replace(partial, destination, null); else File.Move(partial, destination);
                installed[file.Key] = relative;
            }
            AssetDatabase.Refresh();
            return new ModelInstallResult { Identity = new ModelIdentity(manifest.Family, manifest.ModelId, manifest.Revision, manifest.Variant, ComputeManifestHash(manifest), manifest.ContractVersion), ProjectPaths = installed };
        }

        public static async Task<ModelInstallResult> InstallIntoAsync(ModelVariantManifest manifest, ModelSet target,
            string projectRoot, IProgress<ModelInstallProgress> progress = null, CancellationToken cancellationToken = default)
        {
            if (target == null) throw new ArgumentNullException(nameof(target));
            ModelInstallResult result = await InstallAsync(manifest, projectRoot, progress, cancellationToken);
            var serialized = new SerializedObject(target);
            SetIdentityAndCompatibility(serialized, manifest, result.Identity);
            foreach (ModelFileManifest file in manifest.Files ?? Array.Empty<ModelFileManifest>())
            {
                if (file == null || string.IsNullOrWhiteSpace(file.ModelSetProperty) || !result.ProjectPaths.TryGetValue(file.Key, out string projectPath)) continue;
                SerializedProperty property = serialized.FindProperty(file.ModelSetProperty);
                if (property == null) throw new InvalidOperationException($"Model set {target.GetType().Name} has no serialized property '{file.ModelSetProperty}'.");
                if (file.ExternalModel)
                {
                    SerializedProperty kind = property.FindPropertyRelative("sourceKind");
                    SerializedProperty relativePath = property.FindPropertyRelative("relativePath");
                    SerializedProperty hash = property.FindPropertyRelative("sha256");
                    if (kind == null || relativePath == null)
                        throw new InvalidOperationException($"Property '{file.ModelSetProperty}' is not an OnnxModelReference.");
                    string root = projectRoot.TrimEnd('/', '\\') + "/";
                    if (!projectPath.StartsWith(root, StringComparison.OrdinalIgnoreCase))
                        throw new InvalidOperationException($"Installed model '{projectPath}' is outside model root '{projectRoot}'.");
                    kind.enumValueIndex = (int)OnnxModelReference.SourceKind.File;
                    relativePath.stringValue = projectPath.Substring(root.Length);
                    if (hash != null) hash.stringValue = file.Sha256 ?? string.Empty;
                    var info = new FileInfo(Path.GetFullPath(projectPath));
                    property.FindPropertyRelative("cachedFileSize").longValue = info.Length;
                    property.FindPropertyRelative("cachedWriteTimeUtcTicks").longValue = info.LastWriteTimeUtc.Ticks;
                    bool hasSchema = (file.Outputs?.Length ?? 0) > 0;
                    property.FindPropertyRelative("metadataInspected").boolValue = hasSchema;
                    SetTensorInfos(property.FindPropertyRelative("inputs"), file.Inputs);
                    SetTensorInfos(property.FindPropertyRelative("outputs"), file.Outputs);
                }
                else
                {
                    SerializedProperty kind = property.FindPropertyRelative("sourceKind");
                    SerializedProperty asset = property.FindPropertyRelative("asset");
                    if (kind == null || asset == null)
                        throw new InvalidOperationException($"Property '{file.ModelSetProperty}' is not an OnnxModelReference.");
                    kind.enumValueIndex = (int)OnnxModelReference.SourceKind.Asset;
                    asset.objectReferenceValue = AssetDatabase.LoadAssetAtPath<OnnxModelAsset>(projectPath);
                }
            }
            serialized.ApplyModifiedPropertiesWithoutUndo();
            EditorUtility.SetDirty(target);
            AssetDatabase.SaveAssets();
            return result;
        }

        private static async Task<string> ComputeSha256Async(string path, CancellationToken cancellationToken)
        {
            using var sha = SHA256.Create();
            using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 1024 * 1024, true);
            byte[] hash = await Task.Run(() => { cancellationToken.ThrowIfCancellationRequested(); return sha.ComputeHash(stream); }, cancellationToken);
            return BitConverter.ToString(hash).Replace("-", "").ToLowerInvariant();
        }

        private static string ComputeManifestHash(ModelVariantManifest manifest)
        {
            using var sha = SHA256.Create();
            byte[] digest = sha.ComputeHash(Encoding.UTF8.GetBytes(JsonUtility.ToJson(manifest)));
            return BitConverter.ToString(digest).Replace("-", string.Empty).ToLowerInvariant();
        }

        private static void SetTensorInfos(SerializedProperty target, OnnxModelAsset.TensorInfo[] values)
        {
            values ??= Array.Empty<OnnxModelAsset.TensorInfo>();
            target.arraySize = values.Length;
            for (int i = 0; i < values.Length; i++)
            {
                SerializedProperty item = target.GetArrayElementAtIndex(i);
                item.FindPropertyRelative("name").stringValue = values[i].Name ?? string.Empty;
                item.FindPropertyRelative("elementType").stringValue = values[i].ElementType ?? string.Empty;
                item.FindPropertyRelative("shapeDescription").stringValue = values[i].ShapeDescription ?? string.Empty;
                SerializedProperty shape = item.FindPropertyRelative("shape");
                int[] dimensions = values[i].Shape ?? Array.Empty<int>();
                shape.arraySize = dimensions.Length;
                for (int dimension = 0; dimension < dimensions.Length; dimension++)
                    shape.GetArrayElementAtIndex(dimension).intValue = dimensions[dimension];
            }
        }

        private static void SetIdentityAndCompatibility(SerializedObject serialized, ModelVariantManifest manifest, ModelIdentity identity)
        {
            SetString(serialized, "family", identity.Family);
            SetString(serialized, "modelId", identity.ModelId);
            SetString(serialized, "revision", identity.Revision);
            SetString(serialized, "variant", identity.Variant);
            SetString(serialized, "contentHash", identity.ContentHash);
            SerializedProperty contract = serialized.FindProperty("contractVersion");
            if (contract != null) contract.intValue = identity.ContractVersion;

            SerializedProperty capabilityValues = serialized.FindProperty("capabilities")?.FindPropertyRelative("values");
            if (capabilityValues != null)
            {
                string[] values = manifest.Capabilities ?? Array.Empty<string>(); capabilityValues.arraySize = values.Length;
                for (int i = 0; i < values.Length; i++) capabilityValues.GetArrayElementAtIndex(i).stringValue = values[i] ?? string.Empty;
            }

            SerializedProperty providers = serialized.FindProperty("providerCompatibility");
            if (providers == null) return;
            ModelProviderCompatibility[] compatibility = manifest.ProviderCompatibility ?? Array.Empty<ModelProviderCompatibility>();
            providers.arraySize = compatibility.Length;
            for (int i = 0; i < compatibility.Length; i++)
            {
                SerializedProperty item = providers.GetArrayElementAtIndex(i);
                item.FindPropertyRelative("BackendId").stringValue = compatibility[i].BackendId ?? string.Empty;
                item.FindPropertyRelative("Supported").boolValue = compatibility[i].Supported;
                item.FindPropertyRelative("RecommendedRamMb").intValue = compatibility[i].RecommendedRamMb;
                item.FindPropertyRelative("RecommendedVramMb").intValue = compatibility[i].RecommendedVramMb;
                SerializedProperty operators = item.FindPropertyRelative("RequiredOperators");
                string[] required = compatibility[i].RequiredOperators ?? Array.Empty<string>(); operators.arraySize = required.Length;
                for (int op = 0; op < required.Length; op++) operators.GetArrayElementAtIndex(op).stringValue = required[op] ?? string.Empty;
            }
        }

        private static void SetString(SerializedObject serialized, string name, string value)
        {
            SerializedProperty property = serialized.FindProperty(name);
            if (property != null) property.stringValue = value ?? string.Empty;
        }
    }
}

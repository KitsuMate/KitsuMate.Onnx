using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using KitsuMate.Onnx;
using UnityEditor;
using UnityEditor.Build;
using UnityEditor.Build.Profile;
using UnityEditor.Build.Reporting;
using UnityEngine;

namespace KitsuMate.Onnx.Editor
{
    public static class OnnxRuntimePluginSetup
    {
        public static void ConfigureAndroidImporters()
        {
            ConfigureAndroid("Packages/ai.kitsumate.onnx.backend.onnxruntime/Runtime/Plugins/Android/arm64-v8a/libonnxruntime.so", "ARM64");
            ConfigureAndroid("Packages/ai.kitsumate.onnx.backend.onnxruntime/Runtime/Plugins/Android/armeabi-v7a/libonnxruntime.so", "ARMv7");
            AssetDatabase.SaveAssets();
        }

        private static void ConfigureAndroid(string path, string cpu)
        {
            AssetDatabase.ImportAsset(path, ImportAssetOptions.ForceSynchronousImport | ImportAssetOptions.ForceUpdate);
            if (AssetImporter.GetAtPath(path) is not PluginImporter importer)
                throw new InvalidOperationException($"Native Android plug-in was not imported: {path}");
            importer.SetCompatibleWithAnyPlatform(false);
            importer.SetCompatibleWithEditor(false);
            importer.SetCompatibleWithPlatform(BuildTarget.Android, true);
            importer.SetPlatformData(BuildTarget.Android, "CPU", cpu);
            importer.SaveAndReimport();
        }
    }

    [InitializeOnLoad]
    internal static class OnnxRuntimeProviderPluginFilter
    {
        internal const string DirectMlDefine = "KITSUMATE_ORT_DIRECTML";
        internal const string CudaDefine = "KITSUMATE_ORT_CUDA";
        internal const string TensorRtDefine = "KITSUMATE_ORT_TENSORRT";
        internal const string OpenVinoDefine = "KITSUMATE_ORT_OPENVINO";
        internal const string NnapiDefine = "KITSUMATE_ORT_NNAPI";

        static OnnxRuntimeProviderPluginFilter()
        {
            foreach (string guid in AssetDatabase.FindAssets("t:PluginImporter"))
            {
                string path = AssetDatabase.GUIDToAssetPath(guid).Replace('\\', '/');
                if (!path.Contains("ai.kitsumate.onnx.backend.onnxruntime/Runtime/Plugins/")) continue;
                if (AssetImporter.GetAtPath(path) is PluginImporter importer)
                    importer.SetIncludeInBuildDelegate(ShouldInclude);
            }
        }

        private static bool ShouldInclude(string path)
        {
            HashSet<string> defines = GetActiveDefines();
            string file = Path.GetFileName(path).ToLowerInvariant();
            if (file.Contains("tensorrt")) return defines.Contains(TensorRtDefine);
            if (file.Contains("cuda")) return defines.Contains(CudaDefine);
            if (file.Contains("openvino")) return defines.Contains(OpenVinoDefine);
            if (file.Contains("directml") || file == "directml.dll") return defines.Contains(DirectMlDefine);
            if (file.Contains("providers_shared"))
                return defines.Contains(CudaDefine) || defines.Contains(TensorRtDefine) || defines.Contains(OpenVinoDefine);
            return true;
        }

        internal static HashSet<string> GetActiveDefines()
        {
            BuildProfile profile = BuildProfile.GetActiveBuildProfile();
            return profile == null
                ? new HashSet<string>(StringComparer.Ordinal)
                : new HashSet<string>(profile.scriptingDefines ?? Array.Empty<string>(), StringComparer.Ordinal);
        }
    }

    internal sealed class OnnxRuntimeProviderBuildValidator : IPreprocessBuildWithReport, IPostprocessBuildWithReport
    {
        public int callbackOrder => -1000;

        public void OnPreprocessBuild(BuildReport report)
        {
            HashSet<string> defines = OnnxRuntimeProviderPluginFilter.GetActiveDefines();
            BuildTarget target = report.summary.platform;
            bool windows = target == BuildTarget.StandaloneWindows64;
            bool linux = target == BuildTarget.StandaloneLinux64;
            bool android = target == BuildTarget.Android;

            Require(!defines.Contains(OnnxRuntimeProviderPluginFilter.DirectMlDefine) || windows,
                "DirectML is Windows x64 only.");
            Require(!defines.Contains(OnnxRuntimeProviderPluginFilter.OpenVinoDefine) || linux,
                "OpenVINO is Linux x64 only.");
            Require(!defines.Contains(OnnxRuntimeProviderPluginFilter.TensorRtDefine) || defines.Contains(OnnxRuntimeProviderPluginFilter.CudaDefine),
                "TensorRT requires KITSUMATE_ORT_CUDA.");
            Require(!defines.Contains(OnnxRuntimeProviderPluginFilter.NnapiDefine) || android,
                "NNAPI is Android-only.");
            if (android)
                Require(!defines.Overlaps(new[]
                {
                    OnnxRuntimeProviderPluginFilter.DirectMlDefine,
                    OnnxRuntimeProviderPluginFilter.CudaDefine,
                    OnnxRuntimeProviderPluginFilter.TensorRtDefine,
                    OnnxRuntimeProviderPluginFilter.OpenVinoDefine
                }), "Android Build Profiles cannot contain desktop ONNX Runtime provider defines.");

            string platformFolder = windows ? "Windows/x86_64" : linux ? "Linux/x86_64" : android ? "Android" : null;
            if (platformFolder != null) ValidateNativeClosure(platformFolder, defines, windows, linux, android);
            ValidateBackendAssets(defines, windows, linux, android);
        }

        public void OnPostprocessBuild(BuildReport report)
        {
            string output = report.summary.outputPath;
            string directory = Directory.Exists(output) ? output : Path.GetDirectoryName(output);
            if (string.IsNullOrWhiteSpace(directory)) return;
            var manifest = new ProviderBuildManifest
            {
                runtimeVersion = "1.24.4",
                profileDefines = OnnxRuntimeProviderPluginFilter.GetActiveDefines().OrderBy(value => value).ToArray(),
                coreSha256 = FindBuiltCoreHash(output),
                includedPayloads = FindBuiltPayloads(output).ToArray(),
                expectedExternalDependencies = ExpectedDependencies(OnnxRuntimeProviderPluginFilter.GetActiveDefines()).ToArray()
            };
            File.WriteAllText(Path.Combine(directory, "kitsumate-onnx-providers.json"), JsonUtility.ToJson(manifest, true));
        }

        private static void ValidateNativeClosure(
            string platformFolder, HashSet<string> defines, bool windows, bool linux, bool android)
        {
            string root = "Packages/ai.kitsumate.onnx.backend.onnxruntime/Runtime/Plugins/" + platformFolder;
            string coreName = windows ? "onnxruntime.dll" : "libonnxruntime.so";
            string[] cores = AssetDatabase.FindAssets(Path.GetFileNameWithoutExtension(coreName))
                .Select(AssetDatabase.GUIDToAssetPath)
                .Where(path => path.Replace('\\', '/').StartsWith(root + "/", StringComparison.Ordinal) && Path.GetFileName(path) == coreName)
                .ToArray();
            if (android)
                Require(cores.Length is 1 or 2, $"Android requires one ONNX Runtime core per enabled ABI; found {cores.Length}.");
            else Require(cores.Length == 1, $"Build requires exactly one matching ONNX Runtime core in {root}; found {cores.Length}.");

            if (defines.Contains(OnnxRuntimeProviderPluginFilter.CudaDefine)) RequirePlugin(root, "cuda");
            if (defines.Contains(OnnxRuntimeProviderPluginFilter.TensorRtDefine)) RequirePlugin(root, "tensorrt");
            if (defines.Contains(OnnxRuntimeProviderPluginFilter.OpenVinoDefine)) RequirePlugin(root, "openvino");
            if (defines.Contains(OnnxRuntimeProviderPluginFilter.DirectMlDefine)) RequirePlugin(root, "directml");
            if (defines.Overlaps(new[]
                {
                    OnnxRuntimeProviderPluginFilter.CudaDefine,
                    OnnxRuntimeProviderPluginFilter.TensorRtDefine,
                    OnnxRuntimeProviderPluginFilter.OpenVinoDefine
                }))
                RequirePlugin(root, "providers_shared");
        }

        private static void ValidateBackendAssets(HashSet<string> defines, bool windows, bool linux, bool android)
        {
            var packaged = new HashSet<OnnxExecutionProvider> { OnnxExecutionProvider.Cpu };
            if (windows && defines.Contains(OnnxRuntimeProviderPluginFilter.DirectMlDefine)) packaged.Add(OnnxExecutionProvider.DirectMl);
            if ((windows || linux) && defines.Contains(OnnxRuntimeProviderPluginFilter.CudaDefine)) packaged.Add(OnnxExecutionProvider.Cuda);
            if ((windows || linux) && defines.Contains(OnnxRuntimeProviderPluginFilter.TensorRtDefine)) packaged.Add(OnnxExecutionProvider.TensorRt);
            if (linux && defines.Contains(OnnxRuntimeProviderPluginFilter.OpenVinoDefine)) packaged.Add(OnnxExecutionProvider.OpenVino);
            if (android && defines.Contains(OnnxRuntimeProviderPluginFilter.NnapiDefine))
                packaged.Add(OnnxExecutionProvider.Nnapi);

            RuntimePlatform runtimePlatform = windows
                ? RuntimePlatform.WindowsPlayer
                : linux
                    ? RuntimePlatform.LinuxPlayer
                    : RuntimePlatform.Android;

            foreach (string guid in AssetDatabase.FindAssets("t:OnnxRuntimeBackend"))
            {
                string path = AssetDatabase.GUIDToAssetPath(guid);
                OnnxRuntimeBackend backend = AssetDatabase.LoadAssetAtPath<OnnxRuntimeBackend>(path);
                if (backend == null) continue;
                try
                {
                    OnnxRuntimeBackend.ResolveProviderOrder(backend.ProviderOrder, packaged.ToArray(), runtimePlatform);
                }
                catch (Exception exception)
                {
                    throw new BuildFailedException(
                        $"Backend '{path}' has no eligible provider for {runtimePlatform} with active profile providers " +
                        $"[{string.Join(", ", packaged)}]. {exception.Message}");
                }
            }
        }

        private static void RequirePlugin(string root, string token)
        {
            bool exists = AssetDatabase.FindAssets(token)
                .Select(AssetDatabase.GUIDToAssetPath)
                .Any(path => path.Replace('\\', '/').StartsWith(root + "/", StringComparison.OrdinalIgnoreCase) &&
                             Path.GetFileName(path).Contains(token, StringComparison.OrdinalIgnoreCase));
            Require(exists, $"Provider payload containing '{token}' is missing from {root}.");
        }

        private static IEnumerable<string> ExpectedDependencies(HashSet<string> defines)
        {
            if (defines.Contains(OnnxRuntimeProviderPluginFilter.DirectMlDefine)) yield return "DirectX 12-capable Windows GPU driver";
            if (defines.Contains(OnnxRuntimeProviderPluginFilter.CudaDefine)) yield return "CUDA runtime compatible with the packaged core";
            if (defines.Contains(OnnxRuntimeProviderPluginFilter.TensorRtDefine)) yield return "TensorRT and cuDNN compatible with the packaged core";
            if (defines.Contains(OnnxRuntimeProviderPluginFilter.OpenVinoDefine)) yield return "OpenVINO runtime compatible with the packaged core";
        }

        private static string FindBuiltCoreHash(string outputPath)
        {
            string root = Directory.Exists(outputPath) ? outputPath : Path.GetDirectoryName(outputPath);
            if (string.IsNullOrWhiteSpace(root) || !Directory.Exists(root)) return string.Empty;
            string core = Directory.EnumerateFiles(root, "*onnxruntime*", SearchOption.AllDirectories)
                .FirstOrDefault(path => Path.GetFileName(path) is "onnxruntime.dll" or "libonnxruntime.so");
            if (core == null) return string.Empty;
            using var stream = File.OpenRead(core);
            using var hash = SHA256.Create();
            return string.Concat(hash.ComputeHash(stream).Select(value => value.ToString("x2")));
        }

        private static IEnumerable<string> FindBuiltPayloads(string outputPath)
        {
            string root = Directory.Exists(outputPath) ? outputPath : Path.GetDirectoryName(outputPath);
            if (string.IsNullOrWhiteSpace(root) || !Directory.Exists(root)) return Array.Empty<string>();
            return Directory.EnumerateFiles(root, "*", SearchOption.AllDirectories)
                .Where(path => Path.GetFileName(path).Contains("onnxruntime", StringComparison.OrdinalIgnoreCase) ||
                               Path.GetFileName(path).Contains("directml", StringComparison.OrdinalIgnoreCase) ||
                               Path.GetFileName(path).Contains("openvino", StringComparison.OrdinalIgnoreCase))
                .Select(path => Path.GetRelativePath(root, path).Replace('\\', '/'))
                .OrderBy(path => path)
                .ToArray();
        }

        private static void Require(bool condition, string message)
        {
            if (!condition) throw new BuildFailedException(message);
        }

        [Serializable]
        private sealed class ProviderBuildManifest
        {
            public string runtimeVersion;
            public string coreSha256;
            public string[] includedPayloads;
            public string[] profileDefines;
            public string[] expectedExternalDependencies;
        }
    }
}

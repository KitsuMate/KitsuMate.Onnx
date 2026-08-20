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
    [InitializeOnLoad]
    internal static class OnnxRuntimeProviderSelectionMigration
    {
        private const string ProviderSerializedField = "_providerSelectionMode:";
        private const string DeviceSerializedField = "_deviceSelectionMode:";

        static OnnxRuntimeProviderSelectionMigration()
        {
            EditorApplication.delayCall += MigrateLegacyAssets;
        }

        internal static void MigrateLegacyAssets()
        {
            bool changed = false;
            foreach (string guid in AssetDatabase.FindAssets("t:OnnxRuntimeBackend"))
            {
                string path = AssetDatabase.GUIDToAssetPath(guid);
                if (string.IsNullOrWhiteSpace(path) || !File.Exists(path)) continue;
                string yaml;
                try { yaml = File.ReadAllText(path); }
                catch (IOException) { continue; }
                OnnxRuntimeBackend backend = AssetDatabase.LoadAssetAtPath<OnnxRuntimeBackend>(path);
                if (backend == null) continue;
                bool assetChanged = false;
                if (!yaml.Contains(ProviderSerializedField, StringComparison.Ordinal))
                {
                    backend.ProviderSelectionMode = OnnxProviderSelectionMode.Explicit;
                    assetChanged = true;
                }
                if (!yaml.Contains(DeviceSerializedField, StringComparison.Ordinal))
                {
                    backend.DeviceSelectionMode = OnnxDeviceSelectionMode.Explicit;
                    assetChanged = true;
                }
                if (!assetChanged) continue;
                EditorUtility.SetDirty(backend);
                changed = true;
                Debug.Log($"Migrated legacy ONNX Runtime backend '{path}' to Explicit provider/device selection.");
            }
            if (changed) AssetDatabase.SaveAssets();
        }
    }

    [InitializeOnLoad]
    internal static class OnnxRuntimeEditorBootstrap
    {
        static OnnxRuntimeEditorBootstrap()
        {
            OnnxRuntimeGraphicsDeviceHint.CaptureFromUnity();
            RegisterResolvedPackageRoots();
            // Extension assemblies (notably NVIDIA) may register later in the same
            // Editor initialization pass, so repeat after all InitializeOnLoad work.
            EditorApplication.delayCall += RegisterResolvedPackageRoots;
        }

        private static void RegisterResolvedPackageRoots()
        {
            foreach (IOnnxRuntimeProviderModule module in OnnxRuntimeProviderRegistry.GetModules())
            {
                UnityEditor.PackageManager.PackageInfo package =
                    UnityEditor.PackageManager.PackageInfo.FindForAssembly(module.GetType().Assembly);
                if (package == null) continue;
                OnnxRuntimeProviderRegistry.RegisterNativeSearchRoot(
                    module.GetType().Assembly,
                    Path.Combine(package.resolvedPath, "Runtime", "Plugins"));
            }
        }
    }

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
        internal const string NvidiaDefine = "KITSUMATE_ORT_NVIDIA";
        internal const string CudaDefine = "KITSUMATE_ORT_CUDA";
        internal const string TensorRtDefine = "KITSUMATE_ORT_TENSORRT";
        internal const string OpenVinoDefine = "KITSUMATE_ORT_OPENVINO";

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
            if (file.Contains("tensorrt") || file.Contains("cuda")) return IsNvidiaEnabled(defines);
            if (file.Contains("openvino")) return false;
            return true;
        }

        internal static bool IsNvidiaEnabled(IReadOnlyCollection<string> defines) =>
            defines.Contains(NvidiaDefine) || defines.Contains(CudaDefine);

        internal static HashSet<string> GetActiveDefines()
        {
            BuildProfile profile = BuildProfile.GetActiveBuildProfile();
            return profile == null
                ? new HashSet<string>(StringComparer.Ordinal)
                : new HashSet<string>(profile.scriptingDefines ?? Array.Empty<string>(), StringComparer.Ordinal);
        }

        internal static string GetDefine(OnnxExecutionProvider provider)
        {
            return provider switch
            {
                OnnxExecutionProvider.Cuda => NvidiaDefine,
                OnnxExecutionProvider.TensorRtRtx => NvidiaDefine,
                OnnxExecutionProvider.OpenVino => OpenVinoDefine,
                _ => null
            };
        }

        internal static IReadOnlyList<OnnxExecutionProvider> GetPackagedProviders(
            BuildTarget target,
            IReadOnlyCollection<string> defines)
        {
            defines ??= Array.Empty<string>();
            bool windows = target == BuildTarget.StandaloneWindows64;
            bool linux = target == BuildTarget.StandaloneLinux64;
            bool android = target == BuildTarget.Android;
            bool macos = target == BuildTarget.StandaloneOSX;
            var providers = new List<OnnxExecutionProvider>();
            if (windows) providers.Add(OnnxExecutionProvider.DirectMl);
            if (linux) providers.Add(OnnxExecutionProvider.WebGpu);
            if (macos) providers.Add(OnnxExecutionProvider.CoreMl);
            if (android) providers.Add(OnnxExecutionProvider.Nnapi);
            providers.Add(OnnxExecutionProvider.Cpu);
            if ((windows || linux) && IsNvidiaEnabled(defines))
            {
                providers.Insert(0, OnnxExecutionProvider.Cuda);
                if (defines.Contains(NvidiaDefine) || defines.Contains(TensorRtDefine))
                    providers.Insert(0, OnnxExecutionProvider.TensorRtRtx);
            }
            return providers.AsReadOnly();
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
            bool macos = target == BuildTarget.StandaloneOSX;

            Require(!defines.Contains(OnnxRuntimeProviderPluginFilter.OpenVinoDefine),
                "KITSUMATE_ORT_OPENVINO is not supported. Remove it; OpenVINO is deferred for this release.");
            Require(!defines.Contains(OnnxRuntimeProviderPluginFilter.TensorRtDefine) || defines.Contains(OnnxRuntimeProviderPluginFilter.CudaDefine),
                "KITSUMATE_ORT_TENSORRT alone is unsupported. Install ai.kitsumate.onnx.backend.onnxruntime.nvidia and use KITSUMATE_ORT_NVIDIA.");
            bool nvidia = OnnxRuntimeProviderPluginFilter.IsNvidiaEnabled(defines);
            Require(!nvidia || windows || linux, "NVIDIA providers support Windows x64 and Linux x64 only.");
            Require(!nvidia || AssetDatabase.IsValidFolder("Packages/ai.kitsumate.onnx.backend.onnxruntime.nvidia"),
                "The active Build Profile enables NVIDIA providers, but package ai.kitsumate.onnx.backend.onnxruntime.nvidia is not installed.");
            if (defines.Contains(OnnxRuntimeProviderPluginFilter.CudaDefine))
                Debug.LogWarning(defines.Contains(OnnxRuntimeProviderPluginFilter.TensorRtDefine)
                    ? "KITSUMATE_ORT_CUDA + KITSUMATE_ORT_TENSORRT is deprecated; use KITSUMATE_ORT_NVIDIA."
                    : "KITSUMATE_ORT_CUDA is deprecated and enables CUDA only; use KITSUMATE_ORT_NVIDIA.");
            if (android)
                Require(!defines.Overlaps(new[]
                {
                    OnnxRuntimeProviderPluginFilter.CudaDefine,
                    OnnxRuntimeProviderPluginFilter.TensorRtDefine,
                    OnnxRuntimeProviderPluginFilter.OpenVinoDefine
                }), "Android Build Profiles cannot contain desktop ONNX Runtime provider defines.");

            string platformFolder = windows ? "Windows/x86_64" : linux ? "Linux/x86_64" : android ? "Android" : null;
            if (platformFolder != null) ValidateNativeClosure(platformFolder, defines, windows, linux, android);
            ValidateBackendAssets(defines, windows, linux, android, macos);
        }

        public void OnPostprocessBuild(BuildReport report)
        {
            string output = report.summary.outputPath;
            string directory = Directory.Exists(output) ? output : Path.GetDirectoryName(output);
            if (string.IsNullOrWhiteSpace(directory)) return;
            var manifest = new ProviderBuildManifest
            {
                runtimeVersion = "1.25.1",
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

            if (windows) RequirePlugin(root, "directml");
            if (linux) RequirePlugin(root, "webgpu");
        }

        private static void ValidateBackendAssets(HashSet<string> defines, bool windows, bool linux, bool android, bool macos)
        {
            IReadOnlyList<OnnxExecutionProvider> packaged =
                OnnxRuntimeProviderPluginFilter.GetPackagedProviders(
                    windows ? BuildTarget.StandaloneWindows64 : linux ? BuildTarget.StandaloneLinux64 : macos ? BuildTarget.StandaloneOSX : BuildTarget.Android,
                    defines);

            RuntimePlatform runtimePlatform = windows
                ? RuntimePlatform.WindowsPlayer
                : linux
                    ? RuntimePlatform.LinuxPlayer
                    : macos ? RuntimePlatform.OSXPlayer : RuntimePlatform.Android;

            foreach (string guid in AssetDatabase.FindAssets("t:OnnxRuntimeBackend"))
            {
                string path = AssetDatabase.GUIDToAssetPath(guid);
                OnnxRuntimeBackend backend = AssetDatabase.LoadAssetAtPath<OnnxRuntimeBackend>(path);
                if (backend == null) continue;
                try
                {
                    OnnxRuntimeBackend.ResolveProviderOrder(backend.ProviderOrder, packaged, runtimePlatform);
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

        private static void RequireFiles(string root, IEnumerable<string> fileNames)
        {
            var existing = new HashSet<string>(
                AssetDatabase.FindAssets(string.Empty, new[] { root })
                    .Select(AssetDatabase.GUIDToAssetPath)
                    .Select(Path.GetFileName),
                StringComparer.OrdinalIgnoreCase);
            string[] missing = fileNames.Where(file => !existing.Contains(file)).ToArray();
            Require(missing.Length == 0,
                $"Bundled Windows CUDA runtime is incomplete in {root}; missing: {string.Join(", ", missing)}.");
        }

        private static IEnumerable<string> ExpectedDependencies(HashSet<string> defines)
        {
            yield return "A platform GPU driver compatible with the default execution provider";
            if (OnnxRuntimeProviderPluginFilter.IsNvidiaEnabled(defines))
                yield return "Compatible NVIDIA display driver (CUDA, cuDNN, and TensorRT-RTX user-mode runtimes are bundled)";
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
                .Where(path => IsProviderPayload(Path.GetFileName(path)))
                .Select(path => Path.GetRelativePath(root, path).Replace('\\', '/'))
                .OrderBy(path => path)
                .ToArray();
        }

        private static bool IsProviderPayload(string fileName)
        {
            string name = fileName.ToLowerInvariant();
            return name.Contains("onnxruntime") || name.Contains("directml") || name.Contains("cuda") ||
                   name.Contains("cudnn") || name.Contains("cublas") || name.Contains("cufft") ||
                   name.Contains("curand") || name.Contains("nvrtc") || name.Contains("nvjit") ||
                   name.Contains("tensorrt");
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

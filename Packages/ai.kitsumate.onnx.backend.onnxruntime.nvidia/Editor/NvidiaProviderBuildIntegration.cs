using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEditor.Build;
using UnityEditor.Build.Profile;
using UnityEditor.Build.Reporting;

namespace KitsuMate.Onnx.Nvidia.Editor
{
    [InitializeOnLoad]
    internal static class NvidiaProviderPluginFilter
    {
        private const string PackageRoot = "Packages/ai.kitsumate.onnx.backend.onnxruntime.nvidia/Runtime/Plugins/";
        private static readonly string[] TensorRtOnlyTokens =
        {
            "nv_tensorrt_rtx", "tensorrt_", "tensorrt.", "nvrtc", "nvjitlink"
        };

        static NvidiaProviderPluginFilter()
        {
            foreach (string guid in AssetDatabase.FindAssets("t:PluginImporter", new[] { PackageRoot }))
            {
                string path = AssetDatabase.GUIDToAssetPath(guid).Replace('\\', '/');
                if (AssetImporter.GetAtPath(path) is PluginImporter importer)
                {
                    string pluginPath = path;
                    importer.SetIncludeInBuildDelegate(_ => ShouldInclude(pluginPath));
                }
            }
        }

        internal static HashSet<string> GetDefines()
        {
            BuildProfile profile = BuildProfile.GetActiveBuildProfile();
            return new HashSet<string>(profile?.scriptingDefines ?? Array.Empty<string>(), StringComparer.Ordinal);
        }

        internal static bool IsEnabled() => IsEnabled(GetDefines());

        internal static bool IsEnabled(IReadOnlyCollection<string> defines) =>
            defines.Contains("KITSUMATE_ORT_NVIDIA") || defines.Contains("KITSUMATE_ORT_CUDA");

        internal static bool IncludesTensorRtRtx(IReadOnlyCollection<string> defines) =>
            defines.Contains("KITSUMATE_ORT_NVIDIA") ||
            defines.Contains("KITSUMATE_ORT_CUDA") && defines.Contains("KITSUMATE_ORT_TENSORRT");

        private static bool ShouldInclude(string path)
        {
            HashSet<string> defines = GetDefines();
            if (!IsEnabled(defines)) return false;
            if (IncludesTensorRtRtx(defines)) return true;
            string fileName = Path.GetFileName(path);
            return !TensorRtOnlyTokens.Any(token =>
                fileName.Contains(token, StringComparison.OrdinalIgnoreCase));
        }
    }

    internal sealed class NvidiaProviderBuildValidator : IPreprocessBuildWithReport
    {
        public int callbackOrder => -900;

        public void OnPreprocessBuild(BuildReport report)
        {
            HashSet<string> defines = NvidiaProviderPluginFilter.GetDefines();
            if (!NvidiaProviderPluginFilter.IsEnabled(defines)) return;
            string platform = report.summary.platform switch
            {
                BuildTarget.StandaloneWindows64 => "Windows/x86_64",
                BuildTarget.StandaloneLinux64 => "Linux/x86_64",
                _ => throw new BuildFailedException("The NVIDIA ONNX Runtime package supports Windows x64 and Linux x64 only.")
            };
            string root = $"Packages/ai.kitsumate.onnx.backend.onnxruntime.nvidia/Runtime/Plugins/{platform}";
            bool windows = report.summary.platform == BuildTarget.StandaloneWindows64;
            Require(root, windows ? WindowsCudaFiles : LinuxCudaFiles);
            if (NvidiaProviderPluginFilter.IncludesTensorRtRtx(defines))
                Require(root, windows ? WindowsTensorRtFiles : LinuxTensorRtFiles);

            bool alternateCore = AssetDatabase.FindAssets("onnxruntime", new[] { root })
                .Select(AssetDatabase.GUIDToAssetPath)
                .Select(Path.GetFileName)
                .Any(name => name is "onnxruntime.dll" or "libonnxruntime.so" or "libonnxruntime.dylib");
            if (alternateCore)
                throw new BuildFailedException("The NVIDIA provider package must not contain an ONNX Runtime core.");
        }

        private static readonly string[] WindowsCudaFiles =
        {
            "onnxruntime_providers_cuda.dll", "onnxruntime_providers_shared.dll", "cudart64_12.dll",
            "cublas64_12.dll", "cublasLt64_12.dll", "cufft64_11.dll", "curand64_10.dll",
            "cudnn64_9.dll", "cudnn_adv64_9.dll", "cudnn_cnn64_9.dll",
            "cudnn_engines_precompiled64_9.dll", "cudnn_engines_runtime_compiled64_9.dll",
            "cudnn_engines_tensor_ir64_9.dll", "cudnn_ext64_9.dll", "cudnn_graph64_9.dll",
            "cudnn_heuristic64_9.dll", "cudnn_ops64_9.dll"
        };

        private static readonly string[] LinuxCudaFiles =
        {
            "libonnxruntime_providers_cuda.so", "libonnxruntime_providers_shared.so", "libcudart.so.12",
            "libcublas.so.12", "libcublasLt.so.12", "libcufft.so.11", "libcurand.so.10",
            "libcudnn.so.9", "libcudnn_adv.so.9", "libcudnn_cnn.so.9",
            "libcudnn_engines_precompiled.so.9", "libcudnn_engines_runtime_compiled.so.9",
            "libcudnn_engines_tensor_ir.so.9", "libcudnn_ext.so.9", "libcudnn_graph.so.9",
            "libcudnn_heuristic.so.9", "libcudnn_ops.so.9"
        };

        private static readonly string[] WindowsTensorRtFiles =
        {
            "onnxruntime_providers_nv_tensorrt_rtx.dll", "nvrtc-builtins64_129.dll",
            "nvrtc64_120_0.dll", "nvJitLink_120_0.dll", "tensorrt_rtx_1_5.dll",
            "tensorrt_onnxparser_rtx_1_5.dll", "tensorrt_plugins.dll"
        };

        private static readonly string[] LinuxTensorRtFiles =
        {
            "libonnxruntime_providers_nv_tensorrt_rtx.so", "libnvrtc-builtins.so.12.9",
            "libnvrtc.so.12", "libnvJitLink.so.12", "libtensorrt_rtx.so.1.5.0",
            "libtensorrt_rtx.so.1", "libtensorrt_rtx.so", "libtensorrt_onnxparser_rtx.so.1.5.0",
            "libtensorrt_onnxparser_rtx.so.1", "libtensorrt_onnxparser_rtx.so",
            "libtensorrt_plugins.so"
        };

        private static void Require(string root, IEnumerable<string> names)
        {
            var actual = new HashSet<string>(AssetDatabase.FindAssets(string.Empty, new[] { root })
                .Select(AssetDatabase.GUIDToAssetPath).Select(Path.GetFileName), StringComparer.OrdinalIgnoreCase);
            string[] missing = names.Where(name => !actual.Contains(name)).ToArray();
            if (missing.Length != 0)
                throw new BuildFailedException($"NVIDIA provider payload is incomplete in {root}; missing: {string.Join(", ", missing)}.");
        }
    }
}

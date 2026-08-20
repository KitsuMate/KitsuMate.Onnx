using System.Collections.Generic;
using UnityEngine;

namespace KitsuMate.Onnx.Nvidia
{
    internal static class NvidiaProviderRegistration
    {
        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.SubsystemRegistration)]
        private static void Register()
        {
            OnnxRuntimeProviderRegistry.Register(new CudaProviderModule());
            OnnxRuntimeProviderRegistry.Register(new TensorRtRtxProviderModule());
        }

#if UNITY_EDITOR
        [UnityEditor.InitializeOnLoadMethod]
        private static void RegisterInEditor() => Register();
#endif
    }

    internal sealed class CudaProviderModule : OnnxRuntimePluginProviderModule
    {
        public override OnnxExecutionProvider Provider => OnnxExecutionProvider.Cuda;
        public override string RuntimeProviderName => "CUDAExecutionProvider";
        public override int AutomaticPriority => 400;
#if KITSUMATE_ORT_NVIDIA || KITSUMATE_ORT_CUDA
        public override bool IsEnabled => true;
#else
        public override bool IsEnabled => false;
#endif
        public override bool Supports(RuntimePlatform platform) =>
            platform is RuntimePlatform.WindowsEditor or RuntimePlatform.WindowsPlayer or
                RuntimePlatform.LinuxEditor or RuntimePlatform.LinuxPlayer;
        protected override string RegistrationName => "CUDAExecutionProvider";
        protected override IReadOnlyList<string> LibraryNames => new[]
        {
            "onnxruntime_providers_cuda.dll",
            "libonnxruntime_providers_cuda.so"
        };
        protected override IReadOnlyList<string> DependencyLibraryNames =>
            Application.platform is RuntimePlatform.LinuxEditor or RuntimePlatform.LinuxPlayer
                ? new[]
                {
                    "libcudart.so.12", "libcublasLt.so.12", "libcublas.so.12", "libcurand.so.10",
                    "libcufft.so.11", "libcudnn.so.9", "libcudnn_adv.so.9", "libcudnn_cnn.so.9",
                    "libcudnn_engines_precompiled.so.9", "libcudnn_engines_runtime_compiled.so.9",
                    "libcudnn_engines_tensor_ir.so.9", "libcudnn_ext.so.9", "libcudnn_graph.so.9",
                    "libcudnn_heuristic.so.9", "libcudnn_ops.so.9", "libonnxruntime_providers_shared.so"
                }
                : new[]
                {
                    "cudart64_12.dll", "cublasLt64_12.dll", "cublas64_12.dll", "curand64_10.dll",
                    "cufft64_11.dll", "cudnn64_9.dll", "cudnn_adv64_9.dll", "cudnn_cnn64_9.dll",
                    "cudnn_engines_precompiled64_9.dll", "cudnn_engines_runtime_compiled64_9.dll",
                    "cudnn_engines_tensor_ir64_9.dll", "cudnn_ext64_9.dll", "cudnn_graph64_9.dll",
                    "cudnn_heuristic64_9.dll", "cudnn_ops64_9.dll", "onnxruntime_providers_shared.dll"
                };
    }

    internal sealed class TensorRtRtxProviderModule : OnnxRuntimePluginProviderModule
    {
        public override OnnxExecutionProvider Provider => OnnxExecutionProvider.TensorRtRtx;
        public override string RuntimeProviderName => "NvTensorRTRTXExecutionProvider";
        public override int AutomaticPriority => 500;
#if KITSUMATE_ORT_NVIDIA || (KITSUMATE_ORT_CUDA && KITSUMATE_ORT_TENSORRT)
        public override bool IsEnabled => true;
#else
        public override bool IsEnabled => false;
#endif
        public override bool Supports(RuntimePlatform platform) =>
            platform is RuntimePlatform.WindowsEditor or RuntimePlatform.WindowsPlayer or
                RuntimePlatform.LinuxEditor or RuntimePlatform.LinuxPlayer;
        protected override string RegistrationName => "NvTensorRTRTXExecutionProvider";
        protected override IReadOnlyList<string> LibraryNames => new[]
        {
            "onnxruntime_providers_nv_tensorrt_rtx.dll",
            "libonnxruntime_providers_nv_tensorrt_rtx.so"
        };
        protected override IReadOnlyList<string> DependencyLibraryNames =>
            Application.platform is RuntimePlatform.LinuxEditor or RuntimePlatform.LinuxPlayer
                ? new[]
                {
                    "libcudart.so.12", "libnvrtc-builtins.so.12.9", "libnvrtc.so.12", "libnvJitLink.so.12",
                    "libtensorrt_rtx.so.1.5.0", "libtensorrt_onnxparser_rtx.so.1.5.0", "libtensorrt_plugins.so"
                }
                : new[]
                {
                    "cudart64_12.dll", "nvrtc-builtins64_129.dll", "nvrtc64_120_0.dll", "nvJitLink_120_0.dll",
                    "tensorrt_rtx_1_5.dll", "tensorrt_onnxparser_rtx_1_5.dll", "tensorrt_plugins.dll"
                };
    }
}

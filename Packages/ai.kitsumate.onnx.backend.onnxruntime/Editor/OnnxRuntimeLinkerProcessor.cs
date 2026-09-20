using System;
using System.IO;
using UnityEditor.Build;
using UnityEditor.Build.Reporting;
using UnityEditor.PackageManager;
using UnityEditor.UnityLinker;

namespace KitsuMate.Onnx.Editor
{
    internal sealed class OnnxRuntimeLinkerProcessor : IUnityLinkerProcessor
    {
        public int callbackOrder => 0;

        public string GenerateAdditionalLinkXmlFile(BuildReport report, UnityLinkerBuildPipelineData data)
        {
            var package = PackageInfo.FindForAssembly(GetType().Assembly)
                ?? throw new InvalidOperationException("Could not locate the ONNX Runtime backend package.");
            string path = Path.Combine(package.resolvedPath, "Runtime", "link.xml");
            if (!File.Exists(path)) throw new FileNotFoundException("ONNX Runtime linker rules are missing.", path);
            return path;
        }
    }
}

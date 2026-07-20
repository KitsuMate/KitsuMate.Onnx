#if UNITY_EDITOR
using System;
using System.IO;
using System.Linq;

namespace KitsuMate.Onnx.Editor
{
    public static class ExternalOnnxModelAssetUtility
    {
        public sealed class InspectionResult
        {
            public OnnxModelAsset.TensorInfo[] Inputs = Array.Empty<OnnxModelAsset.TensorInfo>();
            public OnnxModelAsset.TensorInfo[] Outputs = Array.Empty<OnnxModelAsset.TensorInfo>();
        }

        public static InspectionResult Inspect(string absolutePath)
        {
            if (!File.Exists(absolutePath)) throw new FileNotFoundException("External ONNX model was not found.", absolutePath);
            OnnxLightweightMetadataReader.Result metadata = OnnxLightweightMetadataReader.Read(absolutePath);
            return new InspectionResult { Inputs = metadata.Inputs.ToArray(), Outputs = metadata.Outputs.ToArray() };
        }

        public static void ConfigureAndInspect(OnnxModelReference reference, string modelRootRelativePath, string sha256)
        {
            if (reference == null) throw new ArgumentNullException(nameof(reference));
            OnnxSettings settings = OnnxSettings.Load();
            string root = settings != null ? settings.ModelStorageRoot : "Assets/StreamingAssets/KitsuMateModels";
            string absolutePath = Path.GetFullPath(Path.Combine(Directory.GetCurrentDirectory(), root, modelRootRelativePath));
            if (!File.Exists(absolutePath)) throw new FileNotFoundException("External ONNX model was not found.", absolutePath);
            InspectionResult result = Inspect(absolutePath);
            reference.ConfigureFile(modelRootRelativePath, sha256, result.Inputs, result.Outputs);
        }

    }
}
#endif

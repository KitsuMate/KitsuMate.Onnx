using System;
using System.Collections.Generic;
using System.Linq;
using Unity.InferenceEngine;
using UnityEngine;

namespace KitsuMate.Onnx
{
    public enum UnityAiInferenceDevice
    {
        Automatic,
        Cpu,
        GpuCompute,
        GpuPixel
    }

    /// <summary>
    /// Unity AI Inference-specific model source. It deliberately owns a Unity-imported ModelAsset,
    /// rather than accepting raw ONNX bytes as ONNX Runtime does.
    /// </summary>
    [CreateAssetMenu(fileName = "UnityAiInferenceModel", menuName = "KitsuMate/ONNX/Models/Unity AI Inference")]
    public sealed class UnityAiInferenceModelAsset : ScriptableObject, IOnnxModelSource
    {
        [SerializeField] private ModelAsset modelAsset;

        public ModelAsset ModelAsset => modelAsset;
        public string SourceName => modelAsset != null ? modelAsset.name : name;
        public bool IsAvailable => modelAsset != null;
        public bool HasInspectedMetadata => false;
        public bool IsMetadataStale => false;
        public IReadOnlyList<OnnxModelAsset.TensorInfo> Inputs => Array.Empty<OnnxModelAsset.TensorInfo>();
        public IReadOnlyList<OnnxModelAsset.TensorInfo> Outputs => Array.Empty<OnnxModelAsset.TensorInfo>();
        public OnnxModelAsset ImportedAsset => null;
        public string ResolveModelPath() => null;

#if UNITY_EDITOR
        public void SetModelAsset(ModelAsset value)
        {
            modelAsset = value;
            UnityEditor.EditorUtility.SetDirty(this);
        }
#endif
    }

    /// <summary>
    /// Backend for Unity's com.unity.ai.inference package. Model compatibility is intentionally
    /// explicit: unsupported operators or tensor element types fail at session creation or run time.
    /// </summary>
    [CreateAssetMenu(fileName = "UnityAiInferenceBackend", menuName = "KitsuMate/ONNX/Backends/Unity AI Inference")]
    public sealed class UnityAiInferenceBackend : OnnxBackend
    {
        [SerializeField] private UnityAiInferenceDevice device = UnityAiInferenceDevice.Automatic;

        private static readonly RuntimePlatform[] Supported =
        {
            RuntimePlatform.WindowsPlayer,
            RuntimePlatform.WindowsEditor,
            RuntimePlatform.OSXPlayer,
            RuntimePlatform.OSXEditor,
            RuntimePlatform.LinuxPlayer,
            RuntimePlatform.LinuxEditor,
            RuntimePlatform.Android,
            RuntimePlatform.IPhonePlayer,
            RuntimePlatform.WebGLPlayer
        };

        public override string BackendId => "unity-ai-inference";
        public override string DisplayName => "Unity AI Inference";
        public override IReadOnlyList<RuntimePlatform> SupportedPlatforms => Supported;
        public override bool IsAvailable => true;
        public override bool RequiresMainThread => true;
        public override int Priority => 50;
        public UnityAiInferenceDevice Device { get => device; set => device = value; }

        public override IOnnxSession CreateSession(IOnnxModelSource source)
        {
            if (source is not UnityAiInferenceModelAsset inferenceModel)
                throw new NotSupportedException("Unity AI Inference requires a UnityAiInferenceModelAsset. Raw ONNX byte and file sources are supported by the ONNX Runtime backend only.");
            if (inferenceModel.ModelAsset == null)
                throw new InvalidOperationException($"Unity AI Inference model '{inferenceModel.name}' has no assigned ModelAsset.");

            var model = ModelLoader.Load(inferenceModel.ModelAsset);
            if (model == null)
                throw new InvalidOperationException($"Unity AI Inference could not load model '{inferenceModel.name}'. Check its operator and quantization compatibility.");
            return new UnityAiInferenceSession(model, ResolveBackendType());
        }

        public override IOnnxSession CreateSession(byte[] modelData, OnnxSessionOptions options)
            => throw new NotSupportedException("Unity AI Inference sessions require a UnityAiInferenceModelAsset imported by Unity.");

        public override IOnnxSession CreateSession(string modelPath, OnnxSessionOptions options)
            => throw new NotSupportedException("Unity AI Inference sessions require a UnityAiInferenceModelAsset imported by Unity.");

        private BackendType ResolveBackendType()
        {
            return device switch
            {
                UnityAiInferenceDevice.Cpu => BackendType.CPU,
                UnityAiInferenceDevice.GpuCompute => BackendType.GPUCompute,
                UnityAiInferenceDevice.GpuPixel => BackendType.GPUPixel,
                _ => SystemInfo.supportsComputeShaders ? BackendType.GPUCompute : BackendType.CPU
            };
        }
    }

    internal sealed class UnityAiInferenceSession : IOnnxSession
    {
        private readonly Worker worker;
        private readonly string[] inputNames;
        private readonly string[] outputNames;

        public UnityAiInferenceSession(Model model, BackendType backendType)
        {
            inputNames = model.inputs.Select(input => input.name).ToArray();
            outputNames = model.outputs.Select(output => output.name).ToArray();
            worker = new Worker(model, backendType);
        }

        public IReadOnlyList<string> InputNames => inputNames;
        public IReadOnlyList<string> OutputNames => outputNames;
        public OnnxSessionDiagnostics Diagnostics { get; } = new(
            new[] { OnnxExecutionProvider.Cpu },
            new[] { OnnxExecutionProvider.Cpu },
            Array.Empty<OnnxProviderSkip>(),
            new[] { OnnxExecutionProvider.Cpu },
            Array.Empty<string>(),
            new[] { OnnxExecutionProvider.Cpu },
            OnnxExecutionProvider.Cpu,
            typeof(Model).Assembly.GetName().Version?.ToString(),
            0);
        public bool VerboseLogging { get; set; }

        public IReadOnlyDictionary<string, OnnxTensor> Run(IReadOnlyDictionary<string, OnnxTensor> inputs)
        {
            if (inputs == null) throw new ArgumentNullException(nameof(inputs));
            return Run(inputs.Select(pair => new OnnxNamedValue(pair.Key, pair.Value)).ToArray());
        }

        public IReadOnlyDictionary<string, OnnxTensor> Run(IReadOnlyList<OnnxNamedValue> inputs)
        {
            if (inputs == null) throw new ArgumentNullException(nameof(inputs));
            var convertedInputs = new List<Tensor>(inputs.Count);
            try
            {
                foreach (OnnxNamedValue input in inputs)
                {
                    if (input.Value == null)
                        throw new ArgumentException($"Input '{input.Name}' has no tensor.", nameof(inputs));
                    Tensor tensor = CreateTensor(input.Value);
                    convertedInputs.Add(tensor);
                    worker.SetInput(input.Name, tensor);
                }

                worker.Schedule();
                var results = new Dictionary<string, OnnxTensor>(outputNames.Length);
                foreach (string outputName in outputNames)
                    results[outputName] = ToOnnxTensor(outputName, worker.PeekOutput(outputName));
                return results;
            }
            finally
            {
                foreach (Tensor tensor in convertedInputs) tensor.Dispose();
            }
        }

        public async Awaitable<IReadOnlyDictionary<string, OnnxTensor>> RunAsync(IReadOnlyDictionary<string, OnnxTensor> inputs)
        {
            return Run(inputs);
        }

        public void Dispose() => worker?.Dispose();

        private static Tensor CreateTensor(OnnxTensor tensor)
        {
            var shape = new TensorShape(tensor.Shape);
            return tensor.ElementType switch
            {
                OnnxTensorElementType.Float => new Tensor<float>(shape, tensor.AsFloatArray()),
                OnnxTensorElementType.Int32 => new Tensor<int>(shape, tensor.AsIntArray()),
                OnnxTensorElementType.Int16 => new Tensor<short>(shape, (short[])tensor.Data),
                OnnxTensorElementType.UInt8 => new Tensor<byte>(shape, tensor.AsByteArray()),
                _ => throw new NotSupportedException($"Unity AI Inference does not support portable tensor type {tensor.ElementType}. Use an engine-specific model/implementation or the ONNX Runtime backend.")
            };
        }

        private static OnnxTensor ToOnnxTensor(string name, Tensor tensor)
        {
            var shape = new int[tensor.shape.rank];
            for (var index = 0; index < shape.Length; index++) shape[index] = tensor.shape[index];
            return tensor.dataType switch
            {
                DataType.Float => OnnxTensor.FromArray(((Tensor<float>)tensor).DownloadToArray(), shape, name),
                DataType.Int => OnnxTensor.FromArray(((Tensor<int>)tensor).DownloadToArray(), shape, name),
                DataType.Short => new OnnxTensor(shape, OnnxTensorElementType.Int16, ((Tensor<short>)tensor).DownloadToArray(), name),
                DataType.Byte => OnnxTensor.FromArray(((Tensor<byte>)tensor).DownloadToArray(), shape, name),
                _ => throw new NotSupportedException($"Unity AI Inference produced unsupported tensor type {tensor.dataType}.")
            };
        }
    }
}

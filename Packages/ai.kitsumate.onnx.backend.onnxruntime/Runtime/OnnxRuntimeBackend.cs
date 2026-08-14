using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.IO;
using System.Threading;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using UnityEngine;
using Debug = UnityEngine.Debug;

namespace KitsuMate.Onnx
{
    /// <summary>
    /// ONNX Runtime backend implementation.
    /// Uses Microsoft.ML.OnnxRuntime for CPU and GPU inference.
    /// 
    /// Supported platforms: Windows x64, Linux x64, and Android ARM.
    /// 
    /// GPU Support:
    /// - Windows: CUDA (NVIDIA), TensorRT (NVIDIA)
    /// - Linux: CUDA (NVIDIA), TensorRT (NVIDIA)
    /// - Android: NNAPI with CPU fallback
    /// </summary>
    [CreateAssetMenu(fileName = "OnnxRuntimeBackend", menuName = "KitsuMate/ONNX/Backends/ONNX Runtime")]
    public class OnnxRuntimeBackend : OnnxBackend
    {
        public override string BackendId => "onnxruntime";
        [SerializeField, Tooltip("Providers are attempted in order. Include CPU explicitly to allow initialization fallback.")]
        private List<OnnxExecutionProvider> _providerOrder = new()
        {
            OnnxExecutionProvider.TensorRt,
            OnnxExecutionProvider.Cuda,
            OnnxExecutionProvider.DirectMl,
            OnnxExecutionProvider.OpenVino,
            OnnxExecutionProvider.Nnapi,
            OnnxExecutionProvider.Cpu,
        };
        
        [SerializeField, Tooltip("GPU device ID (for multi-GPU systems).")]
        private int _gpuDeviceId;
        
        private static readonly RuntimePlatform[] _supportedPlatforms =
        {
            RuntimePlatform.WindowsPlayer,
            RuntimePlatform.WindowsEditor,
            RuntimePlatform.LinuxPlayer,
            RuntimePlatform.LinuxEditor,
            RuntimePlatform.Android,
        };
        
        /// <summary>Display name for Editor UI.</summary>
        public override string DisplayName => $"ONNX Runtime ({string.Join(" > ", EffectiveProviderOrder)})";

        /// <summary>
        /// Ordered initialization policy. CPU must be present to permit initialization fallback.
        /// ONNX Runtime can still assign unsupported graph nodes to its CPU implementation.
        /// </summary>
        public IReadOnlyList<OnnxExecutionProvider> ProviderOrder => EffectiveProviderOrder;

        public int DeviceId
        {
            get => _gpuDeviceId;
            set
            {
                if (value < 0) throw new ArgumentOutOfRangeException(nameof(value));
                _gpuDeviceId = value;
            }
        }

        public void SetProviderOrder(params OnnxExecutionProvider[] providers)
        {
            if (providers == null) throw new ArgumentNullException(nameof(providers));
            ValidateProviderOrder(providers);
            _providerOrder = new List<OnnxExecutionProvider>(providers);
        }

        private IReadOnlyList<OnnxExecutionProvider> EffectiveProviderOrder =>
            _providerOrder != null
                ? _providerOrder
                : Array.Empty<OnnxExecutionProvider>();
        
        /// <summary>Platforms this backend supports.</summary>
        public override IReadOnlyList<RuntimePlatform> SupportedPlatforms => _supportedPlatforms;
        
        /// <summary>Whether this platform is supported and the native ONNX Runtime can be loaded.</summary>
        public override bool IsAvailable
        {
            get
            {
                if (!_supportedPlatforms.Contains(Application.platform)) return false;
                try
                {
                    _ = OrtEnv.Instance();
                    return true;
                }
                catch (Exception)
                {
                    return false;
                }
            }
        }
        
        /// <summary>Priority - prefer ONNX Runtime over fallbacks.</summary>
        public override int Priority => 100;
        
        /// <summary>Create an inference session from model bytes.</summary>
        public override IOnnxSession CreateSession(byte[] modelData, OnnxSessionOptions options)
        {
            if (modelData == null || modelData.Length == 0)
                throw new ArgumentException("Model data is required.", nameof(modelData));
            options ??= OnnxSessionOptions.Default;
            return CreateSessionCore(options, ortOptions => new InferenceSession(modelData, ortOptions),
                $"model={modelData.Length / (1024f * 1024f):F1}MB");
        }

        /// <summary>Create a session without copying a large model into managed memory.</summary>
        public override IOnnxSession CreateSession(string modelPath, OnnxSessionOptions options)
        {
            if (string.IsNullOrWhiteSpace(modelPath))
                throw new ArgumentException("Model path is required.", nameof(modelPath));
            if (!Path.IsPathRooted(modelPath))
                throw new ArgumentException("Model path must be absolute.", nameof(modelPath));
            if (!File.Exists(modelPath))
                throw new FileNotFoundException("ONNX model was not found.", modelPath);

            options ??= OnnxSessionOptions.Default;
            return CreateSessionCore(options, ortOptions => new InferenceSession(modelPath, ortOptions),
                $"path={Path.GetFileName(modelPath)}");
        }

        private IOnnxSession CreateSessionCore(
            OnnxSessionOptions options,
            Func<SessionOptions, InferenceSession> create,
            string modelDescription)
        {
            IReadOnlyList<OnnxExecutionProvider> requested = EffectiveProviderOrder.ToArray();
            IReadOnlyList<OnnxExecutionProvider> packaged = GetPackagedProviders();
            ValidateProviderOrder(requested);
            ResolveProviderOrder(requested, packaged, Application.platform,
                out IReadOnlyList<OnnxExecutionProvider> eligible,
                out IReadOnlyList<OnnxProviderSkip> skipped);

            var failures = new List<string>();
            var attempted = new List<OnnxExecutionProvider>();
            for (int index = 0; index < eligible.Count; index++)
            {
                OnnxExecutionProvider provider = eligible[index];
                attempted.Add(provider);
                var sw = Stopwatch.StartNew();
                try
                {
                    using SessionOptions ortOptions = CreateOrtSessionOptions(options, provider);
                    InferenceSession session = create(ortOptions);
                    var diagnostics = new OnnxSessionDiagnostics(
                        requested, eligible, skipped, attempted, failures, packaged, provider,
                        typeof(InferenceSession).Assembly.GetName().Version?.ToString(), _gpuDeviceId);
                    Debug.Log($"[OnnxRuntimeBackend] Session created in {sw.ElapsedMilliseconds}ms " +
                              $"({modelDescription}, provider={provider})");
                    return new OnnxRuntimeSession(session, diagnostics, GetDeviceProviderName(provider), _gpuDeviceId);
                }
                catch (Exception exception) when (index + 1 < eligible.Count)
                {
                    string failure = $"{provider}: {exception.GetBaseException().Message}";
                    failures.Add(failure);
                    Debug.LogWarning($"[OnnxRuntimeBackend] Provider initialization failed; trying {eligible[index + 1]}. {failure}");
                }
                catch (Exception exception)
                {
                    throw new OnnxProviderUnavailableException(provider,
                        $"Required ONNX execution provider '{provider}' failed to initialize and no fallback remains.", exception);
                }
            }

            throw new OnnxProviderUnavailableException(requested[0], "No eligible ONNX execution provider could be initialized.");
        }

        private SessionOptions CreateOrtSessionOptions(OnnxSessionOptions options, OnnxExecutionProvider provider)
        {
            var ortOptions = new SessionOptions
            {
                GraphOptimizationLevel = ToOrtOptimizationLevel(options.OptimizationLevel),
                EnableMemoryPattern = provider == OnnxExecutionProvider.DirectMl ? false : options.EnableMemoryPattern,
                EnableCpuMemArena = options.EnableCpuMemArena,
                LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR
            };

            if (provider == OnnxExecutionProvider.DirectMl)
                ortOptions.ExecutionMode = ExecutionMode.ORT_SEQUENTIAL;
            
            if (options.IntraOpThreads > 0)
                ortOptions.IntraOpNumThreads = options.IntraOpThreads;
            
            if (options.InterOpThreads > 0)
                ortOptions.InterOpNumThreads = options.InterOpThreads;
            
            AppendProvider(ortOptions, provider);
            return ortOptions;
        }

        private static GraphOptimizationLevel ToOrtOptimizationLevel(OnnxOptimizationLevel level)
        {
            return level switch
            {
                OnnxOptimizationLevel.DisableAll => GraphOptimizationLevel.ORT_DISABLE_ALL,
                OnnxOptimizationLevel.Basic => GraphOptimizationLevel.ORT_ENABLE_BASIC,
                OnnxOptimizationLevel.Extended => GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
                _ => GraphOptimizationLevel.ORT_ENABLE_ALL
            };
        }
        
        private void AppendProvider(SessionOptions options, OnnxExecutionProvider provider)
        {
            switch (provider)
            {
                case OnnxExecutionProvider.Cpu:
                    options.AppendExecutionProvider_CPU(0);
                    return;
                case OnnxExecutionProvider.Cuda:
                    options.AppendExecutionProvider_CUDA(_gpuDeviceId);
                    return;
                case OnnxExecutionProvider.TensorRt:
                    options.AppendExecutionProvider_Tensorrt(_gpuDeviceId);
                    return;
                case OnnxExecutionProvider.Nnapi:
#if UNITY_ANDROID && !UNITY_EDITOR
                    EnsureNnapiApiLevel();
                    options.AppendExecutionProvider_Nnapi();
                    return;
#else
                    throw new PlatformNotSupportedException("NNAPI is only available in Android players.");
#endif
                case OnnxExecutionProvider.DirectMl:
                    InvokeProviderMethod(options, "AppendExecutionProvider_DML", _gpuDeviceId);
                    return;
                case OnnxExecutionProvider.OpenVino:
                    InvokeProviderMethod(options, "AppendExecutionProvider_OpenVINO", "CPU_FP32");
                    return;
                default:
                    throw new ArgumentOutOfRangeException(nameof(provider), provider, null);
            }
        }

        private static void InvokeProviderMethod(SessionOptions options, string methodName, object argument)
        {
            System.Reflection.MethodInfo method = typeof(SessionOptions).GetMethods()
                .FirstOrDefault(candidate => candidate.Name == methodName && candidate.GetParameters().Length == 1);
            if (method == null)
                throw new MissingMethodException(typeof(SessionOptions).FullName, methodName);
            method.Invoke(options, new[] { argument });
        }

        private static void ValidateProviderOrder(IReadOnlyList<OnnxExecutionProvider> providers)
        {
            if (providers == null || providers.Count == 0)
                throw new ArgumentException("Provider order must contain at least one provider.", nameof(providers));
            var unique = new HashSet<OnnxExecutionProvider>();
            foreach (OnnxExecutionProvider provider in providers)
                if (!unique.Add(provider))
                    throw new ArgumentException($"Provider order contains duplicate provider '{provider}'.", nameof(providers));
        }

        public static bool IsProviderValidForPlatform(OnnxExecutionProvider provider, RuntimePlatform platform)
        {
            return platform switch
            {
                RuntimePlatform.WindowsEditor or RuntimePlatform.WindowsPlayer =>
                    provider is OnnxExecutionProvider.Cpu or OnnxExecutionProvider.DirectMl or OnnxExecutionProvider.Cuda or OnnxExecutionProvider.TensorRt,
                RuntimePlatform.LinuxEditor or RuntimePlatform.LinuxPlayer =>
                    provider is OnnxExecutionProvider.Cpu or OnnxExecutionProvider.Cuda or OnnxExecutionProvider.TensorRt or OnnxExecutionProvider.OpenVino,
                RuntimePlatform.Android => provider is OnnxExecutionProvider.Cpu or OnnxExecutionProvider.Nnapi,
                _ => false
            };
        }

        public static IReadOnlyList<OnnxExecutionProvider> ResolveProviderOrder(
            IReadOnlyList<OnnxExecutionProvider> configured,
            IReadOnlyList<OnnxExecutionProvider> packaged,
            RuntimePlatform platform)
        {
            ResolveProviderOrder(configured, packaged, platform, out IReadOnlyList<OnnxExecutionProvider> eligible, out _);
            return eligible;
        }

        private static void ResolveProviderOrder(
            IReadOnlyList<OnnxExecutionProvider> configured,
            IReadOnlyList<OnnxExecutionProvider> packaged,
            RuntimePlatform platform,
            out IReadOnlyList<OnnxExecutionProvider> eligible,
            out IReadOnlyList<OnnxProviderSkip> skipped)
        {
            ValidateProviderOrder(configured);
            if (packaged == null) throw new ArgumentNullException(nameof(packaged));

            var resolved = new List<OnnxExecutionProvider>();
            var omitted = new List<OnnxProviderSkip>();
            foreach (OnnxExecutionProvider provider in configured)
            {
                if (!IsProviderValidForPlatform(provider, platform))
                {
                    omitted.Add(new OnnxProviderSkip(provider, $"Not supported on {platform}."));
                    continue;
                }
                if (!packaged.Contains(provider))
                {
                    omitted.Add(new OnnxProviderSkip(provider, "Not included by the active Unity Build Profile."));
                    continue;
                }
                resolved.Add(provider);
            }

            if (resolved.Count == 0)
                throw new OnnxProviderUnavailableException(configured[0],
                    $"None of the configured ONNX providers [{string.Join(", ", configured)}] are eligible on {platform} " +
                    $"with packaged providers [{string.Join(", ", packaged)}].");

            eligible = resolved.AsReadOnly();
            skipped = omitted.AsReadOnly();
        }

        public static IReadOnlyList<OnnxExecutionProvider> GetPackagedProviders()
        {
            var providers = new List<OnnxExecutionProvider> { OnnxExecutionProvider.Cpu };
#if KITSUMATE_ORT_DIRECTML
            providers.Add(OnnxExecutionProvider.DirectMl);
#endif
#if KITSUMATE_ORT_CUDA
            providers.Add(OnnxExecutionProvider.Cuda);
#endif
#if KITSUMATE_ORT_TENSORRT
            providers.Add(OnnxExecutionProvider.TensorRt);
#endif
#if KITSUMATE_ORT_OPENVINO
            providers.Add(OnnxExecutionProvider.OpenVino);
#endif
#if UNITY_ANDROID && !UNITY_EDITOR && KITSUMATE_ORT_NNAPI
            providers.Add(OnnxExecutionProvider.Nnapi);
#endif
            return providers.AsReadOnly();
        }

        private static string GetDeviceProviderName(OnnxExecutionProvider provider)
        {
            return provider switch
            {
                OnnxExecutionProvider.Cuda => "Cuda",
                OnnxExecutionProvider.TensorRt => "Tensorrt",
                OnnxExecutionProvider.DirectMl => "DML",
                _ => null
            };
        }

#if UNITY_ANDROID && !UNITY_EDITOR
        private static void EnsureNnapiApiLevel()
        {
            using var version = new AndroidJavaClass("android.os.Build$VERSION");
            int sdkVersion = version.GetStatic<int>("SDK_INT");
            if (sdkVersion < 27)
                throw new PlatformNotSupportedException($"NNAPI requires Android API 27; device API is {sdkVersion}.");
        }
#endif

    }
    
    /// <summary>
    /// ONNX Runtime session wrapper implementing IOnnxSession and IOnnxDeviceSession.
    /// Supports IO Binding for keeping tensors on the inference device between calls.
    /// </summary>
    internal class OnnxRuntimeSession : IOnnxDeviceSession
    {
        private readonly InferenceSession _session;
        private readonly List<string> _inputNames;
        private readonly List<string> _outputNames;
        private readonly string _gpuProviderName; // e.g. "Cuda", "DML", "Tensorrt", or null for CPU-only
        private readonly int _gpuDeviceId;
        private OrtMemoryInfo _gpuMemInfo; // lazily created for IO Binding
        private int _disposed; // 0 = alive, 1 = disposed (atomic via Interlocked)
        
        // Tracks all OrtDeviceTensors created by this session so orphans can be
        // disposed when the session is torn down (e.g. domain reload mid-inference).
        private readonly HashSet<OrtDeviceTensor> _trackedDeviceTensors = new();
        
        /// <summary>
        /// When true, logs timing for input conversion, inference, and output conversion.
        /// Set by the engine's VerboseLogging flag.
        /// </summary>
        public bool VerboseLogging { get; set; }
        
        public IReadOnlyList<string> InputNames => _inputNames;
        public IReadOnlyList<string> OutputNames => _outputNames;
        public OnnxSessionDiagnostics Diagnostics { get; }
        
        public OnnxRuntimeSession(
            InferenceSession session,
            OnnxSessionDiagnostics diagnostics,
            string gpuProviderName = null,
            int gpuDeviceId = 0)
        {
            _session = session ?? throw new ArgumentNullException(nameof(session));
            Diagnostics = diagnostics ?? throw new ArgumentNullException(nameof(diagnostics));
            _gpuProviderName = gpuProviderName;
            _gpuDeviceId = gpuDeviceId;
            _inputNames = session.InputMetadata.Keys.ToList();
            _outputNames = session.OutputMetadata.Keys.ToList();
        }
        
        public IReadOnlyDictionary<string, OnnxTensor> Run(IReadOnlyDictionary<string, OnnxTensor> inputs)
        {
            ThrowIfDisposed();
            
            // Avoid LINQ allocation — populate list directly
            var namedInputs = new List<OnnxNamedValue>(inputs.Count);
            foreach (var kv in inputs)
                namedInputs.Add(new OnnxNamedValue(kv.Key, kv.Value));
            return Run(namedInputs);
        }
        
        public IReadOnlyDictionary<string, OnnxTensor> Run(IReadOnlyList<OnnxNamedValue> inputs)
        {
            ThrowIfDisposed();
            var sw = VerboseLogging ? Stopwatch.StartNew() : null;
            
            // Convert to ONNX Runtime format, auto-casting float↔float16 when needed
            var ortInputs = new List<NamedOnnxValue>(inputs.Count);
            foreach (var input in inputs)
            {
                var tensor = input.Value;
                if (_session.InputMetadata.TryGetValue(input.Name, out var meta))
                    tensor = CastTensorIfNeeded(tensor, meta.ElementDataType);
                ortInputs.Add(ConvertToOrtValue(input.Name, tensor));
            }
            
            long inputConvertMs = 0;
            if (sw != null)
            {
                inputConvertMs = sw.ElapsedMilliseconds;
                sw.Restart();
            }
            
            // Run inference
            using var results = _session.Run(ortInputs);
            
            long inferenceMs = 0;
            if (sw != null)
            {
                inferenceMs = sw.ElapsedMilliseconds;
                sw.Restart();
            }
            
            // Convert results back
            var outputs = new Dictionary<string, OnnxTensor>();
            foreach (var result in results)
            {
                outputs[result.Name] = ConvertFromOrtValue(result);
            }
            
            if (sw != null)
            {
                long outputConvertMs = sw.ElapsedMilliseconds;
                Debug.Log($"[OnnxSession] Run: inputConvert={inputConvertMs}ms, inference={inferenceMs}ms, outputConvert={outputConvertMs}ms ({inputs.Count} inputs → {outputs.Count} outputs)");
            }
            
            return outputs;
        }
        
        public async Awaitable<IReadOnlyDictionary<string, OnnxTensor>> RunAsync(IReadOnlyDictionary<string, OnnxTensor> inputs)
        {
            // Run on background thread
            await Awaitable.BackgroundThreadAsync();
            var result = Run(inputs);
            await Awaitable.MainThreadAsync();
            return result;
        }
        
        /// <summary>
        /// Cast tensor data type to match model expectations (e.g. float32→float16 or int64→int32).
        /// Returns the original tensor if no conversion is needed.
        /// </summary>
        private static OnnxTensor CastTensorIfNeeded(OnnxTensor tensor, TensorElementType expectedType)
        {
            var srcType = tensor.ElementType;
            
            // Float32 → Float16
            if (srcType == OnnxTensorElementType.Float && expectedType == TensorElementType.Float16)
            {
                var src = tensor.AsFloatArray();
                var dst = new ushort[src.Length];
                for (int i = 0; i < src.Length; i++)
                    dst[i] = ((Float16)src[i]).value;
                return OnnxTensor.FromArray(dst, tensor.Shape, tensor.Name);
            }
            
            // Float16 → Float32
            if (srcType == OnnxTensorElementType.Float16 && expectedType == TensorElementType.Float)
            {
                var src = tensor.AsFloat16Array();
                var dst = new float[src.Length];
                for (int i = 0; i < src.Length; i++)
                    dst[i] = (float)new Float16(src[i]);
                return OnnxTensor.FromArray(dst, tensor.Shape, tensor.Name);
            }
            
            // Int64 → Int32
            if (srcType == OnnxTensorElementType.Int64 && expectedType == TensorElementType.Int32)
            {
                var src = tensor.AsLongArray();
                var dst = new int[src.Length];
                for (int i = 0; i < src.Length; i++)
                    dst[i] = checked((int)src[i]);
                return OnnxTensor.FromArray(dst, tensor.Shape, tensor.Name);
            }
            
            // Int32 → Int64
            if (srcType == OnnxTensorElementType.Int32 && expectedType == TensorElementType.Int64)
            {
                var src = tensor.AsIntArray();
                var dst = new long[src.Length];
                for (int i = 0; i < src.Length; i++)
                    dst[i] = src[i];
                return OnnxTensor.FromArray(dst, tensor.Shape, tensor.Name);
            }
            
            return tensor;
        }
        
        /// <summary>Convert float32 to IEEE 754 half-precision (ushort).</summary>
        /// <summary>Convert IEEE 754 half-precision (ushort) to float32.</summary>
        private static NamedOnnxValue ConvertToOrtValue(string name, OnnxTensor tensor)
        {
            var shape = tensor.Shape;
            
            return tensor.ElementType switch
            {
                OnnxTensorElementType.Float => NamedOnnxValue.CreateFromTensor(name, 
                    new DenseTensor<float>(tensor.AsFloatArray(), shape)),
                OnnxTensorElementType.Int32 => NamedOnnxValue.CreateFromTensor(name,
                    new DenseTensor<int>(tensor.AsIntArray(), shape)),
                OnnxTensorElementType.Int64 => NamedOnnxValue.CreateFromTensor(name,
                    new DenseTensor<long>(tensor.AsLongArray(), shape)),
                OnnxTensorElementType.UInt8 => NamedOnnxValue.CreateFromTensor(name,
                    new DenseTensor<byte>(tensor.AsByteArray(), shape)),
                OnnxTensorElementType.Bool => NamedOnnxValue.CreateFromTensor(name,
                    new DenseTensor<bool>(tensor.AsBoolArray(), shape)),
                OnnxTensorElementType.Float16 => ConvertFloat16ToOrtValue(name, tensor.AsFloat16Array(), shape),
                _ => throw new NotSupportedException($"Tensor element type {tensor.ElementType} not supported")
            };
        }
        
        /// <summary>
        /// Convert ushort[] (raw FP16 bits) to DenseTensor&lt;Float16&gt; for ONNX Runtime.
        /// DenseTensor&lt;ushort&gt; maps to UInt16, not Float16 — ORT requires the Float16 struct.
        /// </summary>
        private static NamedOnnxValue ConvertFloat16ToOrtValue(string name, ushort[] data, int[] shape)
        {
            var f16Data = new Float16[data.Length];
            for (int i = 0; i < data.Length; i++)
                f16Data[i] = new Float16(data[i]);
            return NamedOnnxValue.CreateFromTensor(name, new DenseTensor<Float16>(f16Data, shape));
        }
        
        private static OnnxTensor ConvertFromOrtValue(NamedOnnxValue value)
        {
            if (value.Value is Tensor<float> tensor)
            {
                var shape = tensor.Dimensions.ToArray();
                var data = tensor.ToArray();
                return OnnxTensor.FromArray(data, shape, value.Name);
            }
            
            if (value.Value is Tensor<int> intTensor)
            {
                var shape = intTensor.Dimensions.ToArray();
                var data = intTensor.ToArray();
                return OnnxTensor.FromArray(data, shape, value.Name);
            }
            
            if (value.Value is Tensor<long> longTensor)
            {
                var shape = longTensor.Dimensions.ToArray();
                var data = longTensor.ToArray();
                return OnnxTensor.FromArray(data, shape, value.Name);
            }

            if (value.Value is Tensor<byte> byteTensor)
            {
                return OnnxTensor.FromArray(byteTensor.ToArray(), byteTensor.Dimensions.ToArray(), value.Name);
            }

            if (value.Value is Tensor<bool> boolTensor)
            {
                var shape = boolTensor.Dimensions.ToArray();
                var data = boolTensor.ToArray();
                return OnnxTensor.FromArray(data, shape, value.Name);
            }
            
            // Float16 output (e.g. quantized models) — convert to float
            if (value.Value is Tensor<Float16> f16Tensor)
            {
                var shape = f16Tensor.Dimensions.ToArray();
                var f16Data = f16Tensor.ToArray();
                var floatData = new float[f16Data.Length];
                for (int i = 0; i < f16Data.Length; i++)
                    floatData[i] = (float)f16Data[i];
                return OnnxTensor.FromArray(floatData, shape, value.Name);
            }
            
            throw new NotSupportedException($"Cannot convert output '{value.Name}' - unsupported type");
        }
        
        // ── IO Binding / RunOnDevice ──────────────────────────────────────
        
        public IReadOnlyList<IDeviceTensor> RunOnDevice(
            IReadOnlyList<OnnxNamedValue> cpuInputs,
            IReadOnlyList<IDeviceTensor> deviceInputs,
            IReadOnlyCollection<string> cpuOutputNames)
        {
            ThrowIfDisposed();
            
            var sw = VerboseLogging ? Stopwatch.StartNew() : null;
            
            // Lazily create GPU memory info for IO Binding output placement
            if (_gpuMemInfo == null && _gpuProviderName != null)
            {
                _gpuMemInfo = new OrtMemoryInfo(
                    _gpuProviderName, OrtAllocatorType.DeviceAllocator, _gpuDeviceId, OrtMemType.Default);
            }
            
            // If no GPU available, fall back to regular Run wrapped in CpuDeviceTensors
            if (_gpuMemInfo == null)
                return RunOnDeviceFallback(cpuInputs, deviceInputs, cpuOutputNames);
            
            var cpuMemInfo = OrtMemoryInfo.DefaultInstance;
            var cpuOutputSet = cpuOutputNames != null
                ? new HashSet<string>(cpuOutputNames)
                : null; // null = all on CPU
            
            using var binding = _session.CreateIoBinding();
            var cpuOrtValues = new List<OrtValue>(cpuInputs.Count);
            
            try
            {
                // Bind CPU inputs (small tensors: embeddings, attention mask)
                foreach (var input in cpuInputs)
                {
                    var tensor = input.Value;
                    if (_session.InputMetadata.TryGetValue(input.Name, out var meta))
                        tensor = CastTensorIfNeeded(tensor, meta.ElementDataType);
                    var ortVal = CreateOrtValueFromTensor(tensor);
                    cpuOrtValues.Add(ortVal);
                    binding.BindInput(input.Name, ortVal);
                }
                
                // Bind device inputs (KV cache from previous step — already on GPU)
                foreach (var dt in deviceInputs)
                {
                    if (dt is not OrtDeviceTensor ortDt)
                        throw new ArgumentException($"Device input '{dt?.Name}' was created by an incompatible backend.", nameof(deviceInputs));
                    ortDt.ValidateFor(this, _session.InputMetadata.TryGetValue(dt.Name, out var metadata) ? metadata : null);
                    binding.BindInput(dt.Name, ortDt.Value);
                }
                
                // Bind outputs: CPU for those the caller needs to read, GPU for passthrough
                foreach (var outName in _outputNames)
                {
                    if (cpuOutputSet == null || cpuOutputSet.Contains(outName))
                        binding.BindOutputToDevice(outName, cpuMemInfo);
                    else
                        binding.BindOutputToDevice(outName, _gpuMemInfo);
                }
                
                long bindMs = 0;
                if (sw != null)
                {
                    bindMs = sw.ElapsedMilliseconds;
                    sw.Restart();
                }
                
                // Run inference with IO Binding
                using var runOptions = new RunOptions();
                _session.RunWithBinding(runOptions, binding);
                
                long inferenceMs = 0;
                if (sw != null)
                {
                    inferenceMs = sw.ElapsedMilliseconds;
                    sw.Restart();
                }
                
                // Extract output OrtValues
                var resultCollection = binding.GetOutputValues();
                
                // Wrap each output as IDeviceTensor and dispose partial ownership on failure.
                var outputs = new List<IDeviceTensor>(_outputNames.Count);
                try
                {
                    for (int i = 0; i < resultCollection.Count; i++)
                    {
                        var deviceTensor = new OrtDeviceTensor(resultCollection[i], _outputNames[i], this);
                        lock (_trackedDeviceTensors) _trackedDeviceTensors.Add(deviceTensor);
                        outputs.Add(deviceTensor);
                    }
                }
                catch
                {
                    foreach (IDeviceTensor output in outputs) output.Dispose();
                    for (int i = outputs.Count; i < resultCollection.Count; i++) resultCollection[i].Dispose();
                    throw;
                }
                
                if (sw != null)
                {
                    long extractMs = sw.ElapsedMilliseconds;
                    Debug.Log($"[OnnxSession] RunOnDevice: bind={bindMs}ms, inference={inferenceMs}ms, extract={extractMs}ms " +
                              $"({cpuInputs.Count} cpu + {deviceInputs.Count} device inputs → {outputs.Count} outputs)");
                }
                
                return outputs;
            }
            finally
            {
                // Dispose temporary CPU OrtValues (memory was pinned during binding/run)
                foreach (var ov in cpuOrtValues)
                    ov.Dispose();
            }
        }
        
        private IReadOnlyList<IDeviceTensor> RunOnDeviceFallback(
            IReadOnlyList<OnnxNamedValue> cpuInputs,
            IReadOnlyList<IDeviceTensor> deviceInputs,
            IReadOnlyCollection<string> cpuOutputNames)
        {
            var allInputs = new Dictionary<string, OnnxTensor>(cpuInputs.Count + deviceInputs.Count);
            foreach (var input in cpuInputs)
                allInputs[input.Name] = input.Value;
            foreach (var dt in deviceInputs)
                allInputs[dt.Name] = dt.ToCpu();
            
            var result = Run(allInputs);
            var outputs = new List<IDeviceTensor>(_outputNames.Count);
            foreach (string outputName in _outputNames)
            {
                if (!result.TryGetValue(outputName, out OnnxTensor tensor))
                    throw new OnnxModelContractException($"ONNX Runtime did not return declared output '{outputName}'.");
                outputs.Add(new CpuDeviceTensor(tensor) { Name = outputName });
            }
            return outputs;
        }
        
        /// <summary>
        /// Create an OrtValue from an OnnxTensor, pinning the managed array memory.
        /// The OrtValue must be disposed after use to unpin the memory.
        /// </summary>
        private static OrtValue CreateOrtValueFromTensor(OnnxTensor tensor)
        {
            var longShape = Array.ConvertAll(tensor.Shape, d => (long)d);
            
            return tensor.ElementType switch
            {
                OnnxTensorElementType.Float => OrtValue.CreateTensorValueFromMemory<float>(
                    tensor.AsFloatArray(), longShape),
                OnnxTensorElementType.Int32 => OrtValue.CreateTensorValueFromMemory<int>(
                    tensor.AsIntArray(), longShape),
                OnnxTensorElementType.Int64 => OrtValue.CreateTensorValueFromMemory<long>(
                    tensor.AsLongArray(), longShape),
                OnnxTensorElementType.UInt8 => OrtValue.CreateTensorValueFromMemory<byte>(
                    tensor.AsByteArray(), longShape),
                OnnxTensorElementType.Bool => OrtValue.CreateTensorValueFromMemory<bool>(
                    tensor.AsBoolArray(), longShape),
                OnnxTensorElementType.Float16 => CreateFloat16OrtValueNative(tensor.AsFloat16Array(), longShape),
                _ => throw new NotSupportedException($"Tensor type {tensor.ElementType} not supported for OrtValue creation")
            };
        }
        
        /// <summary>Convert ushort[] (raw FP16 bits) into an OrtValue with Float16 element type.</summary>
        private static OrtValue CreateFloat16OrtValueNative(ushort[] data, long[] shape)
        {
            var f16Data = new Float16[data.Length];
            for (int i = 0; i < data.Length; i++)
                f16Data[i] = new Float16(data[i]);
            return OrtValue.CreateTensorValueFromMemory<Float16>(f16Data, shape);
        }
        
        internal void UntrackDeviceTensor(OrtDeviceTensor tensor)
        {
            lock (_trackedDeviceTensors) _trackedDeviceTensors.Remove(tensor);
        }
        
        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
            
            // Dispose any orphaned device tensors still tracked (e.g. domain reload mid-inference)
            OrtDeviceTensor[] orphans;
            lock (_trackedDeviceTensors)
            {
                orphans = _trackedDeviceTensors.Count > 0
                    ? new OrtDeviceTensor[_trackedDeviceTensors.Count]
                    : null;
                if (orphans != null) _trackedDeviceTensors.CopyTo(orphans);
                _trackedDeviceTensors.Clear();
            }
            if (orphans != null)
            {
                Debug.LogWarning($"[OnnxSession] Disposing {orphans.Length} orphaned device tensor(s) during session teardown");
                foreach (var dt in orphans)
                    dt.Dispose();
            }
            
            _gpuMemInfo?.Dispose();
            _gpuMemInfo = null;
            _session?.Dispose();
        }
        
        private void ThrowIfDisposed()
        {
            if (_disposed != 0)
                throw new ObjectDisposedException(nameof(OnnxRuntimeSession));
        }
    }
    
    /// <summary>
    /// Device-resident tensor backed by an ORT native OrtValue.
    /// Can live on GPU (via IO Binding) or CPU. Manages the OrtValue lifecycle explicitly.
    /// </summary>
    internal class OrtDeviceTensor : IDeviceTensor
    {
        private OrtValue _value;
        private OnnxRuntimeSession _ownerSession;
        
        public string Name { get; set; }
        
        /// <summary>Internal access to the underlying OrtValue for IO Binding.</summary>
        internal OrtValue Value => _value;
        
        public OrtDeviceTensor(OrtValue value, string name, OnnxRuntimeSession owner = null)
        {
            _value = value ?? throw new ArgumentNullException(nameof(value));
            Name = name;
            _ownerSession = owner;
        }
        
        public OnnxTensor ToCpu()
        {
            if (_value == null)
                throw new ObjectDisposedException(nameof(OrtDeviceTensor));
            
            var typeAndShape = _value.GetTensorTypeAndShape();
            var shape = Array.ConvertAll(typeAndShape.Shape, l => (int)l);
            
            return typeAndShape.ElementDataType switch
            {
                TensorElementType.Float => OnnxTensor.FromArray(
                    _value.GetTensorDataAsSpan<float>().ToArray(), shape, Name),
                TensorElementType.Int32 => OnnxTensor.FromArray(
                    _value.GetTensorDataAsSpan<int>().ToArray(), shape, Name),
                TensorElementType.Int64 => OnnxTensor.FromArray(
                    _value.GetTensorDataAsSpan<long>().ToArray(), shape, Name),
                TensorElementType.UInt8 => OnnxTensor.FromArray(
                    _value.GetTensorDataAsSpan<byte>().ToArray(), shape, Name),
                TensorElementType.Bool => OnnxTensor.FromArray(
                    _value.GetTensorDataAsSpan<bool>().ToArray(), shape, Name),
                TensorElementType.Float16 => ConvertFloat16ToCpu(shape),
                _ => throw new NotSupportedException(
                    $"Cannot convert device tensor type {typeAndShape.ElementDataType} to CPU")
            };
        }
        
        private OnnxTensor ConvertFloat16ToCpu(int[] shape)
        {
            var f16Span = _value.GetTensorDataAsSpan<Float16>();
            var floats = new float[f16Span.Length];
            for (int i = 0; i < f16Span.Length; i++)
                floats[i] = (float)f16Span[i];
            return OnnxTensor.FromArray(floats, shape, Name);
        }

        internal void ValidateFor(OnnxRuntimeSession session, NodeMetadata metadata)
        {
            if (_value == null) throw new ObjectDisposedException(nameof(OrtDeviceTensor));
            if (!ReferenceEquals(_ownerSession, session))
                throw new ArgumentException($"Device tensor '{Name}' belongs to a different ONNX Runtime session.");
            if (metadata == null)
                throw new OnnxModelContractException($"Model does not declare device input '{Name}'.");
            var actual = _value.GetTensorTypeAndShape();
            if (actual.ElementDataType != metadata.ElementDataType)
                throw new OnnxModelContractException(
                    $"Device tensor '{Name}' has type {actual.ElementDataType}; model requires {metadata.ElementDataType}.");
            if (metadata.Dimensions.Length != actual.Shape.Length)
                throw new OnnxModelContractException($"Device tensor '{Name}' has rank {actual.Shape.Length}; model requires {metadata.Dimensions.Length}.");
            for (int i = 0; i < metadata.Dimensions.Length; i++)
                if (metadata.Dimensions[i] > 0 && metadata.Dimensions[i] != actual.Shape[i])
                    throw new OnnxModelContractException(
                        $"Device tensor '{Name}' dimension {i} is {actual.Shape[i]}; model requires {metadata.Dimensions[i]}.");
        }
        
        public void Dispose()
        {
            if (_value == null) return;
            _ownerSession?.UntrackDeviceTensor(this);
            _ownerSession = null;
            _value.Dispose();
            _value = null;
        }
    }
}

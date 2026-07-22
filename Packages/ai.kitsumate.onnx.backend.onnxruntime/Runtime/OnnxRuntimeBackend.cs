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
    /// GPU execution provider preference for Windows/Linux.
    /// </summary>
    public enum GpuProvider
    {
        /// <summary>Automatically detect best available provider.</summary>
        Auto,
        /// <summary>NVIDIA CUDA (requires NVIDIA GPU and CUDA toolkit).</summary>
        CUDA,
        /// <summary>DirectML - Works with any GPU on Windows (AMD, Intel, NVIDIA).</summary>
        DirectML,
        /// <summary>NVIDIA TensorRT (requires NVIDIA GPU and TensorRT).</summary>
        TensorRT,
        /// <summary>Force CPU only.</summary>
        CPU
    }
    
    /// <summary>
    /// ONNX Runtime backend implementation.
    /// Uses Microsoft.ML.OnnxRuntime for CPU and GPU inference.
    /// 
    /// Supported platforms: Windows, macOS, Linux, Android, iOS
    /// 
    /// GPU Support:
    /// - Windows: CUDA (NVIDIA), DirectML (AMD, Intel, NVIDIA), TensorRT (NVIDIA)
    /// - Linux: CUDA (NVIDIA), TensorRT (NVIDIA)
    /// - macOS: CoreML (Apple Silicon/Intel)
    /// - Android: NNAPI
    /// - iOS: CoreML
    /// </summary>
    [CreateAssetMenu(fileName = "OnnxRuntimeBackend", menuName = "KitsuMate/ONNX/Backends/ONNX Runtime")]
    public class OnnxRuntimeBackend : OnnxBackend
    {
        public override string BackendId => "onnxruntime";
        [SerializeField, Tooltip("Enable GPU acceleration if available.")]
        private bool _enableGpu = true;
        
        [SerializeField, Tooltip("Preferred GPU provider. Auto will try providers in order of performance.")]
        private GpuProvider _preferredProvider = GpuProvider.Auto;
        
        [SerializeField, Tooltip("GPU device ID (for multi-GPU systems).")]
        private int _gpuDeviceId;
        
        // Tracks the active GPU provider name for IO Binding (e.g., "Cuda", "DML")
        private string _activeGpuProviderName;
        
        private static readonly RuntimePlatform[] _supportedPlatforms =
        {
            RuntimePlatform.WindowsPlayer,
            RuntimePlatform.WindowsEditor,
            RuntimePlatform.OSXPlayer,
            RuntimePlatform.OSXEditor,
            RuntimePlatform.LinuxPlayer,
            RuntimePlatform.LinuxEditor,
            RuntimePlatform.Android,
            RuntimePlatform.IPhonePlayer,
        };
        
        /// <summary>Display name for Editor UI.</summary>
        public override string DisplayName => _enableGpu ? $"ONNX Runtime ({_preferredProvider})" : "ONNX Runtime (CPU)";
        
        /// <summary>Preferred GPU provider.</summary>
        public GpuProvider PreferredProvider
        {
            get => _preferredProvider;
            set => _preferredProvider = value;
        }
        
        /// <summary>Platforms this backend supports.</summary>
        public override IReadOnlyList<RuntimePlatform> SupportedPlatforms => _supportedPlatforms;
        
        /// <summary>Whether ONNX Runtime is available.</summary>
        public override bool IsAvailable => true; // DLLs are embedded
        
        /// <summary>Priority - prefer ONNX Runtime over fallbacks.</summary>
        public override int Priority => 100;
        
        /// <summary>Whether GPU acceleration is enabled.</summary>
        public bool EnableGpu
        {
            get => _enableGpu;
            set => _enableGpu = value;
        }
        
        /// <summary>Create an inference session from model bytes.</summary>
        public override IOnnxSession CreateSession(byte[] modelData, OnnxSessionOptions options)
        {
            options ??= OnnxSessionOptions.Default;
            
            var sw = Stopwatch.StartNew();
            using var ortOptions = CreateOrtSessionOptions(options);
            var session = new InferenceSession(modelData, ortOptions);
            var elapsed = sw.ElapsedMilliseconds;
            UnityEngine.Debug.Log($"[OnnxRuntimeBackend] Session created in {elapsed}ms (model={modelData.Length / (1024f * 1024f):F1}MB)");
            return new OnnxRuntimeSession(session, _activeGpuProviderName, _gpuDeviceId);
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
            var sw = Stopwatch.StartNew();
            using var ortOptions = CreateOrtSessionOptions(options);
            var session = new InferenceSession(modelPath, ortOptions);
            UnityEngine.Debug.Log(
                $"[OnnxRuntimeBackend] Session created in {sw.ElapsedMilliseconds}ms " +
                $"(path={Path.GetFileName(modelPath)})");
            return new OnnxRuntimeSession(session, _activeGpuProviderName, _gpuDeviceId);
        }
        
        private SessionOptions CreateOrtSessionOptions(OnnxSessionOptions options)
        {
            var ortOptions = new SessionOptions
            {
                GraphOptimizationLevel = ToOrtOptimizationLevel(options.OptimizationLevel),
                EnableMemoryPattern = options.EnableMemoryPattern,
                EnableCpuMemArena = options.EnableCpuMemArena,
                LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR
            };
            
            if (options.IntraOpThreads > 0)
                ortOptions.IntraOpNumThreads = options.IntraOpThreads;
            
            if (options.InterOpThreads > 0)
                ortOptions.InterOpNumThreads = options.InterOpThreads;
            
            // Configure execution providers
            if (_enableGpu)
            {
                TryAddGpuProvider(ortOptions);
            }
            
            // Always add CPU as fallback
            ortOptions.AppendExecutionProvider_CPU(0);
            
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
        
        private void TryAddGpuProvider(SessionOptions options)
        {
            _activeGpuProviderName = null;
            if (_preferredProvider == GpuProvider.CPU)
                return;
                
            #if UNITY_EDITOR_WIN || UNITY_STANDALONE_WIN
            TryAddWindowsGpuProvider(options);
            #elif UNITY_EDITOR_LINUX || UNITY_STANDALONE_LINUX
            TryAddLinuxGpuProvider(options);
            #elif UNITY_EDITOR_OSX || UNITY_STANDALONE_OSX
            TryAddMacOSGpuProvider(options);
            #elif UNITY_ANDROID
            TryAddAndroidGpuProvider(options);
            #elif UNITY_IOS
            TryAddIOSGpuProvider(options);
            #endif
        }
        
        #if UNITY_EDITOR_WIN || UNITY_STANDALONE_WIN
        private void TryAddWindowsGpuProvider(SessionOptions options)
        {
            switch (_preferredProvider)
            {
                case GpuProvider.CUDA:
                    if (TryAddCuda(options)) return;
                    break;
                    
                case GpuProvider.DirectML:
                    if (TryAddDirectML(options)) return;
                    break;
                    
                case GpuProvider.TensorRT:
                    if (TryAddTensorRT(options)) return;
                    if (TryAddCuda(options)) return; // Fallback to CUDA
                    break;
                    
                case GpuProvider.Auto:
                default:
                    // Auto: Try TensorRT -> CUDA -> DirectML
                    // TensorRT is fastest for NVIDIA, CUDA is good fallback
                    // DirectML works with ALL GPUs (AMD, Intel, NVIDIA)
                    if (TryAddTensorRT(options)) return;
                    if (TryAddCuda(options)) return;
                    if (TryAddDirectML(options)) return;
                    break;
            }
            
            Debug.LogWarning("[OnnxRuntimeBackend] No GPU provider available on Windows. Using CPU.");
        }
        
        private bool TryAddCuda(SessionOptions options)
        {
            try
            {
                options.AppendExecutionProvider_CUDA(_gpuDeviceId);
                _activeGpuProviderName = "Cuda";
                Debug.Log($"[OnnxRuntimeBackend] Using CUDA execution provider (device {_gpuDeviceId})");
                return true;
            }
            catch (Exception e)
            {
                Debug.Log($"[OnnxRuntimeBackend] CUDA not available: {e.Message}");
                return false;
            }
        }
        
        private bool TryAddDirectML(SessionOptions options)
        {
            try
            {
                options.AppendExecutionProvider_DML(_gpuDeviceId);
                _activeGpuProviderName = "DML";
                Debug.Log($"[OnnxRuntimeBackend] Using DirectML execution provider (device {_gpuDeviceId}) - supports AMD, Intel, and NVIDIA GPUs");
                return true;
            }
            catch (Exception e)
            {
                Debug.Log($"[OnnxRuntimeBackend] DirectML not available: {e.Message}");
                return false;
            }
        }
        
        private bool TryAddTensorRT(SessionOptions options)
        {
            try
            {
                options.AppendExecutionProvider_Tensorrt(_gpuDeviceId);
                _activeGpuProviderName = "Tensorrt";
                Debug.Log($"[OnnxRuntimeBackend] Using TensorRT execution provider (device {_gpuDeviceId})");
                return true;
            }
            catch (Exception e)
            {
                Debug.Log($"[OnnxRuntimeBackend] TensorRT not available: {e.Message}");
                return false;
            }
        }
        #endif
        
        #if UNITY_EDITOR_LINUX || UNITY_STANDALONE_LINUX
        private void TryAddLinuxGpuProvider(SessionOptions options)
        {
            switch (_preferredProvider)
            {
                case GpuProvider.CUDA:
                    if (TryAddCudaLinux(options)) return;
                    break;
                    
                case GpuProvider.TensorRT:
                    if (TryAddTensorRTLinux(options)) return;
                    if (TryAddCudaLinux(options)) return;
                    break;
                    
                case GpuProvider.DirectML:
                    Debug.LogWarning("[OnnxRuntimeBackend] DirectML is not available on Linux. Use CUDA for NVIDIA GPUs.");
                    if (TryAddCudaLinux(options)) return;
                    break;
                    
                case GpuProvider.Auto:
                default:
                    // Auto: Try TensorRT -> CUDA
                    if (TryAddTensorRTLinux(options)) return;
                    if (TryAddCudaLinux(options)) return;
                    break;
            }
            
            Debug.LogWarning("[OnnxRuntimeBackend] No GPU provider available on Linux. Using CPU. Note: AMD GPUs require ROCm build of ONNX Runtime.");
        }
        
        private bool TryAddCudaLinux(SessionOptions options)
        {
            try
            {
                options.AppendExecutionProvider_CUDA(_gpuDeviceId);
                _activeGpuProviderName = "Cuda";
                Debug.Log($"[OnnxRuntimeBackend] Using CUDA execution provider (device {_gpuDeviceId})");
                return true;
            }
            catch (Exception e)
            {
                Debug.Log($"[OnnxRuntimeBackend] CUDA not available: {e.Message}");
                return false;
            }
        }
        
        private bool TryAddTensorRTLinux(SessionOptions options)
        {
            try
            {
                options.AppendExecutionProvider_Tensorrt(_gpuDeviceId);
                _activeGpuProviderName = "Tensorrt";
                Debug.Log($"[OnnxRuntimeBackend] Using TensorRT execution provider (device {_gpuDeviceId})");
                return true;
            }
            catch (Exception e)
            {
                Debug.Log($"[OnnxRuntimeBackend] TensorRT not available: {e.Message}");
                return false;
            }
        }
        #endif
        
        #if UNITY_EDITOR_OSX || UNITY_STANDALONE_OSX
        private void TryAddMacOSGpuProvider(SessionOptions options)
        {
            try
            {
                options.AppendExecutionProvider_CoreML();
                Debug.Log("[OnnxRuntimeBackend] Using CoreML execution provider");
            }
            catch (Exception e)
            {
                Debug.LogWarning($"[OnnxRuntimeBackend] CoreML not available: {e.Message}. Using CPU.");
            }
        }
        #endif
        
        #if UNITY_ANDROID
        private void TryAddAndroidGpuProvider(SessionOptions options)
        {
            try
            {
                options.AppendExecutionProvider_Nnapi();
                Debug.Log("[OnnxRuntimeBackend] Using NNAPI execution provider");
            }
            catch (Exception e)
            {
                Debug.LogWarning($"[OnnxRuntimeBackend] NNAPI not available: {e.Message}. Using CPU.");
            }
        }
        #endif
        
        #if UNITY_IOS
        private void TryAddIOSGpuProvider(SessionOptions options)
        {
            try
            {
                options.AppendExecutionProvider_CoreML();
                Debug.Log("[OnnxRuntimeBackend] Using CoreML execution provider");
            }
            catch (Exception e)
            {
                Debug.LogWarning($"[OnnxRuntimeBackend] CoreML not available: {e.Message}. Using CPU.");
            }
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
        
        public OnnxRuntimeSession(InferenceSession session, string gpuProviderName = null, int gpuDeviceId = 0)
        {
            _session = session ?? throw new ArgumentNullException(nameof(session));
            _gpuProviderName = gpuProviderName;
            _gpuDeviceId = gpuDeviceId;
            _inputNames = session.InputMetadata.Keys.ToList();
            _outputNames = session.OutputMetadata.Keys.ToList();
            
            // Prevent InferenceSession's native finalizer from ever running.
            // We control disposal explicitly; a leak is preferable to a crash
            // during domain reload when native DLLs may already be unloaded.
            GC.SuppressFinalize(_session);
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
                    dst[i] = (int)src[i];
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
        private static ushort FloatToHalf(float value)
        {
            uint bits = unchecked((uint)BitConverter.SingleToInt32Bits(value));
            uint sign = (bits >> 16) & 0x8000;
            int exp = (int)((bits >> 23) & 0xFF) - 127 + 15;
            uint mantissa = bits & 0x7FFFFF;
            
            if (exp <= 0)
            {
                if (exp < -10) return (ushort)sign; // too small → ±0
                mantissa |= 0x800000;
                int shift = 14 - exp;
                mantissa >>= shift;
                return (ushort)(sign | mantissa);
            }
            if (exp == 0xFF - 127 + 15)
            {
                // Inf/NaN
                return mantissa == 0
                    ? (ushort)(sign | 0x7C00)
                    : (ushort)(sign | 0x7C00 | (mantissa >> 13));
            }
            if (exp > 30) return (ushort)(sign | 0x7C00); // overflow → ±Inf
            
            return (ushort)(sign | ((uint)exp << 10) | (mantissa >> 13));
        }
        
        /// <summary>Convert IEEE 754 half-precision (ushort) to float32.</summary>
        private static float HalfToFloat(ushort value)
        {
            uint sign = (uint)(value & 0x8000) << 16;
            uint exp = (uint)(value >> 10) & 0x1F;
            uint mantissa = (uint)(value & 0x3FF);
            
            uint bits;
            if (exp == 0)
            {
                if (mantissa == 0) { bits = sign; } // ±0
                else
                {
                    // denormalized → normalize
                    exp = 1;
                    while ((mantissa & 0x400) == 0) { mantissa <<= 1; exp--; }
                    mantissa &= 0x3FF;
                    bits = sign | ((127 - 15 + exp) << 23) | (mantissa << 13);
                }
            }
            else if (exp == 31)
            {
                // Inf/NaN
                bits = sign | 0x7F800000 | (mantissa << 13);
            }
            else
            {
                bits = sign | ((exp + 127 - 15) << 23) | (mantissa << 13);
            }
            
            return BitConverter.Int32BitsToSingle(unchecked((int)bits));
        }
        
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
                    var ortDt = (OrtDeviceTensor)dt;
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
                
                // Wrap each output as IDeviceTensor
                var outputs = new List<IDeviceTensor>(_outputNames.Count);
                for (int i = 0; i < resultCollection.Count; i++)
                {
                    var ortValue = resultCollection[i];
                    // Take ownership: suppress SafeHandle finalizer to prevent crashes
                    // during domain reload. We dispose explicitly in OrtDeviceTensor.Dispose().
                    GC.SuppressFinalize(ortValue);
                    var deviceTensor = new OrtDeviceTensor(ortValue, _outputNames[i], this);
                    lock (_trackedDeviceTensors) _trackedDeviceTensors.Add(deviceTensor);
                    outputs.Add(deviceTensor);
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
            var outputs = new List<IDeviceTensor>(result.Count);
            foreach (var kv in result)
                outputs.Add(new CpuDeviceTensor(kv.Value) { Name = kv.Key });
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

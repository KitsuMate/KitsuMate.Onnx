using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;

[assembly: InternalsVisibleTo("KitsuMate.Onnx.Tests")]

namespace KitsuMate.Onnx
{
    public enum OnnxExecutionProvider
    {
        Cpu = 0,
        DirectMl = 1,
        Cuda = 2,
        TensorRt = 3,
        OpenVino = 4,
        Nnapi = 5,
        WebGpu = 6,
        CoreMl = 7,
        TensorRtRtx = 8
    }

    /// <summary>Controls whether a backend asset uses its serialized order or the registered platform policy.</summary>
    public enum OnnxProviderSelectionMode
    {
        /// <summary>Preserves the serialized provider order. This is the default value for pre-existing assets.</summary>
        Explicit = 0,
        /// <summary>Resolves the best registered providers for the current platform and installed extension packages.</summary>
        Automatic = 1
    }

    /// <summary>Controls how a provider-relative hardware device is selected.</summary>
    public enum OnnxDeviceSelectionMode
    {
        /// <summary>Use the backend's serialized provider-relative device index. This is the default for pre-existing assets.</summary>
        Explicit = 0,
        /// <summary>Prefer the GPU used by Unity's active graphics device, then use a deterministic fallback.</summary>
        Automatic = 1
    }

    /// <summary>A hardware device exposed by an ONNX Runtime execution provider.</summary>
    public sealed class OnnxExecutionDeviceInfo
    {
        public OnnxExecutionDeviceInfo(
            OnnxExecutionProvider provider,
            int providerDeviceIndex,
            string hardwareType,
            uint vendorId,
            uint hardwareDeviceId,
            string vendor,
            string providerVendor)
        {
            Provider = provider;
            ProviderDeviceIndex = providerDeviceIndex;
            HardwareType = hardwareType ?? string.Empty;
            VendorId = vendorId;
            HardwareDeviceId = hardwareDeviceId;
            Vendor = vendor ?? string.Empty;
            ProviderVendor = providerVendor ?? string.Empty;
        }

        public OnnxExecutionProvider Provider { get; }
        public int ProviderDeviceIndex { get; }
        public string HardwareType { get; }
        public uint VendorId { get; }
        public uint HardwareDeviceId { get; }
        public string Vendor { get; }
        public string ProviderVendor { get; }
        public string DisplayName => string.IsNullOrWhiteSpace(Vendor)
            ? $"{Provider} {HardwareType} #{ProviderDeviceIndex}"
            : $"{Vendor} {HardwareType} ({Provider} #{ProviderDeviceIndex})";
    }

    /// <summary>Records an execution-time provider transition.</summary>
    public sealed class OnnxProviderTransition
    {
        public OnnxProviderTransition(OnnxExecutionProvider from, OnnxExecutionProvider to, string reason)
        {
            From = from;
            To = to;
            Reason = reason ?? string.Empty;
        }

        public OnnxExecutionProvider From { get; }
        public OnnxExecutionProvider To { get; }
        public string Reason { get; }
    }

    public sealed class OnnxSessionDiagnostics
    {
        private readonly object _gate = new();
        private readonly List<string> _executionFailures = new();
        private readonly List<OnnxProviderTransition> _providerTransitions = new();
        private OnnxExecutionProvider _activeProvider;
        private OnnxExecutionDeviceInfo _activeDevice;

        public OnnxSessionDiagnostics(
            IEnumerable<OnnxExecutionProvider> configuredOrder,
            IEnumerable<OnnxExecutionProvider> eligibleOrder,
            IEnumerable<OnnxProviderSkip> skippedProviders,
            IEnumerable<OnnxExecutionProvider> attemptedProviders,
            IEnumerable<string> initializationFailures,
            IEnumerable<OnnxExecutionProvider> packagedProviders,
            OnnxExecutionProvider initializedPrimaryProvider,
            string runtimeVersion,
            int deviceId)
        {
            ConfiguredOrder = ReadOnly(configuredOrder, nameof(configuredOrder));
            EligibleOrder = ReadOnly(eligibleOrder, nameof(eligibleOrder));
            SkippedProviders = ReadOnly(skippedProviders, nameof(skippedProviders));
            AttemptedProviders = ReadOnly(attemptedProviders, nameof(attemptedProviders));
            InitializationFailures = ReadOnly(initializationFailures, nameof(initializationFailures));
            PackagedProviders = Array.AsReadOnly((packagedProviders ?? throw new ArgumentNullException(nameof(packagedProviders))).ToArray());
            InitializedPrimaryProvider = initializedPrimaryProvider;
            InitializationFallbackReason = InitializationFailures.Count == 0
                ? null
                : string.Join(" | ", InitializationFailures);
            RuntimeVersion = runtimeVersion ?? string.Empty;
            DeviceId = deviceId;
            _activeProvider = initializedPrimaryProvider;
        }

        public IReadOnlyList<OnnxExecutionProvider> ConfiguredOrder { get; }
        public IReadOnlyList<OnnxExecutionProvider> EligibleOrder { get; }
        public IReadOnlyList<OnnxProviderSkip> SkippedProviders { get; }
        public IReadOnlyList<OnnxExecutionProvider> AttemptedProviders { get; }
        public IReadOnlyList<string> InitializationFailures { get; }
        public IReadOnlyList<OnnxExecutionProvider> PackagedProviders { get; }
        public OnnxExecutionProvider InitializedPrimaryProvider { get; }
        public string InitializationFallbackReason { get; }
        public string RuntimeVersion { get; }
        public int DeviceId { get; }
        public OnnxExecutionProvider ActiveProvider { get { lock (_gate) return _activeProvider; } }
        public OnnxExecutionDeviceInfo ActiveDevice { get { lock (_gate) return _activeDevice; } }
        public IReadOnlyList<string> ExecutionFailures { get { lock (_gate) return _executionFailures.ToArray(); } }
        public IReadOnlyList<OnnxProviderTransition> ProviderTransitions { get { lock (_gate) return _providerTransitions.ToArray(); } }

        public void SetInitialDevice(OnnxExecutionDeviceInfo device)
        {
            lock (_gate) _activeDevice = device;
        }

        public void RecordExecutionFallback(
            OnnxExecutionProvider from,
            OnnxExecutionProvider to,
            OnnxExecutionDeviceInfo device,
            Exception exception)
        {
            string reason = exception?.GetBaseException().Message ?? "Execution provider failed.";
            lock (_gate)
            {
                _executionFailures.Add($"{from}: {reason}");
                _providerTransitions.Add(new OnnxProviderTransition(from, to, reason));
                _activeProvider = to;
                _activeDevice = device;
            }
        }

        private static IReadOnlyList<T> ReadOnly<T>(IEnumerable<T> values, string parameterName) =>
            Array.AsReadOnly((values ?? throw new ArgumentNullException(parameterName)).ToArray());
    }

    public sealed class OnnxProviderSkip
    {
        public OnnxProviderSkip(OnnxExecutionProvider provider, string reason)
        {
            Provider = provider;
            Reason = reason ?? string.Empty;
        }

        public OnnxExecutionProvider Provider { get; }
        public string Reason { get; }
    }

    public class OnnxProviderUnavailableException : InvalidOperationException
    {
        public OnnxProviderUnavailableException(OnnxExecutionProvider provider, string message, Exception innerException = null)
            : base(message, innerException) => Provider = provider;

        public OnnxExecutionProvider Provider { get; }
    }

    public class OnnxModelPreparationException : InvalidOperationException
    {
        public OnnxModelPreparationException(string message, Exception innerException = null) : base(message, innerException) { }
    }

    public class OnnxModelContractException : InvalidOperationException
    {
        public OnnxModelContractException(string message, Exception innerException = null) : base(message, innerException) { }
    }

    public class OnnxRuntimeLifecycleException : InvalidOperationException
    {
        public OnnxRuntimeLifecycleException(string message, Exception innerException = null) : base(message, innerException) { }
    }
}

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;

[assembly: InternalsVisibleTo("KitsuMate.Onnx.Tests")]

namespace KitsuMate.Onnx
{
    public enum OnnxExecutionProvider
    {
        Cpu,
        DirectMl,
        Cuda,
        TensorRt,
        OpenVino,
        Nnapi
    }

    public sealed class OnnxSessionDiagnostics
    {
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

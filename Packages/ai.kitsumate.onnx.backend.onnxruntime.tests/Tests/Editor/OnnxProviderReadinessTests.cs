using System.Collections.Generic;
using NUnit.Framework;
using UnityEditor;
using UnityEngine;

namespace KitsuMate.Onnx.Editor.Tests
{
    public sealed class OnnxProviderReadinessTests
    {
        [Test]
        public void EditorReadiness_ReportsEachBlockingGate()
        {
            Assert.That(Evaluate(configured: false).Kind, Is.EqualTo(OnnxProviderReadinessKind.NotConfigured));
            Assert.That(Evaluate(platform: RuntimePlatform.Android).Kind, Is.EqualTo(OnnxProviderReadinessKind.Unsupported));
            Assert.That(Evaluate(included: false).Kind, Is.EqualTo(OnnxProviderReadinessKind.NotIncluded));
            Assert.That(Evaluate(runtimeAvailable: false).Kind, Is.EqualTo(OnnxProviderReadinessKind.UnavailableInRuntime));
            Assert.That(Evaluate(probeComplete: false).Kind, Is.EqualTo(OnnxProviderReadinessKind.Checking));
            Assert.That(Evaluate(probeFailure: "Failed to load shared library").Kind, Is.EqualTo(OnnxProviderReadinessKind.MissingNativeDependencies));
            Assert.That(Evaluate(probeFailure: "GPU compute capability is unsupported").Kind, Is.EqualTo(OnnxProviderReadinessKind.UnsupportedHardware));
            Assert.That(Evaluate(probeFailure: "invalid device ordinal").Kind, Is.EqualTo(OnnxProviderReadinessKind.InvalidDevice));
            Assert.That(Evaluate(registrationFailure: "native payload was not found").Kind, Is.EqualTo(OnnxProviderReadinessKind.MissingPayload));
            Assert.That(Evaluate(registrationFailure: "failed to preload driver").Kind, Is.EqualTo(OnnxProviderReadinessKind.MissingNativeDependencies));
            Assert.That(Evaluate().Kind, Is.EqualTo(OnnxProviderReadinessKind.Ready));
        }

        [Test]
        public void PlayerReadiness_UsesInclusionRatherThanRuntimeReadiness()
        {
            OnnxProviderReadiness included = OnnxProviderReadinessEvaluator.EvaluatePlayer(
                OnnxExecutionProvider.Nnapi, true, RuntimePlatform.Android, true, true, true);
            Assert.That(included.Kind, Is.EqualTo(OnnxProviderReadinessKind.Ready));
            Assert.That(included.Text, Is.EqualTo("Included"));
            Assert.That(included.Reason, Does.Contain("verified only at runtime"));

            OnnxProviderReadiness omitted = OnnxProviderReadinessEvaluator.EvaluatePlayer(
                OnnxExecutionProvider.Nnapi, true, RuntimePlatform.Android, false, true, false);
            Assert.That(omitted.Kind, Is.EqualTo(OnnxProviderReadinessKind.NotIncluded));
            Assert.That(omitted.Reason, Does.Contain("No Build Profile"));
        }

        [Test]
        public void PlayerReadiness_ReportsMissingOwningPackagePayload()
        {
            OnnxProviderReadiness directMl = OnnxProviderReadinessEvaluator.EvaluatePlayer(
                OnnxExecutionProvider.DirectMl, true, RuntimePlatform.WindowsPlayer, true, false, true);
            Assert.That(directMl.Text, Is.EqualTo("Payload missing"));
            Assert.That(directMl.Reason, Does.Contain("owning package"));

            OnnxProviderReadiness cuda = OnnxProviderReadinessEvaluator.EvaluatePlayer(
                OnnxExecutionProvider.Cuda, true, RuntimePlatform.WindowsPlayer, true, false, true);
            Assert.That(cuda.Text, Is.EqualTo("Payload missing"));
        }

        [Test]
        public void BuildProfilePackaging_RespectsTargetAndDefines()
        {
            IReadOnlyList<OnnxExecutionProvider> windows = OnnxRuntimeProviderPluginFilter.GetPackagedProviders(
                BuildTarget.StandaloneWindows64,
                new[] { OnnxRuntimeProviderPluginFilter.NvidiaDefine });
            Assert.That(windows, Is.EqualTo(new[]
            {
                OnnxExecutionProvider.TensorRtRtx,
                OnnxExecutionProvider.Cuda,
                OnnxExecutionProvider.DirectMl,
                OnnxExecutionProvider.Cpu
            }));

            Assert.That(OnnxRuntimeProviderPluginFilter.GetPackagedProviders(
                BuildTarget.StandaloneWindows64,
                new[] { OnnxRuntimeProviderPluginFilter.CudaDefine }),
                Is.EqualTo(new[]
                {
                    OnnxExecutionProvider.Cuda,
                    OnnxExecutionProvider.DirectMl,
                    OnnxExecutionProvider.Cpu
                }));
            Assert.That(OnnxRuntimeProviderPluginFilter.GetPackagedProviders(
                BuildTarget.StandaloneLinux64,
                new[] { OnnxRuntimeProviderPluginFilter.CudaDefine, OnnxRuntimeProviderPluginFilter.TensorRtDefine }),
                Is.EqualTo(new[]
                {
                    OnnxExecutionProvider.TensorRtRtx,
                    OnnxExecutionProvider.Cuda,
                    OnnxExecutionProvider.WebGpu,
                    OnnxExecutionProvider.Cpu
                }));

            IReadOnlyList<OnnxExecutionProvider> android = OnnxRuntimeProviderPluginFilter.GetPackagedProviders(
                BuildTarget.Android,
                System.Array.Empty<string>());
            Assert.That(android, Is.EquivalentTo(new[] { OnnxExecutionProvider.Cpu, OnnxExecutionProvider.Nnapi }));
            Assert.That(OnnxRuntimeProviderPluginFilter.GetPackagedProviders(
                BuildTarget.StandaloneOSX, System.Array.Empty<string>()),
                Is.EqualTo(new[] { OnnxExecutionProvider.CoreMl, OnnxExecutionProvider.Cpu }));
        }

        [Test]
        public void AutomaticPolicy_UsesPlatformDefaultThenCpu()
        {
            bool nvidia = OnnxRuntimeProviderRegistry.TryGet(OnnxExecutionProvider.Cuda, out IOnnxRuntimeProviderModule cuda) && cuda.IsEnabled;
            Assert.That(OnnxRuntimeProviderRegistry.ResolveAutomatic(RuntimePlatform.WindowsPlayer),
                Is.EqualTo(nvidia
                    ? new[] { OnnxExecutionProvider.TensorRtRtx, OnnxExecutionProvider.Cuda, OnnxExecutionProvider.DirectMl, OnnxExecutionProvider.Cpu }
                    : new[] { OnnxExecutionProvider.DirectMl, OnnxExecutionProvider.Cpu }));
            Assert.That(OnnxRuntimeProviderRegistry.ResolveAutomatic(RuntimePlatform.LinuxPlayer),
                Is.EqualTo(nvidia
                    ? new[] { OnnxExecutionProvider.TensorRtRtx, OnnxExecutionProvider.Cuda, OnnxExecutionProvider.WebGpu, OnnxExecutionProvider.Cpu }
                    : new[] { OnnxExecutionProvider.WebGpu, OnnxExecutionProvider.Cpu }));
            Assert.That(OnnxRuntimeProviderRegistry.ResolveAutomatic(RuntimePlatform.OSXPlayer),
                Is.EqualTo(new[] { OnnxExecutionProvider.CoreMl, OnnxExecutionProvider.Cpu }));
            Assert.That(OnnxRuntimeProviderRegistry.ResolveAutomatic(RuntimePlatform.Android),
                Is.EqualTo(new[] { OnnxExecutionProvider.Nnapi, OnnxExecutionProvider.Cpu }));
        }

        [Test]
        public void LegacyProviders_HaveNoModules()
        {
            Assert.That(OnnxRuntimeProviderRegistry.TryGet(OnnxExecutionProvider.TensorRt, out _), Is.False);
            Assert.That(OnnxRuntimeProviderRegistry.TryGet(OnnxExecutionProvider.OpenVino, out _), Is.False);
        }

        [Test]
        public void NewAssetsDefaultAutomatic_AndExplicitSetterMigratesMode()
        {
            OnnxRuntimeBackend backend = ScriptableObject.CreateInstance<OnnxRuntimeBackend>();
            try
            {
                Assert.That(backend.ProviderSelectionMode, Is.EqualTo(OnnxProviderSelectionMode.Automatic));
                backend.SetProviderOrder(OnnxExecutionProvider.Cpu);
                Assert.That(backend.ProviderSelectionMode, Is.EqualTo(OnnxProviderSelectionMode.Explicit));
                Assert.That(backend.ProviderOrder, Is.EqualTo(new[] { OnnxExecutionProvider.Cpu }));
            }
            finally
            {
                Object.DestroyImmediate(backend);
            }
        }

        private static OnnxProviderReadiness Evaluate(
            bool configured = true,
            RuntimePlatform platform = RuntimePlatform.WindowsEditor,
            bool included = true,
            bool runtimeAvailable = true,
            bool probeComplete = true,
            string probeFailure = null,
            string registrationFailure = null)
        {
            return OnnxProviderReadinessEvaluator.EvaluateEditor(
                OnnxExecutionProvider.Cuda,
                configured,
                platform,
                included,
                runtimeAvailable,
                probeComplete,
                probeFailure,
                registrationFailure,
                null);
        }
    }
}

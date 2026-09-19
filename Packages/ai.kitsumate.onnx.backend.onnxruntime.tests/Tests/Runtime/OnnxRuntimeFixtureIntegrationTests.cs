using System;
using System.IO;
using NUnit.Framework;
using UnityEngine;

namespace KitsuMate.Onnx.Tests
{
    public sealed class OnnxRuntimeFixtureIntegrationTests
    {
        [Test]
        public void WindowsRuntime_RejectsMissingCoreBeforeNativeInitialization()
        {
            Assert.Throws<DllNotFoundException>(() => OnnxRuntimeProviderRegistry.ValidateWindowsRuntime(
                Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"), "onnxruntime.dll")));
        }

#if UNITY_EDITOR_WIN || UNITY_STANDALONE_WIN
        [Test]
        public void WindowsRuntime_RejectsWrongVersionBeforeNativeInitialization()
        {
            string directory = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(directory);
            try
            {
                string core = Path.Combine(directory, "onnxruntime.dll");
                File.Copy(typeof(object).Assembly.Location, core);
                Assert.That(Assert.Throws<InvalidOperationException>(() =>
                    OnnxRuntimeProviderRegistry.ValidateWindowsRuntime(core)).Message, Does.Contain("does not match"));
            }
            finally { Directory.Delete(directory, true); }
        }
#endif

        [Test]
        public void ProviderOrder_RejectsDuplicates()
        {
            var backend = ScriptableObject.CreateInstance<OnnxRuntimeBackend>();
            try
            {
                Assert.Throws<ArgumentException>(() => backend.SetProviderOrder(
                    OnnxExecutionProvider.Cpu, OnnxExecutionProvider.Cpu));
            }
            finally { UnityEngine.Object.DestroyImmediate(backend); }
        }

        [Test]
        public void ProviderOrder_RejectsEmptyOrder()
        {
            var backend = ScriptableObject.CreateInstance<OnnxRuntimeBackend>();
            try { Assert.Throws<ArgumentException>(() => backend.SetProviderOrder()); }
            finally { UnityEngine.Object.DestroyImmediate(backend); }
        }

        [Test]
        public void ProviderOrder_FiltersPlatformAndProfileWithoutReordering()
        {
            OnnxExecutionProvider[] configured =
            {
                OnnxExecutionProvider.TensorRt,
                OnnxExecutionProvider.Cuda,
                OnnxExecutionProvider.WebGpu,
                OnnxExecutionProvider.OpenVino,
                OnnxExecutionProvider.Nnapi,
                OnnxExecutionProvider.Cpu,
            };

            Assert.That(OnnxRuntimeBackend.ResolveProviderOrder(configured,
                    new[] { OnnxExecutionProvider.WebGpu, OnnxExecutionProvider.Cpu }, RuntimePlatform.WindowsPlayer),
                Is.EqualTo(new[] { OnnxExecutionProvider.WebGpu, OnnxExecutionProvider.Cpu }));
            Assert.That(OnnxRuntimeBackend.ResolveProviderOrder(configured,
                    new[] { OnnxExecutionProvider.Nnapi, OnnxExecutionProvider.Cpu }, RuntimePlatform.Android),
                Is.EqualTo(new[] { OnnxExecutionProvider.Nnapi, OnnxExecutionProvider.Cpu }));
            Assert.That(OnnxRuntimeBackend.ResolveProviderOrder(configured,
                    new[] { OnnxExecutionProvider.Cpu }, RuntimePlatform.LinuxPlayer),
                Is.EqualTo(new[] { OnnxExecutionProvider.Cpu }));
        }

        [Test]
        public void ProviderOrder_FailsWhenNoPreferenceIsEligible()
        {
            OnnxProviderUnavailableException exception = Assert.Throws<OnnxProviderUnavailableException>(() =>
                OnnxRuntimeBackend.ResolveProviderOrder(
                    new[] { OnnxExecutionProvider.WebGpu },
                    new[] { OnnxExecutionProvider.Cpu },
                    RuntimePlatform.WindowsPlayer));
            Assert.That(exception.Provider, Is.EqualTo(OnnxExecutionProvider.WebGpu));
        }

        [Test]
        public void NewBackend_DefaultsToAutomaticProviderAndDeviceSelection()
        {
            var backend = ScriptableObject.CreateInstance<OnnxRuntimeBackend>();
            try
            {
                Assert.That(backend.ProviderSelectionMode, Is.EqualTo(OnnxProviderSelectionMode.Automatic));
                Assert.That(backend.DeviceSelectionMode, Is.EqualTo(OnnxDeviceSelectionMode.Automatic));
                backend.DeviceId = 1;
                Assert.That(backend.DeviceSelectionMode, Is.EqualTo(OnnxDeviceSelectionMode.Explicit));
            }
            finally { UnityEngine.Object.DestroyImmediate(backend); }
        }

        [Test]
        public void AutomaticDeviceSelection_MatchesUnityRenderDevicePerProvider()
        {
            var devices = new[]
            {
                Device(0, 0x1002, 0x164e, "AMD"),
                Device(1, 0x10de, 0x2684, "NVIDIA")
            };
            OnnxRuntimeGraphicsDeviceHint.SetForTests(0x10de, 0x2684, "RTX");

            OnnxExecutionDeviceInfo selected = OnnxRuntimeBackend.SelectDevice(
                OnnxExecutionProvider.WebGpu, devices, OnnxDeviceSelectionMode.Automatic, 0, out string reason);

            Assert.That(selected.ProviderDeviceIndex, Is.EqualTo(1));
            StringAssert.Contains("matched Unity graphics device", reason);
        }

        [Test]
        public void AutomaticDeviceSelection_UsesDeterministicProviderZeroWhenSeveralDoNotMatch()
        {
            var devices = new[]
            {
                Device(0, 0x1002, 1, "AMD"),
                Device(1, 0x10de, 2, "NVIDIA")
            };
            OnnxRuntimeGraphicsDeviceHint.SetForTests(0x8086, 3, "Intel");

            OnnxExecutionDeviceInfo selected = OnnxRuntimeBackend.SelectDevice(
                OnnxExecutionProvider.WebGpu, devices, OnnxDeviceSelectionMode.Automatic, 1, out string reason);

            Assert.That(selected.ProviderDeviceIndex, Is.EqualTo(0));
            StringAssert.Contains("deterministically selected", reason);
        }

        [Test]
        public void ExplicitDeviceSelection_RejectsProviderRelativeIndexOutsideRange()
        {
            Assert.Throws<OnnxProviderUnavailableException>(() => OnnxRuntimeBackend.SelectDevice(
                OnnxExecutionProvider.WebGpu,
                new[] { Device(0, 0x1002, 1, "AMD") },
                OnnxDeviceSelectionMode.Explicit,
                1,
                out _));
        }

        private static OnnxExecutionDeviceInfo Device(int index, uint vendorId, uint deviceId, string vendor) =>
            new(OnnxExecutionProvider.WebGpu, index, "GPU", vendorId, deviceId, vendor, "Microsoft");

        [TestCase(OnnxExecutionProvider.Cpu)]
        [TestCase(OnnxExecutionProvider.WebGpu)]
        [Category("Integration")]
        public void DynamicHostInput_RunsAndReturnsExpectedValues(OnnxExecutionProvider provider)
        {
            if (!OnnxRuntimeProviderRegistry.TryGet(provider, out var module) || !module.Supports(Application.platform))
                Assert.Ignore("Provider is not supported on this platform.");
            var backend = ScriptableObject.CreateInstance<OnnxRuntimeBackend>();
            try
            {
                backend.SetProviderOrder(provider);
                // A dynamic shape feeds a scalar into GPU arithmetic, exercising host transfers.
                byte[] model = Convert.FromBase64String("CAk6zAEKEQoBeBIFc2hhcGUiBVNoYXBlCioKBXNoYXBlCgVpbmRleBIFYmF0Y2giBkdhdGhlcioLCgRheGlzGACgAQIKIAoFYmF0Y2gSBm9mZnNldCIEQ2FzdCoJCgJ0bxgBoAECChMKAXgKBm9mZnNldBIBeSIDQWRkEhJkeW5hbWljX2hvc3RfaW5wdXQqDBAHOgEAQgVpbmRleFoYCgF4EhMKEQgBEg0KBxIFYmF0Y2gKAggCYhgKAXkSEwoRCAESDQoHEgViYXRjaAoCCAJCBAoAEBE=");
                IOnnxSession session;
                try { session = backend.CreateSession(model, new OnnxSessionOptions()); }
                catch (OnnxProviderUnavailableException ex) when (IsMissingHardwareAdapter(ex))
                {
                    Assert.Ignore($"No {provider} adapter is available in this environment.");
                    return;
                }
                using (session)
                {
                    using var input = OnnxTensor.FromArray(new[] { 1f, 2f, 3f, 4f }, new[] { 2, 2 });
                    var inputs = new System.Collections.Generic.Dictionary<string, OnnxTensor> { ["x"] = input };
                    for (int run = 0; run < 2; run++)
                    {
                        var outputs = session.Run(inputs);
                        try { Assert.That(outputs["y"].AsFloatArray(), Is.EqualTo(new[] { 3f, 4f, 5f, 6f })); }
                        finally { foreach (OnnxTensor output in outputs.Values) output.Dispose(); }
                    }
                    Assert.That(session.Diagnostics.InitializedPrimaryProvider, Is.EqualTo(provider));
                }
            }
            finally { UnityEngine.Object.DestroyImmediate(backend); }
        }

        private static bool IsMissingHardwareAdapter(OnnxProviderUnavailableException ex) =>
            ex.ToString().IndexOf("adapter", StringComparison.OrdinalIgnoreCase) >= 0;

        [TestCase("Gelu", -0.15865525f, 0.84134475f)]
        [TestCase("Elu", -0.63212056f, 1f)]
        [Category("Integration")]
        public void WebGpu_ConvActivation_RunsWithoutCpuSessionFallback(string activation, float negative, float positive)
        {
            if (!OnnxRuntimeProviderRegistry.TryGet(OnnxExecutionProvider.WebGpu, out var module) || !module.Supports(Application.platform))
                Assert.Ignore("WebGPU is not supported on this platform.");
            // Opset 20: a unit-weight Conv followed by one activation, with no model weights required.
            string model = activation == "Gelu"
                ? "CAk6dQoPCgF4CgF3EgFjIgRDb252CgwKAWMSAXkiBEdlbHUSD2NvbnZfYWN0aXZhdGlvbioRCAEIAQgBEAEiBAAAgD9CAXdaFwoBeBISChAIARIMCgIIAQoCCAEKAggDYhcKAXkSEgoQCAESDAoCCAEKAggBCgIIA0IECgAQFA=="
                : "CAk6dAoPCgF4CgF3EgFjIgRDb252CgsKAWMSAXkiA0VsdRIPY29udl9hY3RpdmF0aW9uKhEIAQgBCAEQASIEAACAP0IBd1oXCgF4EhIKEAgBEgwKAggBCgIIAQoCCANiFwoBeRISChAIARIMCgIIAQoCCAEKAggDQgQKABAU";
            var backend = ScriptableObject.CreateInstance<OnnxRuntimeBackend>();
            try
            {
                backend.SetProviderOrder(OnnxExecutionProvider.WebGpu);
                IOnnxSession session;
                try { session = backend.CreateSession(Convert.FromBase64String(model), new OnnxSessionOptions()); }
                catch (OnnxProviderUnavailableException ex) when (IsMissingHardwareAdapter(ex))
                {
                    Assert.Ignore("No WebGpu adapter is available in this environment.");
                    return;
                }
                using (session)
                {
                    using var input = OnnxTensor.FromArray(new[] { -1f, 0f, 1f }, new[] { 1, 1, 3 });
                    var outputs = session.Run(new System.Collections.Generic.Dictionary<string, OnnxTensor> { ["x"] = input });
                    try { Assert.That(outputs["y"].AsFloatArray(), Is.EqualTo(new[] { negative, 0f, positive }).Within(0.00001f)); }
                    finally { foreach (OnnxTensor output in outputs.Values) output.Dispose(); }
                    var deviceSession = (IOnnxDeviceSession)session;
                    var deviceOutputs = deviceSession.RunOnDevice(new[] { new OnnxNamedValue("x", input) },
                        Array.Empty<IDeviceTensor>(), Array.Empty<string>());
                    try
                    {
                        // This output is genuinely GPU-resident, so ToCpu must perform
                        // a transfer rather than exposing a GPU address as a managed span.
                        using OnnxTensor output = deviceOutputs[0].ToCpu();
                        Assert.That(output.AsFloatArray(), Is.EqualTo(new[] { negative, 0f, positive }).Within(0.00001f));
                    }
                    finally { foreach (IDeviceTensor output in deviceOutputs) output.Dispose(); }
                    deviceOutputs = deviceSession.RunOnDevice(new[] { new OnnxNamedValue("x", input) },
                        Array.Empty<IDeviceTensor>(), new[] { "y" });
                    try
                    {
                        using OnnxTensor output = deviceOutputs[0].ToCpu();
                        Assert.That(output.AsFloatArray(), Is.EqualTo(new[] { negative, 0f, positive }).Within(0.00001f));
                    }
                    finally { foreach (IDeviceTensor output in deviceOutputs) output.Dispose(); }
                    Assert.That(session.Diagnostics.InitializedPrimaryProvider, Is.EqualTo(OnnxExecutionProvider.WebGpu));
                }
            }
            finally { UnityEngine.Object.DestroyImmediate(backend); }
        }

        private const string FixturePath = "KitsuMateOnnxFixtures/whisper/mel.onnx";

        [Test]
        [Category("Integration")]
        public void SmokeFixture_CreatesAnOnnxRuntimeSession()
        {
            string absolutePath = Path.GetFullPath(Path.Combine(Application.dataPath, "..", "KitsuMateOnnxFixtures", "whisper", "mel.onnx"));
            Assert.That(File.Exists(absolutePath), Is.True, $"Missing required ONNX Runtime fixture at {FixturePath}. Download the required CPU CI fixture set.");

            var backend = ScriptableObject.CreateInstance<OnnxRuntimeBackend>();
            try
            {
                backend.SetProviderOrder(OnnxExecutionProvider.Cpu);
                Assert.That(backend.ProviderOrder, Is.EqualTo(new[]
                {
                    OnnxExecutionProvider.Cpu
                }));
                CollectionAssert.AreEquivalent(new[]
                {
                    RuntimePlatform.WindowsEditor,
                    RuntimePlatform.WindowsPlayer,
                    RuntimePlatform.LinuxEditor,
                    RuntimePlatform.LinuxPlayer,
                    RuntimePlatform.OSXEditor,
                    RuntimePlatform.OSXPlayer,
                    RuntimePlatform.Android,
                }, backend.SupportedPlatforms);
                Assert.That(backend.IsAvailable, Is.True, "The provisioned native runtime should be loadable.");
                using IOnnxSession session = backend.CreateSession(absolutePath, new OnnxSessionOptions());
                Assert.That(session.InputNames, Is.Not.Empty);
                Assert.That(session.OutputNames, Is.Not.Empty);
                Assert.That(session.Diagnostics.ConfiguredOrder, Is.EqualTo(backend.ProviderOrder));
                Assert.That(session.Diagnostics.EligibleOrder, Is.EqualTo(new[] { OnnxExecutionProvider.Cpu }));
                Assert.That(session.Diagnostics.AttemptedProviders, Is.EqualTo(new[] { OnnxExecutionProvider.Cpu }));
                Assert.That(session.Diagnostics.InitializedPrimaryProvider, Is.EqualTo(OnnxExecutionProvider.Cpu));
                Assert.That(session.Diagnostics.SkippedProviders, Is.Empty);
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(backend);
            }
        }
    }
}

using System;
using System.IO;
using NUnit.Framework;
using UnityEngine;

namespace KitsuMate.Onnx.Tests
{
    public sealed class OnnxRuntimeFixtureIntegrationTests
    {
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
                OnnxExecutionProvider.DirectMl,
                OnnxExecutionProvider.OpenVino,
                OnnxExecutionProvider.Nnapi,
                OnnxExecutionProvider.Cpu,
            };

            Assert.That(OnnxRuntimeBackend.ResolveProviderOrder(configured,
                    new[] { OnnxExecutionProvider.DirectMl, OnnxExecutionProvider.Cpu }, RuntimePlatform.WindowsPlayer),
                Is.EqualTo(new[] { OnnxExecutionProvider.DirectMl, OnnxExecutionProvider.Cpu }));
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
                    new[] { OnnxExecutionProvider.DirectMl },
                    new[] { OnnxExecutionProvider.Cpu },
                    RuntimePlatform.WindowsPlayer));
            Assert.That(exception.Provider, Is.EqualTo(OnnxExecutionProvider.DirectMl));
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
                OnnxExecutionProvider.DirectMl, devices, OnnxDeviceSelectionMode.Automatic, 0, out string reason);

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
                OnnxExecutionProvider.DirectMl, devices, OnnxDeviceSelectionMode.Automatic, 1, out string reason);

            Assert.That(selected.ProviderDeviceIndex, Is.EqualTo(0));
            StringAssert.Contains("deterministically selected", reason);
        }

        [Test]
        public void ExplicitDeviceSelection_RejectsProviderRelativeIndexOutsideRange()
        {
            Assert.Throws<OnnxProviderUnavailableException>(() => OnnxRuntimeBackend.SelectDevice(
                OnnxExecutionProvider.DirectMl,
                new[] { Device(0, 0x1002, 1, "AMD") },
                OnnxDeviceSelectionMode.Explicit,
                1,
                out _));
        }

        private static OnnxExecutionDeviceInfo Device(int index, uint vendorId, uint deviceId, string vendor) =>
            new(OnnxExecutionProvider.DirectMl, index, "GPU", vendorId, deviceId, vendor, "Microsoft");

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

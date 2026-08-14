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
                Assert.That(backend.ProviderOrder, Is.EqualTo(new[]
                {
                    OnnxExecutionProvider.TensorRt,
                    OnnxExecutionProvider.Cuda,
                    OnnxExecutionProvider.DirectMl,
                    OnnxExecutionProvider.OpenVino,
                    OnnxExecutionProvider.Nnapi,
                    OnnxExecutionProvider.Cpu,
                }));
                CollectionAssert.AreEquivalent(new[]
                {
                    RuntimePlatform.WindowsEditor,
                    RuntimePlatform.WindowsPlayer,
                    RuntimePlatform.LinuxEditor,
                    RuntimePlatform.LinuxPlayer,
                    RuntimePlatform.Android,
                }, backend.SupportedPlatforms);
                Assert.That(backend.IsAvailable, Is.True, "The hydrated native runtime should be loadable.");
                using IOnnxSession session = backend.CreateSession(absolutePath, new OnnxSessionOptions());
                Assert.That(session.InputNames, Is.Not.Empty);
                Assert.That(session.OutputNames, Is.Not.Empty);
                Assert.That(session.Diagnostics.ConfiguredOrder, Is.EqualTo(backend.ProviderOrder));
                Assert.That(session.Diagnostics.EligibleOrder, Is.EqualTo(new[] { OnnxExecutionProvider.Cpu }));
                Assert.That(session.Diagnostics.AttemptedProviders, Is.EqualTo(new[] { OnnxExecutionProvider.Cpu }));
                Assert.That(session.Diagnostics.InitializedPrimaryProvider, Is.EqualTo(OnnxExecutionProvider.Cpu));
                Assert.That(session.Diagnostics.SkippedProviders, Has.Count.EqualTo(5));
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(backend);
            }
        }
    }
}

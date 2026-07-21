using System.IO;
using NUnit.Framework;
using UnityEngine;

namespace KitsuMate.Onnx.Tests
{
    public sealed class OnnxRuntimeFixtureIntegrationTests
    {
        private const string FixturePath = "Assets/KitsuMateOnnxFixtures/onnxruntime/smoke.onnx";

        [Test]
        [Category("Integration")]
        public void SmokeFixture_CreatesAnOnnxRuntimeSession()
        {
            string absolutePath = Path.Combine(Application.dataPath, "KitsuMateOnnxFixtures", "onnxruntime", "smoke.onnx");
            Assert.That(File.Exists(absolutePath), Is.True, $"Missing required ONNX Runtime fixture at {FixturePath}. Hydrate the required CPU CI fixture set.");

            var backend = ScriptableObject.CreateInstance<OnnxRuntimeBackend>();
            try
            {
                using IOnnxSession session = backend.CreateSession(absolutePath, new OnnxSessionOptions());
                Assert.That(session.InputNames, Is.Not.Empty);
                Assert.That(session.OutputNames, Is.Not.Empty);
            }
            finally
            {
                Object.DestroyImmediate(backend);
            }
        }
    }
}

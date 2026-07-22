using System.IO;
using NUnit.Framework;
using UnityEngine;

namespace KitsuMate.Onnx.Tests
{
    public sealed class OnnxRuntimeFixtureIntegrationTests
    {
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

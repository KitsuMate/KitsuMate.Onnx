using System;
using NUnit.Framework;
using UnityEngine;

namespace KitsuMate.Onnx.Tests
{
    public sealed class UnityAiInferenceBackendTests
    {
        [Test]
        public void Backend_IdentifiesOnlyTheUnityAiInferenceContract()
        {
            var backend = ScriptableObject.CreateInstance<UnityAiInferenceBackend>();
            try
            {
                Assert.That(backend.BackendId, Is.EqualTo("unity-ai-inference"));
                Assert.That(backend.DisplayName, Is.EqualTo("Unity AI Inference"));
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(backend);
            }
        }

        [Test]
        public void RawModelBytes_AreRejectedWithAnActionableMessage()
        {
            var backend = ScriptableObject.CreateInstance<UnityAiInferenceBackend>();
            try
            {
                var error = Assert.Throws<NotSupportedException>(() => backend.CreateSession(Array.Empty<byte>(), new OnnxSessionOptions()));
                Assert.That(error.Message, Does.Contain("UnityAiInferenceModelAsset"));
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(backend);
            }
        }
    }
}

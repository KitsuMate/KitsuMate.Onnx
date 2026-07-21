#if UNITY_EDITOR
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEditor;
using UnityEngine;

namespace KitsuMate.Onnx.Tests
{
    public sealed class UnityAiInferenceFixtureIntegrationTests
    {
        private const string FixturePath = "Assets/KitsuMateOnnxFixtures/unity-ai-inference/smoke.onnx";

        [Test]
        [Category("Integration")]
        public void SmokeFixture_ImportsAndCreatesAUnityAiInferenceSession()
        {
            var modelAsset = AssetDatabase.LoadAssetAtPath<ModelAsset>(FixturePath);
            Assert.That(modelAsset, Is.Not.Null, $"Missing or unsupported Unity AI Inference fixture at {FixturePath}. Hydrate the required CPU CI fixture set and allow Unity to import it.");

            var source = ScriptableObject.CreateInstance<UnityAiInferenceModelAsset>();
            var backend = ScriptableObject.CreateInstance<UnityAiInferenceBackend>();
            try
            {
                var serialized = new SerializedObject(source);
                serialized.FindProperty("modelAsset").objectReferenceValue = modelAsset;
                serialized.ApplyModifiedPropertiesWithoutUndo();

                using IOnnxSession session = backend.CreateSession(source);
                Assert.That(session.InputNames, Is.Not.Empty);
                Assert.That(session.OutputNames, Is.Not.Empty);
            }
            finally
            {
                Object.DestroyImmediate(source);
                Object.DestroyImmediate(backend);
            }
        }
    }
}
#endif

#if UNITY_EDITOR
using System.IO;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEditor;
using UnityEngine;

namespace KitsuMate.Onnx.Tests
{
    public sealed class UnityAiInferenceFixtureIntegrationTests
    {
        private const string FixturePath = "Assets/KitsuMateOnnxFixtures/unity-ai-inference/model.onnx";

        [Test]
        [Category("Integration")]
        public void DefaultOnnxFixture_LoadsWithBothEngines()
        {
            var modelAsset = AssetDatabase.LoadAssetAtPath<ModelAsset>(FixturePath);
            Assert.That(modelAsset, Is.Not.Null, $"Missing or unsupported Unity AI Inference fixture at {FixturePath}. Download the required CPU CI fixture set and allow Unity to import it.");

            var source = ScriptableObject.CreateInstance<UnityAiInferenceModelAsset>();
            var sentis = ScriptableObject.CreateInstance<UnityAiInferenceBackend>();
            var onnxRuntime = ScriptableObject.CreateInstance<OnnxRuntimeBackend>();
            try
            {
                var serialized = new SerializedObject(source);
                serialized.FindProperty("modelAsset").objectReferenceValue = modelAsset;
                serialized.ApplyModifiedPropertiesWithoutUndo();

                using IOnnxSession sentisSession = sentis.CreateSession(source);
                Assert.That(sentisSession.InputNames, Is.Not.Empty);
                Assert.That(sentisSession.OutputNames, Is.Not.Empty);

                onnxRuntime.SetProviderOrder(OnnxExecutionProvider.Cpu);
                string rawOnnxPath = Path.GetFullPath(Path.Combine(
                    Directory.GetCurrentDirectory(), FixturePath));
                using IOnnxSession onnxRuntimeSession = onnxRuntime.CreateSession(rawOnnxPath);
                Assert.That(onnxRuntimeSession.InputNames, Is.EquivalentTo(sentisSession.InputNames));
                Assert.That(onnxRuntimeSession.OutputNames, Is.EquivalentTo(sentisSession.OutputNames));
            }
            finally
            {
                Object.DestroyImmediate(source);
                Object.DestroyImmediate(sentis);
                Object.DestroyImmediate(onnxRuntime);
            }
        }
    }
}
#endif

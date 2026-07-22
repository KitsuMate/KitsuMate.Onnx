#if UNITY_EDITOR
using System;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Threading.Tasks;
using KitsuMate.Onnx.Embeddings.UnityAiInference;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEditor;
using UnityEngine;

namespace KitsuMate.Onnx.Embeddings.UnityAiInference.Tests
{
    public sealed class UnityAiInferenceEmbeddingIntegrationTests
    {
        private const string ModelPath = "Assets/KitsuMateOnnxFixtures/unity-ai-inference/model_unity_fp32.onnx";

        [Test]
        [Category("Integration")]
        public async Task AllMiniLm_ProducesANormalizedEmbeddingOnUnityCpu()
        {
            ModelAsset imported = AssetDatabase.LoadAssetAtPath<ModelAsset>(ModelPath);
            Assert.That(imported, Is.Not.Null, $"Unity AI Inference did not import {ModelPath}.");
            var source = ScriptableObject.CreateInstance<UnityAiInferenceModelAsset>();
            var modelSet = ScriptableObject.CreateInstance<UnityAiInferenceTextEmbeddingModelSet>();
            var engine = ScriptableObject.CreateInstance<UnityAiInferenceTextEmbeddingEngine>();
            var backend = ScriptableObject.CreateInstance<UnityAiInferenceBackend>();
            string root = Path.GetFullPath(Path.Combine(Application.dataPath, "..", "KitsuMateOnnxFixtures", "text-embedding", "all-minilm"));
            var tokenizer = new TextAsset(File.ReadAllText(Path.Combine(root, "tokenizer.json")));
            var vocabulary = new TextAsset(File.ReadAllText(Path.Combine(root, "vocab.txt")));
            InferenceEngineRuntime<EmbeddingRequest, EmbeddingResult> runtime = null;
            try
            {
                source.SetModelAsset(imported);
                modelSet.SetModels(source, vocabulary, tokenizer);
                SetField(engine, "modelSet", modelSet);
                backend.Device = UnityAiInferenceDevice.Cpu;
                runtime = await engine.CreateRuntimeAsync(backend);
                EmbeddingResult result = await runtime.RunAsync(new EmbeddingRequest("Unity CPU inference test."));
                Assert.That(result.Embedding, Has.Length.EqualTo(384));
                Assert.That(result.Embedding.All(float.IsFinite), Is.True);
                Assert.That(Math.Sqrt(result.Embedding.Sum(value => value * value)), Is.EqualTo(1d).Within(0.001d));
            }
            finally
            {
                runtime?.Dispose();
                UnityEngine.Object.DestroyImmediate(tokenizer);
                UnityEngine.Object.DestroyImmediate(vocabulary);
                UnityEngine.Object.DestroyImmediate(backend);
                UnityEngine.Object.DestroyImmediate(engine);
                UnityEngine.Object.DestroyImmediate(modelSet);
                UnityEngine.Object.DestroyImmediate(source);
            }
        }

        private static void SetField(object target, string fieldName, object value)
        {
            FieldInfo field = target.GetType().GetField(fieldName, BindingFlags.Instance | BindingFlags.NonPublic)
                ?? throw new MissingFieldException(target.GetType().FullName, fieldName);
            field.SetValue(target, value);
        }
    }
}
#endif

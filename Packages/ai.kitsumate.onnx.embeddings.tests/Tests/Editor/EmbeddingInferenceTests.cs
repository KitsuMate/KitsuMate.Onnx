#if UNITY_EDITOR
using System;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Threading.Tasks;
using NUnit.Framework;
using UnityEditor;
using UnityEngine;

namespace KitsuMate.Onnx.Embeddings.Tests
{
    public sealed class EmbeddingInferenceTests
    {
        [Test]
        [Category("Integration")]
        public async Task AllMiniLm_ProducesANormalizedEmbeddingOnCpu()
        {
            ConfigureModelRoot();
            var modelSet = ScriptableObject.CreateInstance<TextEmbeddingModelSet>();
            var engine = ScriptableObject.CreateInstance<TextEmbeddingEngine>();
            var backend = CpuBackend();
            TextAsset tokenizer = LoadText("text-embedding/all-minilm/tokenizer.json");
            TextAsset vocabulary = LoadText("text-embedding/all-minilm/vocab.txt");
            InferenceEngineRuntime<EmbeddingRequest, EmbeddingResult> runtime = null;
            try
            {
                modelSet.EmbeddingModel.ConfigureFile("text-embedding/all-minilm/model_q4f16.onnx", string.Empty, null, null);
                modelSet.SetTokenizer(vocabulary, tokenizer);
                SetField(engine, "modelSet", modelSet);
                runtime = await engine.CreateRuntimeAsync(backend);
                EmbeddingResult result = await runtime.RunAsync(new EmbeddingRequest("A small real-model CPU integration test."));
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
            }
        }

        [Test]
        [Category("Integration")]
        public async Task Llm2Vec_ProducesA4096ValueEmbeddingOnCpu()
        {
            ConfigureModelRoot();
            var modelSet = ScriptableObject.CreateInstance<Llm2VecModelSet>();
            var engine = ScriptableObject.CreateInstance<Llm2VecEmbeddingEngine>();
            var backend = CpuBackend();
            TextAsset tokenizer = LoadText("llm2vec/tokenizer.json");
            TextAsset tokenizerConfig = LoadText("llm2vec/tokenizer_config.json");
            InferenceEngineRuntime<EmbeddingRequest, EmbeddingResult> runtime = null;
            try
            {
                modelSet.Encoder.ConfigureFile("llm2vec/model_int4_b128_fp32act_s64.onnx", string.Empty, null, null);
                modelSet.SetFiles(tokenizer, tokenizerConfig);
                SetField(engine, "modelSet", modelSet);
                runtime = await engine.CreateRuntimeAsync(backend);
                EmbeddingResult result = await runtime.RunAsync(new EmbeddingRequest("A person walks forward and waves."));
                Assert.That(result.Embedding, Has.Length.EqualTo(4096));
                Assert.That(result.Embedding.All(float.IsFinite), Is.True);
                Assert.That(result.Embedding.Any(value => Math.Abs(value) > 0.000001f), Is.True);
            }
            finally
            {
                runtime?.Dispose();
                UnityEngine.Object.DestroyImmediate(tokenizer);
                UnityEngine.Object.DestroyImmediate(tokenizerConfig);
                UnityEngine.Object.DestroyImmediate(backend);
                UnityEngine.Object.DestroyImmediate(engine);
                UnityEngine.Object.DestroyImmediate(modelSet);
            }
        }

        private static OnnxRuntimeBackend CpuBackend()
        {
            var backend = ScriptableObject.CreateInstance<OnnxRuntimeBackend>();
            backend.EnableGpu = false;
            backend.PreferredProvider = GpuProvider.CPU;
            return backend;
        }

        private static TextAsset LoadText(string path)
        {
            return new TextAsset(File.ReadAllText(Path.Combine(ModelRoot(), path.Replace('/', Path.DirectorySeparatorChar))));
        }

        private static void ConfigureModelRoot()
        {
            const string assetPath = "Assets/Resources/OnnxSettings.asset";
            if (!AssetDatabase.IsValidFolder("Assets/Resources")) AssetDatabase.CreateFolder("Assets", "Resources");
            OnnxSettings settings = AssetDatabase.LoadAssetAtPath<OnnxSettings>(assetPath);
            if (settings == null)
            {
                settings = ScriptableObject.CreateInstance<OnnxSettings>();
                AssetDatabase.CreateAsset(settings, assetPath);
            }
            var serialized = new SerializedObject(settings);
            serialized.FindProperty("_modelStorageRoot").stringValue = "KitsuMateOnnxFixtures";
            serialized.ApplyModifiedPropertiesWithoutUndo();
        }

        private static string ModelRoot()
        {
            return Path.GetFullPath(Path.Combine(Application.dataPath, "..", "KitsuMateOnnxFixtures"));
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

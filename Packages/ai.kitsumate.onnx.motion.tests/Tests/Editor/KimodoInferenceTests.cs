#if UNITY_EDITOR
using System;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Threading.Tasks;
using KitsuMate.Onnx.Embeddings;
using KitsuMate.Onnx.Motion.Kimodo;
using NUnit.Framework;
using UnityEditor;
using UnityEngine;

namespace KitsuMate.Onnx.Motion.Tests
{
    public sealed class KimodoInferenceTests
    {
        [Test]
        [Category("Integration")]
        public async Task Kimodo_GeneratesFiniteMotionOnCpu()
        {
            ConfigureModelRoot();
            var embeddingSet = ScriptableObject.CreateInstance<Llm2VecModelSet>();
            var modelSet = ScriptableObject.CreateInstance<KimodoModelSet>();
            var engine = ScriptableObject.CreateInstance<KimodoEngine>();
            var backend = ScriptableObject.CreateInstance<OnnxRuntimeBackend>();
            var tokenizer = new TextAsset(File.ReadAllText(Path.Combine(ModelRoot(), "llm2vec", "tokenizer.json")));
            var tokenizerConfig = new TextAsset(File.ReadAllText(Path.Combine(ModelRoot(), "llm2vec", "tokenizer_config.json")));
            InferenceEngineRuntime<CharacterMotionRequest, CharacterMotionResult> runtime = null;
            try
            {
                backend.EnableGpu = false;
                backend.PreferredProvider = GpuProvider.CPU;
                embeddingSet.Encoder.ConfigureFile("llm2vec/model_int4_b128_fp32act_s64.onnx", string.Empty, null, null);
                embeddingSet.SetFiles(tokenizer, tokenizerConfig);
                modelSet.MotionModel.ConfigureFile("kimodo/model_fp16_b3_t60.onnx", string.Empty, null, null);
                modelSet.SetRequiredEmbedding(embeddingSet);
                SetField(engine, "modelSet", modelSet);
                runtime = await engine.CreateRuntimeAsync(backend);
                CharacterMotionResult result = await runtime.RunAsync(new CharacterMotionRequest
                {
                    Embedding = new KimodoTextEmbedding(new float[KimodoTextEmbedding.Dimension]),
                    Generation = new KimodoGenerationRequest(seed: 1, denoisingSteps: 2)
                });
                Assert.That(result.Motion, Is.Not.Null);
                Assert.That(result.Motion.FrameCount, Is.EqualTo(60));
                Assert.That(result.Motion.RootPositions.All(position => float.IsFinite(position.x) && float.IsFinite(position.y) && float.IsFinite(position.z)), Is.True);
            }
            finally
            {
                runtime?.Dispose();
                UnityEngine.Object.DestroyImmediate(tokenizer);
                UnityEngine.Object.DestroyImmediate(tokenizerConfig);
                UnityEngine.Object.DestroyImmediate(backend);
                UnityEngine.Object.DestroyImmediate(engine);
                UnityEngine.Object.DestroyImmediate(modelSet);
                UnityEngine.Object.DestroyImmediate(embeddingSet);
            }
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

        private static string ModelRoot() => Path.GetFullPath(Path.Combine(Application.dataPath, "..", "KitsuMateOnnxFixtures"));

        private static void SetField(object target, string fieldName, object value)
        {
            FieldInfo field = target.GetType().GetField(fieldName, BindingFlags.Instance | BindingFlags.NonPublic)
                ?? throw new MissingFieldException(target.GetType().FullName, fieldName);
            field.SetValue(target, value);
        }
    }
}
#endif

#if UNITY_EDITOR
using System;
using System.IO;
using System.Reflection;
using System.Threading.Tasks;
using KitsuMate.Onnx.LipSync.Uni2005;
using NUnit.Framework;
using UnityEditor;
using UnityEngine;

namespace KitsuMate.Onnx.LipSync.Tests
{
    public sealed class Uni2005InferenceTests
    {
        [Test]
        [Category("Integration")]
        public async Task Uni2005_ProducesAVisemeTimelineOnCpu()
        {
            ConfigureModelRoot();
            var modelSet = ScriptableObject.CreateInstance<Uni2005ModelSet>();
            var engine = ScriptableObject.CreateInstance<Uni2005Engine>();
            var backend = ScriptableObject.CreateInstance<OnnxRuntimeBackend>();
            var vocabulary = new TextAsset(File.ReadAllText(Path.Combine(ModelRoot(), "uni2005", "vocab.json")));
            InferenceEngineRuntime<LipSyncRequest, VisemeTimeline> runtime = null;
            try
            {
                backend.EnableGpu = false;
                backend.PreferredProvider = GpuProvider.CPU;
                modelSet.AcousticModel.ConfigureFile("uni2005/model_fp32.onnx", string.Empty, null, null);
                modelSet.SetVocabulary(vocabulary);
                SetField(engine, "modelSet", modelSet);
                SetField(engine, "volumeThreshold", 0f);
                runtime = await engine.CreateRuntimeAsync(backend);
                var samples = new float[8000];
                for (int index = 0; index < samples.Length; index++)
                    samples[index] = 0.1f * Mathf.Sin(index * 2f * Mathf.PI * (180f + 60f * Mathf.Sin(index / 500f)) / 8000f);
                VisemeTimeline result = await runtime.RunAsync(new LipSyncRequest(samples, 8000));
                Assert.That(result, Is.Not.Null);
                Assert.That(result.FrameCount, Is.GreaterThan(0));
                Assert.That(result.Duration, Is.GreaterThan(0f));
            }
            finally
            {
                runtime?.Dispose();
                UnityEngine.Object.DestroyImmediate(vocabulary);
                UnityEngine.Object.DestroyImmediate(backend);
                UnityEngine.Object.DestroyImmediate(engine);
                UnityEngine.Object.DestroyImmediate(modelSet);
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

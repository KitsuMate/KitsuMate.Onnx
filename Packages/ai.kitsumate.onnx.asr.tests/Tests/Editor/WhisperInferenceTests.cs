#if UNITY_EDITOR
using System;
using System.IO;
using System.Reflection;
using System.Threading.Tasks;
using KitsuMate.Onnx.Asr.Whisper;
using NUnit.Framework;
using UnityEditor;
using UnityEngine;

namespace KitsuMate.Onnx.Asr.Tests
{
    public sealed class WhisperInferenceTests
    {
        [Test]
        [Category("Integration")]
        public async Task WhisperTiny_RunsMelEncoderAndDecoderOnCpu()
        {
            ConfigureModelRoot();
            var modelSet = ScriptableObject.CreateInstance<WhisperModelSet>();
            var engine = ScriptableObject.CreateInstance<WhisperEngine>();
            var backend = ScriptableObject.CreateInstance<OnnxRuntimeBackend>();
            var tokenizer = new TextAsset(File.ReadAllText(Path.Combine(ModelRoot(), "whisper", "tokenizer.json")));
            AudioClip audio = AudioClip.Create("WhisperCpuFixture", 16000, 1, 16000, false);
            var samples = new float[16000];
            for (int index = 0; index < samples.Length; index++) samples[index] = 0.05f * Mathf.Sin(index * 2f * Mathf.PI * 220f / 16000f);
            audio.SetData(samples, 0);
            InferenceEngineRuntime<AsrRequest, TranscriptionResult> runtime = null;
            try
            {
                backend.EnableGpu = false;
                backend.PreferredProvider = GpuProvider.CPU;
                modelSet.MelProcessor.ConfigureFile("whisper/mel.onnx", string.Empty, null, null);
                modelSet.Encoder.ConfigureFile("whisper/encoder_model_bnb4.onnx", string.Empty, null, null);
                modelSet.Decoder.ConfigureFile("whisper/decoder_model_merged_bnb4.onnx", string.Empty, null, null);
                modelSet.SetTokenizer(tokenizer);
                SetField(engine, "modelSet", modelSet);
                SetField(engine, "languageOverride", "en");
                SetField(engine, "maxTokens", 1);
                runtime = await engine.CreateRuntimeAsync(backend);
                TranscriptionResult result = await runtime.RunAsync(new AsrRequest(audio));
                Assert.That(result, Is.Not.Null);
                Assert.That(result.Text, Is.Not.Null);
                Assert.That(result.Language, Is.EqualTo("en"));
                Assert.That(result.Duration, Is.GreaterThan(0f));
            }
            finally
            {
                runtime?.Dispose();
                UnityEngine.Object.DestroyImmediate(audio);
                UnityEngine.Object.DestroyImmediate(tokenizer);
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

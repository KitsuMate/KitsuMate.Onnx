using System;
using System.Reflection;
using System.Threading.Tasks;
using KitsuMate.Onnx.Asr.Whisper;
using NUnit.Framework;
using Unity.InferenceEngine;
using UnityEditor;
using UnityEngine;

namespace KitsuMate.Onnx.Asr.Sentis.Tests
{
    public sealed class SentisWhisperIntegrationTests
    {
        private const string Root = "Assets/KitsuMateOnnxFixtures/whisper-sentis";

        [Test]
        [Category("Integration")]
        public async Task Tiny_TranscribesRealAudioOnCpu()
        {
            TranscriptionResult result = await Run("tiny", 16);
            Assert.That(result.Text, Is.Not.Empty);
            Assert.That(result.Language, Is.EqualTo("en"));
        }

        [Test]
        [Category("Integration")]
        public async Task Base_RunsInitialAndCachedDecoderOnCpu()
        {
            TranscriptionResult result = await Run("base", 2);
            Assert.That(result, Is.Not.Null);
            Assert.That(result.Language, Is.EqualTo("en"));
        }

        [Test]
        [Category("Integration")]
        public async Task Tiny_FirstDecodedTokenMatchesOnnxRuntime()
        {
            TranscriptionResult sentis = await Run("tiny", 1);
            TranscriptionResult onnxRuntime = await RunOnnxRuntime(1);
            Assert.That(sentis.Text, Is.Not.Empty);
            Assert.That(sentis.Text, Is.EqualTo(onnxRuntime.Text));
        }

        private static async Task<TranscriptionResult> Run(string size, int maxTokens)
        {
            ModelAsset mel = Load<ModelAsset>($"{Root}/mel.onnx");
            ModelAsset encoder = Load<ModelAsset>($"{Root}/{size}/encoder_model.onnx");
            ModelAsset decoder = Load<ModelAsset>($"{Root}/{size}/decoder_model.onnx");
            ModelAsset decoderWithPast = Load<ModelAsset>($"{Root}/{size}/decoder_with_past_model.onnx");
            TextAsset tokenizer = Load<TextAsset>($"{Root}/tokenizer.json");
            AudioClip audio = Load<AudioClip>($"{Root}/answering-machine16kHz.wav");
            var melSource = Source(mel);
            var encoderSource = Source(encoder);
            var decoderSource = Source(decoder);
            var cachedDecoderSource = Source(decoderWithPast);
            var modelSet = ScriptableObject.CreateInstance<SentisWhisperModelSet>();
            var engine = ScriptableObject.CreateInstance<SentisWhisperEngine>();
            var backend = ScriptableObject.CreateInstance<UnityAiInferenceBackend>();
            InferenceEngineRuntime<AsrRequest, TranscriptionResult> runtime = null;
            try
            {
                modelSet.SetModels(melSource, encoderSource, decoderSource, cachedDecoderSource, tokenizer);
                SetField(engine, "modelSet", modelSet);
                SetField(engine, "languageOverride", "en");
                SetField(engine, "maxTokens", maxTokens);
                backend.Device = UnityAiInferenceDevice.Cpu;
                runtime = await engine.CreateRuntimeAsync(backend);
                return await runtime.RunAsync(new AsrRequest(audio));
            }
            finally
            {
                runtime?.Dispose();
                UnityEngine.Object.DestroyImmediate(backend);
                UnityEngine.Object.DestroyImmediate(engine);
                UnityEngine.Object.DestroyImmediate(modelSet);
                UnityEngine.Object.DestroyImmediate(cachedDecoderSource);
                UnityEngine.Object.DestroyImmediate(decoderSource);
                UnityEngine.Object.DestroyImmediate(encoderSource);
                UnityEngine.Object.DestroyImmediate(melSource);
            }
        }

        private static async Task<TranscriptionResult> RunOnnxRuntime(int maxTokens)
        {
            ConfigureModelRoot();
            TextAsset tokenizer = Load<TextAsset>($"{Root}/tokenizer.json");
            AudioClip audio = Load<AudioClip>($"{Root}/answering-machine16kHz.wav");
            var modelSet = ScriptableObject.CreateInstance<WhisperModelSet>();
            var engine = ScriptableObject.CreateInstance<WhisperEngine>();
            var backend = ScriptableObject.CreateInstance<OnnxRuntimeBackend>();
            InferenceEngineRuntime<AsrRequest, TranscriptionResult> runtime = null;
            try
            {
                modelSet.MelProcessor.ConfigureFile("whisper-sentis/mel.onnx", string.Empty, null, null);
                modelSet.Encoder.ConfigureFile("whisper-sentis/tiny/encoder_model.onnx", string.Empty, null, null);
                modelSet.Decoder.ConfigureFile("whisper-sentis/tiny/decoder_model.onnx", string.Empty, null, null);
                modelSet.DecoderWithPast.ConfigureFile(
                    "whisper-sentis/tiny/decoder_with_past_model.onnx", string.Empty, null, null);
                modelSet.SetTokenizer(tokenizer);
                SetField(engine, "modelSet", modelSet);
                SetField(engine, "languageOverride", "en");
                SetField(engine, "maxTokens", maxTokens);
                backend.EnableGpu = false;
                backend.PreferredProvider = GpuProvider.CPU;
                runtime = await engine.CreateRuntimeAsync(backend);
                return await runtime.RunAsync(new AsrRequest(audio));
            }
            finally
            {
                runtime?.Dispose();
                UnityEngine.Object.DestroyImmediate(backend);
                UnityEngine.Object.DestroyImmediate(engine);
                UnityEngine.Object.DestroyImmediate(modelSet);
            }
        }

        private static void ConfigureModelRoot()
        {
            const string assetPath = "Assets/Resources/OnnxSettings.asset";
            if (!AssetDatabase.IsValidFolder("Assets/Resources"))
                AssetDatabase.CreateFolder("Assets", "Resources");
            OnnxSettings settings = AssetDatabase.LoadAssetAtPath<OnnxSettings>(assetPath);
            if (settings == null)
            {
                settings = ScriptableObject.CreateInstance<OnnxSettings>();
                AssetDatabase.CreateAsset(settings, assetPath);
            }
            var serialized = new SerializedObject(settings);
            serialized.FindProperty("_modelStorageRoot").stringValue = "Assets/KitsuMateOnnxFixtures";
            serialized.ApplyModifiedPropertiesWithoutUndo();
        }

        private static UnityAiInferenceModelAsset Source(ModelAsset model)
        {
            var source = ScriptableObject.CreateInstance<UnityAiInferenceModelAsset>();
            source.SetModelAsset(model);
            return source;
        }

        private static T Load<T>(string path) where T : UnityEngine.Object
        {
            T asset = AssetDatabase.LoadAssetAtPath<T>(path);
            Assert.That(asset, Is.Not.Null, $"Unity did not import {path} as {typeof(T).Name}.");
            return asset;
        }

        private static void SetField(object target, string fieldName, object value)
        {
            FieldInfo field = target.GetType().GetField(fieldName, BindingFlags.Instance | BindingFlags.NonPublic)
                ?? throw new MissingFieldException(target.GetType().FullName, fieldName);
            field.SetValue(target, value);
        }
    }
}

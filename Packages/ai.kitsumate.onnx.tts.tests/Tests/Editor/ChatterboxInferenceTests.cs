#if UNITY_EDITOR
using System;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Threading.Tasks;
using KitsuMate.Onnx.Tts.Chatterbox;
using NUnit.Framework;
using UnityEditor;
using UnityEngine;

namespace KitsuMate.Onnx.Tts.Tests
{
    public sealed class ChatterboxInferenceTests
    {
        [Test]
        [Category("Integration")]
        public async Task Chatterbox_ProducesAudioOnCpu()
        {
            ConfigureModelRoot();
            var modelSet = ScriptableObject.CreateInstance<ChatterboxModelSet>();
            var engine = ScriptableObject.CreateInstance<ChatterboxEngine>();
            var backend = ScriptableObject.CreateInstance<OnnxRuntimeBackend>();
            var tokenizer = new TextAsset(File.ReadAllText(Path.Combine(ModelRoot(), "chatterbox", "tokenizer.json")));
            AudioClip voice = LoadPcm16Wav(Path.Combine(ModelRoot(), "chatterbox", "default_voice.wav"));
            InferenceEngineRuntime<TtsRequest, TtsResult> runtime = null;
            try
            {
                backend.EnableGpu = false;
                backend.PreferredProvider = GpuProvider.CPU;
                modelSet.SpeechEncoder.ConfigureFile("chatterbox/speech_encoder_q4f16.onnx", string.Empty, null, null);
                modelSet.EmbedTokens.ConfigureFile("chatterbox/embed_tokens_q4f16.onnx", string.Empty, null, null);
                modelSet.LanguageModel.ConfigureFile("chatterbox/language_model_q4f16.onnx", string.Empty, null, null);
                modelSet.ConditionalDecoder.ConfigureFile("chatterbox/conditional_decoder_q4f16.onnx", string.Empty, null, null);
                modelSet.SetFiles(tokenizer, null, voice);
                SetField(engine, "modelSet", modelSet);
                SetField(engine, "voiceCacheCapacity", 0);
                runtime = await engine.CreateRuntimeAsync(backend);
                TtsResult result = await runtime.RunAsync(new TtsRequest("Hello.") { MaxNewTokens = 1 });
                Assert.That(result.Samples, Is.Not.Null.And.Not.Empty);
                Assert.That(result.Samples.All(float.IsFinite), Is.True);
                Assert.That(result.SampleRate, Is.EqualTo(24000));
                Assert.That(result.Duration, Is.GreaterThan(0f));
            }
            finally
            {
                runtime?.Dispose();
                UnityEngine.Object.DestroyImmediate(voice);
                UnityEngine.Object.DestroyImmediate(tokenizer);
                UnityEngine.Object.DestroyImmediate(backend);
                UnityEngine.Object.DestroyImmediate(engine);
                UnityEngine.Object.DestroyImmediate(modelSet);
            }
        }

        private static AudioClip LoadPcm16Wav(string path)
        {
            using var reader = new BinaryReader(File.OpenRead(path));
            if (new string(reader.ReadChars(4)) != "RIFF") throw new InvalidDataException("Voice fixture is not RIFF WAV.");
            reader.ReadInt32();
            if (new string(reader.ReadChars(4)) != "WAVE") throw new InvalidDataException("Voice fixture is not WAVE audio.");
            ushort channels = 0;
            ushort format = 0;
            int sampleRate = 0;
            ushort bits = 0;
            byte[] data = null;
            while (reader.BaseStream.Position + 8 <= reader.BaseStream.Length)
            {
                string chunk = new string(reader.ReadChars(4));
                int length = reader.ReadInt32();
                if (chunk == "fmt ")
                {
                    format = reader.ReadUInt16();
                    channels = reader.ReadUInt16();
                    sampleRate = reader.ReadInt32();
                    reader.BaseStream.Position += 6;
                    bits = reader.ReadUInt16();
                    reader.BaseStream.Position += length - 16;
                }
                else if (chunk == "data") data = reader.ReadBytes(length);
                else reader.BaseStream.Position += length;
                if ((length & 1) != 0) reader.BaseStream.Position++;
            }
            bool pcm16 = format == 1 && bits == 16;
            bool float32 = format == 3 && bits == 32;
            if (channels == 0 || sampleRate == 0 || (!pcm16 && !float32) || data == null)
                throw new InvalidDataException("Voice fixture must contain PCM16 or float32 samples.");
            int sampleBytes = bits / 8;
            var samples = new float[data.Length / sampleBytes];
            for (int index = 0; index < samples.Length; index++)
                samples[index] = float32 ? BitConverter.ToSingle(data, index * sampleBytes) :
                    BitConverter.ToInt16(data, index * sampleBytes) / 32768f;
            var clip = AudioClip.Create("ChatterboxVoiceFixture", samples.Length / channels, channels, sampleRate, false);
            clip.SetData(samples, 0);
            return clip;
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

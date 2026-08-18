#if UNITY_EDITOR
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using KitsuMate.Onnx.Tts.Chatterbox;
using KitsuMate.Tokenizers;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using UnityEditor;
using UnityEngine;
using Debug = UnityEngine.Debug;

namespace KitsuMate.Onnx.Tts.Tests
{
    public sealed class ChatterboxInferenceTests
    {
        [Test]
        [Category("Integration")]
        [Timeout(600000)]
        public void ChatterboxNano_MatchesOfficialFixture()
        {
            string fixtureName = Environment.GetEnvironmentVariable("KITSUMATE_CHATTERBOX_FIXTURE")
                ?? "chatterbox-ci";
            string root = Path.Combine(ModelRoot(), fixtureName);
            JObject fixture = ReadFixture(Path.Combine(root, "official-parity.json.gz"));
            var backend = ScriptableObject.CreateInstance<OnnxRuntimeBackend>();
            AudioClip voice = LoadPcm16Wav(Path.Combine(root, "default_voice.wav"));
            try
            {
                backend.SetProviderOrder(OnnxExecutionProvider.Cpu);

                string normalized = (string)typeof(ChatterboxEngineRuntime)
                    .GetMethod("PrepareModernText", BindingFlags.Static | BindingFlags.NonPublic)
                    ?.Invoke(null, new object[] { fixture.Value<string>("text") });
                Assert.That(normalized, Is.EqualTo(fixture.Value<string>("normalizedText")));

                long[] expectedIds = LongTensor(fixture, "input_ids");
                Tokenizer tokenizer = Tokenizer.FromTokenizerJson(
                    Encoding.UTF8.GetBytes(File.ReadAllText(Path.Combine(root, "tokenizer.json"))));
                long[] inputIds = tokenizer.Encode(normalized, addSpecialTokens: true)
                    .Ids.Select(static id => (long)id).ToArray();
                CollectionAssert.AreEqual(expectedIds, inputIds, "KitsuMate.Tokenizers differs from AutoTokenizer.");

                using IOnnxSession embedSession = backend.CreateSession(FindPath(root, "embed_tokens_*.onnx"));
                IReadOnlyDictionary<string, OnnxTensor> embedOutputs = embedSession.Run(
                    new Dictionary<string, OnnxTensor>
                    {
                        ["input_ids"] = OnnxTensor.FromArray(inputIds, new[] { 1, inputIds.Length })
                    });
                OnnxTensor textEmbeddings = embedOutputs["inputs_embeds"];
                AssertFloatParity(textEmbeddings.AsFloatArray(), FloatTensor(fixture, "inputs_embeds"),
                    "text embeddings");

                var voiceSamples = new float[voice.samples * voice.channels];
                voice.GetData(voiceSamples, 0);
                using IOnnxSession speechSession = backend.CreateSession(
                    Path.Combine(root, "speech_encoder_q4f16.onnx"));
                IReadOnlyDictionary<string, OnnxTensor> speechOutputs = speechSession.Run(
                    new Dictionary<string, OnnxTensor>
                    {
                        ["audio_values"] = OnnxTensor.FromArray(
                            voiceSamples, new[] { 1, voiceSamples.Length })
                    });
                AssertFloatParity(speechOutputs["audio_features"].AsFloatArray(),
                    FloatTensor(fixture, "audio_features"), "audio features");
                CollectionAssert.AreEqual(LongTensor(fixture, "audio_tokens"),
                    speechOutputs["audio_tokens"].AsLongArray(), "audio tokens");
                AssertFloatParity(speechOutputs["speaker_embeddings"].AsFloatArray(),
                    FloatTensor(fixture, "speaker_embeddings"), "speaker embeddings", 5e-3f);
                AssertFloatParity(speechOutputs["speaker_features"].AsFloatArray(),
                    FloatTensor(fixture, "speaker_features"), "speaker features", 5e-3f);

                OnnxTensor audioFeatures = speechOutputs["audio_features"];
                float[] combined = Concat(
                    audioFeatures.AsFloatArray(), textEmbeddings.AsFloatArray());
                int sequenceLength = audioFeatures.Shape[1] + textEmbeddings.Shape[1];
                using IOnnxSession languageSession = backend.CreateSession(
                    Path.Combine(root, "language_model_q4f16.onnx"));
                var languageInputs = new Dictionary<string, OnnxTensor>
                {
                    ["inputs_embeds"] = OnnxTensor.FromArray(
                        combined, new[] { 1, sequenceLength, textEmbeddings.Shape[2] }),
                    ["attention_mask"] = OnnxTensor.FromArray(
                        Enumerable.Repeat(1L, sequenceLength).ToArray(), new[] { 1, sequenceLength }),
                    ["position_ids"] = OnnxTensor.FromArray(
                        Enumerable.Range(0, sequenceLength).Select(static index => (long)index).ToArray(),
                        new[] { 1, sequenceLength })
                };
                foreach (string input in languageSession.InputNames.Where(
                             static name => name.StartsWith("past_key_values.", StringComparison.Ordinal)))
                {
                    languageInputs[input] = OnnxTensor.FromArray(
                        Array.Empty<ushort>(), new[] { 1, 12, 0, ChatterboxConstants.HeadDim });
                }
                IReadOnlyDictionary<string, OnnxTensor> languageOutputs = languageSession.Run(languageInputs);
                float[] logits = languageOutputs["logits"].AsFloatArray();
                int vocabSize = languageOutputs["logits"].Shape[^1];
                int offset = logits.Length - vocabSize;
                int firstToken = 0;
                for (int index = 1; index < vocabSize; index++)
                    if (logits[offset + index] > logits[offset + firstToken]) firstToken = index;
                Assert.That(firstToken, Is.EqualTo(LongTensor(fixture, "first_token")[0]));

                using IOnnxSession decoderSession = backend.CreateSession(
                    FindPath(root, "conditional_decoder_*.onnx"));
                long[] decoderTokens = LongTensor(fixture, "decoder_speech_tokens");
                IReadOnlyDictionary<string, OnnxTensor> decoderOutputs = decoderSession.Run(
                    new Dictionary<string, OnnxTensor>
                    {
                        ["speech_tokens"] = OnnxTensor.FromArray(
                            decoderTokens, new[] { 1, decoderTokens.Length }),
                        ["speaker_embeddings"] = speechOutputs["speaker_embeddings"],
                        ["speaker_features"] = speechOutputs["speaker_features"]
                    });
                AssertFloatParity(decoderOutputs["waveform"].AsFloatArray(),
                    FloatTensor(fixture, "waveform"), "decoded waveform", 1e-3f, 2e-4f);

                Dispose(languageOutputs);
                Dispose(decoderOutputs);
                Dispose(speechOutputs);
                Dispose(embedOutputs);
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(voice);
                UnityEngine.Object.DestroyImmediate(backend);
            }
        }

        [Test]
        [Category("Integration")]
        [Timeout(600000)]
        public async Task Chatterbox_ProducesAudioOnCpu()
        {
            ConfigureModelRoot();
            var modelSet = ScriptableObject.CreateInstance<ChatterboxModelSet>();
            var engine = ScriptableObject.CreateInstance<ChatterboxEngine>();
            var backend = ScriptableObject.CreateInstance<OnnxRuntimeBackend>();
            string fixtureName = Environment.GetEnvironmentVariable("KITSUMATE_CHATTERBOX_FIXTURE")
                ?? "chatterbox-ci";
            string fixtureRoot = Path.Combine(ModelRoot(), fixtureName);
            var tokenizer = new TextAsset(File.ReadAllText(Path.Combine(fixtureRoot, "tokenizer.json")));
            AudioClip voice = LoadPcm16Wav(Path.Combine(fixtureRoot, "default_voice.wav"));
            InferenceEngineRuntime<TtsRequest, TtsResult> runtime = null;
            try
            {
                backend.SetProviderOrder(OnnxExecutionProvider.Cpu);
                modelSet.SpeechEncoder.ConfigureFile(
                    fixtureName + "/speech_encoder_q4f16.onnx", string.Empty, null, null);
                modelSet.EmbedTokens.ConfigureFile(
                    fixtureName + "/" + FindFile(fixtureRoot, "embed_tokens_*.onnx"), string.Empty, null, null);
                modelSet.LanguageModel.ConfigureFile(
                    fixtureName + "/language_model_q4f16.onnx", string.Empty, null, null);
                modelSet.ConditionalDecoder.ConfigureFile(
                    fixtureName + "/" + FindFile(fixtureRoot, "conditional_decoder_*.onnx"), string.Empty, null, null);
                modelSet.SetFiles(tokenizer, null, voice);
                SetField(engine, "modelSet", modelSet);
                SetField(engine, "voiceCacheCapacity", 0);
                SetField(engine, "verboseLogging", true);
                var load = Stopwatch.StartNew();
                runtime = await engine.CreateRuntimeAsync(backend);
                load.Stop();
                var inference = Stopwatch.StartNew();
                TtsResult result = await runtime.RunAsync(
                    new TtsRequest("Oh, that's hilarious! [chuckle]") { MaxNewTokens = 1 });
                inference.Stop();
                Assert.That(result.Samples, Is.Not.Null.And.Not.Empty);
                Assert.That(result.Samples.All(float.IsFinite), Is.True);
                Assert.That(result.SampleRate, Is.EqualTo(24000));
                Assert.That(result.Duration, Is.GreaterThan(0f));
                Debug.Log($"CHATTERBOX_BENCHMARK load_ms={load.ElapsedMilliseconds} " +
                    $"inference_ms={inference.ElapsedMilliseconds}");
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

        private static string FindFile(string directory, string pattern)
        {
            string[] files = Directory.GetFiles(directory, pattern, SearchOption.TopDirectoryOnly);
            if (files.Length != 1)
                throw new InvalidDataException($"Expected one '{pattern}' fixture in '{directory}', found {files.Length}.");
            return Path.GetFileName(files[0]);
        }

        private static string FindPath(string directory, string pattern) =>
            Path.Combine(directory, FindFile(directory, pattern));

        private static JObject ReadFixture(string path)
        {
            Assert.That(File.Exists(path), Is.True, $"Required parity fixture is missing: {path}");
            using var stream = new GZipStream(File.OpenRead(path), CompressionMode.Decompress);
            using var reader = new StreamReader(stream, Encoding.UTF8);
            return JObject.Parse(reader.ReadToEnd());
        }

        private static float[] FloatTensor(JObject fixture, string name)
        {
            JObject tensor = (JObject)fixture["tensors"]?[name];
            Assert.That(tensor?.Value<string>("dtype"), Is.EqualTo("<f4"));
            byte[] bytes = Convert.FromBase64String(tensor.Value<string>("data"));
            var values = new float[bytes.Length / sizeof(float)];
            Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length);
            return values;
        }

        private static long[] LongTensor(JObject fixture, string name)
        {
            JObject tensor = (JObject)fixture["tensors"]?[name];
            Assert.That(tensor?.Value<string>("dtype"), Is.EqualTo("<i8"));
            byte[] bytes = Convert.FromBase64String(tensor.Value<string>("data"));
            var values = new long[bytes.Length / sizeof(long)];
            Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length);
            return values;
        }

        private static void AssertFloatParity(float[] actual, float[] expected, string label,
            float maximumTolerance = 1e-4f, float rootMeanSquareTolerance = float.PositiveInfinity)
        {
            Assert.That(actual.Length, Is.EqualTo(expected.Length), $"{label} length");
            Assert.That(actual, Is.Not.Empty, $"{label} values");
            float largestError = 0f;
            double squaredError = 0d;
            for (int index = 0; index < actual.Length; index++)
            {
                float error = Math.Abs(actual[index] - expected[index]);
                largestError = Math.Max(largestError, error);
                squaredError += error * error;
            }
            double rootMeanSquareError = Math.Sqrt(squaredError / actual.Length);
            Assert.That(largestError, Is.LessThan(maximumTolerance), $"{label} maximum absolute error");
            Assert.That(rootMeanSquareError, Is.LessThan(rootMeanSquareTolerance),
                $"{label} root-mean-square error");
        }

        private static float[] Concat(float[] first, float[] second)
        {
            var result = new float[first.Length + second.Length];
            Array.Copy(first, result, first.Length);
            Array.Copy(second, 0, result, first.Length, second.Length);
            return result;
        }

        private static void Dispose(IReadOnlyDictionary<string, OnnxTensor> tensors)
        {
            foreach (OnnxTensor tensor in tensors.Values) tensor.Dispose();
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
            Type type = target.GetType();
            FieldInfo field = null;
            while (type != null && field == null)
            {
                field = type.GetField(fieldName, BindingFlags.Instance | BindingFlags.NonPublic);
                type = type.BaseType;
            }
            if (field == null) throw new MissingFieldException(target.GetType().FullName, fieldName);
            field.SetValue(target, value);
        }
    }
}
#endif

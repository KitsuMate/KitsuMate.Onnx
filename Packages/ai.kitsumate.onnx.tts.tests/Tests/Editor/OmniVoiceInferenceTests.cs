#if UNITY_EDITOR
using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Threading.Tasks;
using KitsuMate.Onnx.Tts.OmniVoice;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using UnityEditor;
using UnityEngine;
using Debug = UnityEngine.Debug;

namespace KitsuMate.Onnx.Tts.Tests
{
    public sealed class OmniVoiceInferenceTests
    {
        private const string CompactFixtureName = "omnivoice-int4-b128-sym";
        private const string PortableFixtureName = "omnivoice-portable";
        private const string PortableCpuFixtureEnvironment = "KITSUMATE_OMNIVOICE_CPU_FIXTURE";
        private const string SettingsAssetPath = "Assets/Resources/OnnxSettings.asset";
        private static OnnxSettings configuredSettings;
        private static string previousModelRoot;
        private static bool createdSettings;

        [Test]
        [Category("Integration")]
        [Timeout(1200000)]
        public async Task CpuCompact_AutoDesignAndClone_ProduceReviewAudio()
        {
            ConfigureModelRoot();
            string root = Path.Combine(ModelRoot(), CompactFixtureName);
            RequireMergedFixture(root, "cpu-merged-int4");
            var tokenizer = new TextAsset(File.ReadAllText(Path.Combine(root, "tokenizer.json")));
            var set = ScriptableObject.CreateInstance<OmniVoiceModelSet>();
            var engine = ScriptableObject.CreateInstance<OmniVoiceEngine>();
            var backend = ScriptableObject.CreateInstance<OnnxRuntimeBackend>();
            AudioClip voice = LoadWav(Path.Combine(PackageFixtureRoot(), "example.wav"));
            InferenceEngineRuntime<TtsRequest, TtsResult> runtime = null;
            try
            {
                backend.SetProviderOrder(OnnxExecutionProvider.Cpu);
                ConfigureMergedSet(set, tokenizer, CompactFixtureName, "cpu-merged-int4", "CPU compact INT4");
                SetField(engine, "modelSet", set);
                SetField(engine, "generation", new OmniVoiceGenerationConfig
                {
                    Steps = 32, GuidanceScale = 2f, TimeShift = 0.1f,
                    LayerPenaltyFactor = 5f, PositionTemperature = 0f, ClassTemperature = 0f, Seed = 7
                });
                var load = Stopwatch.StartNew();
                runtime = await engine.CreateRuntimeAsync(backend);
                load.Stop();

                Assert.ThrowsAsync<ArgumentException>(async () => await runtime.RunAsync(new TtsRequest("clone")
                    { VoiceReference = voice }));
                using (var cancelled = new System.Threading.CancellationTokenSource())
                {
                    cancelled.Cancel();
                    Assert.CatchAsync<OperationCanceledException>(async () =>
                        await runtime.RunAsync(new TtsRequest("cancelled"), cancelled.Token));
                }

                TtsResult auto = await Generate(runtime, new TtsRequest("The bass player finished the song.")
                    { LanguageId = "en", DurationSeconds = 1.8f }, "cpu-int4-auto.wav");
                TtsResult design = await Generate(runtime, new TtsRequest(
                    "The [B EY1 S] player finished the song [laughter].")
                    {
                        LanguageId = "en", VoiceInstruction = "female, young adult, high pitch, british accent",
                        DurationSeconds = 1.8f
                    }, "cpu-int4-instruct-tags.wav");
                string transcript = JObject.Parse(File.ReadAllText(Path.Combine(PackageFixtureRoot(), "example.json")))
                    .Value<string>("transcript");
                TtsResult clone = await Generate(runtime, new TtsRequest("This is the example voice.")
                    {
                        LanguageId = "en", VoiceReference = voice, VoiceReferenceText = transcript,
                        DurationSeconds = 1.4f
                    }, "cpu-int4-clone.wav");

                AssertAudio(auto); AssertAudio(design); AssertAudio(clone);
                Assert.That(runtime is OmniVoiceEngineRuntime omni &&
                    omni.PrimarySessionDiagnostics.InitializedPrimaryProvider == OnnxExecutionProvider.Cpu, Is.True);
                Debug.Log($"OMNIVOICE_CPU load_ms={load.ElapsedMilliseconds}");
            }
            finally
            {
                runtime?.Dispose();
                UnityEngine.Object.DestroyImmediate(voice);
                UnityEngine.Object.DestroyImmediate(tokenizer);
                UnityEngine.Object.DestroyImmediate(backend);
                UnityEngine.Object.DestroyImmediate(engine);
                UnityEngine.Object.DestroyImmediate(set);
                RestoreModelRoot();
            }
        }

        [Test]
        [Category("Integration")]
        [Timeout(1200000)]
        public async Task PortableFp32_Cpu_InstructAndTags_ProducesReviewAudio()
        {
            ConfigureModelRoot();
            string fixtureName = Environment.GetEnvironmentVariable(PortableCpuFixtureEnvironment);
            if (string.IsNullOrWhiteSpace(fixtureName)) fixtureName = PortableFixtureName;
            string root = Path.Combine(ModelRoot(), fixtureName);
            RequireMergedFixture(root, "portable-merged-fp32");
            var tokenizer = new TextAsset(File.ReadAllText(Path.Combine(root, "tokenizer.json")));
            var set = ScriptableObject.CreateInstance<OmniVoiceModelSet>();
            var engine = ScriptableObject.CreateInstance<OmniVoiceEngine>();
            var backend = ScriptableObject.CreateInstance<OnnxRuntimeBackend>();
            InferenceEngineRuntime<TtsRequest, TtsResult> runtime = null;
            try
            {
                backend.SetProviderOrder(OnnxExecutionProvider.Cpu);
                ConfigureMergedSet(set, tokenizer, fixtureName, "portable-merged-fp32", "Portable FP32");
                SetField(engine, "modelSet", set);
                SetField(engine, "generation", new OmniVoiceGenerationConfig
                {
                    Steps = 32, GuidanceScale = 2f, TimeShift = 0.1f,
                    LayerPenaltyFactor = 5f, PositionTemperature = 0f, ClassTemperature = 0f, Seed = 7
                });
                runtime = await engine.CreateRuntimeAsync(backend);
                TtsResult auto = await Generate(runtime, new TtsRequest("The bass player finished the song.")
                {
                    LanguageId = "en",
                    DurationSeconds = 1.8f
                }, fixtureName + "-auto.wav");
                TtsResult result = await Generate(runtime, new TtsRequest(
                    "The [B EY1 S] player finished the song [laughter].")
                    {
                        LanguageId = "en",
                        VoiceInstruction = "female, young adult, high pitch, british accent",
                        DurationSeconds = 1.8f
                    }, fixtureName + "-instruct-tags.wav");
                AssertAudio(auto);
                AssertAudio(result);
                Assert.That(((OmniVoiceEngineRuntime)runtime).SessionDiagnostics.All(diagnostics =>
                    diagnostics.InitializedPrimaryProvider == OnnxExecutionProvider.Cpu), Is.True);
            }
            finally
            {
                runtime?.Dispose();
                UnityEngine.Object.DestroyImmediate(tokenizer);
                UnityEngine.Object.DestroyImmediate(backend);
                UnityEngine.Object.DestroyImmediate(engine);
                UnityEngine.Object.DestroyImmediate(set);
                RestoreModelRoot();
            }
        }

        [Test]
        [Category("ManualAccelerator")]
        [Timeout(1200000)]
        public async Task PortableFp32_Cuda_InstructAndTags_ProducesReviewAudioWithoutFallback()
        {
            if (!string.Equals(Environment.GetEnvironmentVariable("KITSUMATE_OMNIVOICE_GPU_TEST"), "1",
                    StringComparison.Ordinal))
                Assert.Ignore("Set KITSUMATE_OMNIVOICE_GPU_TEST=1 to run the manual CUDA smoke test.");

            ConfigureModelRoot();
            string root = Path.Combine(ModelRoot(), PortableFixtureName);
            RequireMergedFixture(root, "portable-merged-fp32");
            var tokenizer = new TextAsset(File.ReadAllText(Path.Combine(root, "tokenizer.json")));
            var set = ScriptableObject.CreateInstance<OmniVoiceModelSet>();
            var engine = ScriptableObject.CreateInstance<OmniVoiceEngine>();
            var backend = ScriptableObject.CreateInstance<OnnxRuntimeBackend>();
            InferenceEngineRuntime<TtsRequest, TtsResult> runtime = null;
            try
            {
                backend.SetProviderOrder(OnnxExecutionProvider.Cuda);
                ConfigureMergedSet(set, tokenizer, PortableFixtureName, "portable-merged-fp32", "Portable FP32");
                SetField(engine, "modelSet", set);
                SetField(engine, "generation", new OmniVoiceGenerationConfig
                {
                    Steps = 32, GuidanceScale = 2f, TimeShift = 0.1f,
                    LayerPenaltyFactor = 5f, PositionTemperature = 0f, ClassTemperature = 0f, Seed = 7
                });
                var load = Stopwatch.StartNew();
                runtime = await engine.CreateRuntimeAsync(backend);
                load.Stop();
                var omni = (OmniVoiceEngineRuntime)runtime;
                Assert.That(omni.SessionDiagnostics, Is.Not.Empty);
                Assert.That(omni.SessionDiagnostics.All(diagnostics =>
                    diagnostics.InitializedPrimaryProvider == OnnxExecutionProvider.Cuda), Is.True);
                Assert.That(omni.SessionDiagnostics.All(diagnostics =>
                    string.IsNullOrEmpty(diagnostics.InitializationFallbackReason)), Is.True);

                var request = new TtsRequest("The [B EY1 S] player finished the song [laughter].")
                {
                    LanguageId = "en",
                    VoiceInstruction = "female, young adult, high pitch, british accent",
                    DurationSeconds = 1.8f
                };
                var timer = Stopwatch.StartNew();
                Task<TtsResult> generation = runtime.RunAsync(request);
                int peakGpuMemoryMb = 0, peakGpuUtilization = 0;
                while (!generation.IsCompleted)
                {
                    SampleNvidiaSmi(ref peakGpuMemoryMb, ref peakGpuUtilization);
                    await Task.WhenAny(generation, Task.Delay(250));
                }
                TtsResult result = await generation;
                timer.Stop();
                AssertAudio(result);
                string output = Path.Combine(OutputRoot(), "gpu-instruct-tags.wav");
                Directory.CreateDirectory(Path.GetDirectoryName(output));
                SaveWav(output, result.Samples, result.SampleRate);
                Assert.That(peakGpuUtilization, Is.GreaterThan(0), "nvidia-smi did not observe GPU activity.");
                Debug.Log($"OMNIVOICE_GPU provider=Cuda load_ms={load.ElapsedMilliseconds} " +
                    $"inference_ms={timer.ElapsedMilliseconds} duration={result.Duration:F3} " +
                    $"peak_memory_mb={peakGpuMemoryMb} peak_utilization={peakGpuUtilization}");
            }
            finally
            {
                runtime?.Dispose();
                UnityEngine.Object.DestroyImmediate(tokenizer);
                UnityEngine.Object.DestroyImmediate(backend);
                UnityEngine.Object.DestroyImmediate(engine);
                UnityEngine.Object.DestroyImmediate(set);
                RestoreModelRoot();
            }
        }

        private static async Task<TtsResult> Generate(InferenceEngineRuntime<TtsRequest, TtsResult> runtime,
            TtsRequest request, string outputName)
        {
            var timer = Stopwatch.StartNew();
            TtsResult result = await runtime.RunAsync(request);
            timer.Stop();
            string output = Path.Combine(OutputRoot(), outputName);
            Directory.CreateDirectory(Path.GetDirectoryName(output));
            SaveWav(output, result.Samples, result.SampleRate);
            Debug.Log($"OMNIVOICE_CPU file={outputName} inference_ms={timer.ElapsedMilliseconds} duration={result.Duration:F3}");
            return result;
        }

        private static void ConfigureMergedSet(OmniVoiceModelSet set, TextAsset tokenizer, string fixtureName,
            string backboneDirectory, string profile)
        {
            string root = fixtureName + "/onnx/";
            set.MergedBackbone.ConfigureFile(root + backboneDirectory + "/omnivoice.onnx", string.Empty, null, null);
            set.AcousticEncoder.ConfigureFile(root + "codec-fp32/acoustic_encoder.onnx", string.Empty, null, null);
            set.SemanticEncoder.ConfigureFile(root + "codec-fp32/semantic_encoder.onnx", string.Empty, null, null);
            set.QuantizerEncoder.ConfigureFile(root + "codec-fp32/quantizer_encoder.onnx", string.Empty, null, null);
            set.HiggsDecoder.ConfigureFile(root + "codec-fp32/higgs_decoder.onnx", string.Empty, null, null);
            set.Configure(OmniVoiceBackboneTopology.Merged, OmniVoiceTensorPrecision.Float32, profile, tokenizer);
        }

        private static void AssertAudio(TtsResult result)
        {
            Assert.That(result, Is.Not.Null);
            Assert.That(result.SampleRate, Is.EqualTo(OmniVoiceConstants.SampleRate));
            Assert.That(result.Samples, Is.Not.Null.And.Not.Empty);
            Assert.That(result.Samples.All(float.IsFinite), Is.True);
            Assert.That(result.Duration, Is.InRange(0.5f, 10f));
            Assert.That(result.Samples.Max(Math.Abs), Is.LessThanOrEqualTo(1f));
            Assert.That(Math.Sqrt(result.Samples.Select(value => value * value).Average()), Is.GreaterThan(1e-5));
            int longestSilence = 0, currentSilence = 0;
            foreach (float sample in result.Samples)
            {
                if (Math.Abs(sample) < 1e-4f) longestSilence = Math.Max(longestSilence, ++currentSilence);
                else currentSilence = 0;
            }
            Assert.That(longestSilence / (double)result.Samples.Length, Is.LessThan(0.65),
                "Generated audio contains an extended unintended silent region.");
        }

        private static void RequireMergedFixture(string root, string backboneDirectory)
        {
            string backbone = $"onnx/{backboneDirectory}/omnivoice.onnx";
            Assert.That(File.Exists(Path.Combine(root, backbone)), Is.True,
                $"Missing OmniVoice fixture: {backbone}");
            Assert.That(File.Exists(Path.Combine(root, backbone + ".data")) ||
                File.Exists(Path.Combine(root, backbone + "_data")), Is.True,
                $"Missing OmniVoice external data for: {backbone}");
            foreach (string path in new[]
            {
                "tokenizer.json", "onnx/codec-fp32/acoustic_encoder.onnx",
                "onnx/codec-fp32/semantic_encoder.onnx", "onnx/codec-fp32/quantizer_encoder.onnx",
                "onnx/codec-fp32/higgs_decoder.onnx"
            }) Assert.That(File.Exists(Path.Combine(root, path)), Is.True, $"Missing OmniVoice fixture: {path}");
        }

        private static void SampleNvidiaSmi(ref int peakMemoryMb, ref int peakUtilization)
        {
            using var process = new Process
            {
                StartInfo = new ProcessStartInfo
                {
                    FileName = "nvidia-smi.exe",
                    Arguments = "--query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits",
                    UseShellExecute = false,
                    CreateNoWindow = true,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true
                }
            };
            process.Start();
            string line = process.StandardOutput.ReadLine();
            process.WaitForExit(2000);
            if (string.IsNullOrWhiteSpace(line)) return;
            string[] values = line.Split(',');
            if (values.Length < 2) return;
            if (int.TryParse(values[0].Trim(), out int memory)) peakMemoryMb = Math.Max(peakMemoryMb, memory);
            if (int.TryParse(values[1].Trim(), out int utilization)) peakUtilization = Math.Max(peakUtilization, utilization);
        }

        private static AudioClip LoadWav(string path)
        {
            using var reader = new BinaryReader(File.OpenRead(path));
            if (new string(reader.ReadChars(4)) != "RIFF") throw new InvalidDataException("Expected RIFF WAV.");
            reader.ReadInt32();
            if (new string(reader.ReadChars(4)) != "WAVE") throw new InvalidDataException("Expected WAVE.");
            ushort format = 0, channels = 0, bits = 0; int rate = 0; byte[] data = null;
            while (reader.BaseStream.Position + 8 <= reader.BaseStream.Length)
            {
                string chunk = new string(reader.ReadChars(4)); int length = reader.ReadInt32();
                if (chunk == "fmt ") { format = reader.ReadUInt16(); channels = reader.ReadUInt16(); rate = reader.ReadInt32(); reader.BaseStream.Position += 6; bits = reader.ReadUInt16(); reader.BaseStream.Position += length - 16; }
                else if (chunk == "data") data = reader.ReadBytes(length);
                else reader.BaseStream.Position += length;
                if ((length & 1) != 0) reader.BaseStream.Position++;
            }
            if (format != 1 || bits != 16 || channels == 0 || data == null) throw new InvalidDataException("Expected PCM16 WAV.");
            var samples = new float[data.Length / 2];
            for (int i = 0; i < samples.Length; i++) samples[i] = BitConverter.ToInt16(data, i * 2) / 32768f;
            AudioClip clip = AudioClip.Create("OmniVoiceExample", samples.Length / channels, channels, rate, false);
            clip.SetData(samples, 0); return clip;
        }

        private static void SaveWav(string path, float[] samples, int sampleRate)
        {
            using var writer = new BinaryWriter(File.Create(path));
            writer.Write(System.Text.Encoding.ASCII.GetBytes("RIFF")); writer.Write(36 + samples.Length * 2);
            writer.Write(System.Text.Encoding.ASCII.GetBytes("WAVEfmt ")); writer.Write(16); writer.Write((ushort)1);
            writer.Write((ushort)1); writer.Write(sampleRate); writer.Write(sampleRate * 2); writer.Write((ushort)2);
            writer.Write((ushort)16); writer.Write(System.Text.Encoding.ASCII.GetBytes("data")); writer.Write(samples.Length * 2);
            foreach (float sample in samples) writer.Write((short)Mathf.RoundToInt(Mathf.Clamp(sample, -1f, 1f) * 32767f));
        }

        private static void ConfigureModelRoot()
        {
            configuredSettings = AssetDatabase.LoadAssetAtPath<OnnxSettings>(SettingsAssetPath);
            createdSettings = configuredSettings == null;
            if (createdSettings)
            {
                Directory.CreateDirectory(Path.GetDirectoryName(SettingsAssetPath));
                configuredSettings = ScriptableObject.CreateInstance<OnnxSettings>();
                AssetDatabase.CreateAsset(configuredSettings, SettingsAssetPath);
            }
            var serialized = new SerializedObject(configuredSettings);
            previousModelRoot = serialized.FindProperty("_modelStorageRoot").stringValue;
            serialized.FindProperty("_modelStorageRoot").stringValue = FixtureStorageRoot();
            serialized.ApplyModifiedPropertiesWithoutUndo();
            AssetDatabase.SaveAssets();
        }

        private static void RestoreModelRoot()
        {
            if (configuredSettings == null) return;
            if (createdSettings) AssetDatabase.DeleteAsset(SettingsAssetPath);
            else
            {
                var serialized = new SerializedObject(configuredSettings);
                serialized.FindProperty("_modelStorageRoot").stringValue = previousModelRoot;
                serialized.ApplyModifiedPropertiesWithoutUndo();
                AssetDatabase.SaveAssets();
            }
            configuredSettings = null;
            previousModelRoot = null;
            createdSettings = false;
        }

        private static string FixtureStorageRoot() => Directory.Exists(Path.GetFullPath(
            Path.Combine(Application.dataPath, "../../../..", "KitsuMateOnnxFixtures")))
            ? "../../../KitsuMateOnnxFixtures"
            : "KitsuMateOnnxFixtures";
        private static string ModelRoot() => Path.GetFullPath(Path.Combine(Application.dataPath, "..", FixtureStorageRoot()));
        private static string PackageFixtureRoot()
        {
            string embedded = Path.GetFullPath(Path.Combine(Application.dataPath, "../..",
                "Packages/ai.kitsumate.onnx.tts.tests/Tests/Fixtures/OmniVoice"));
            return Directory.Exists(embedded) ? embedded : Path.GetFullPath(Path.Combine(Application.dataPath, "..",
                "External/KitsuMate.Onnx/Packages/ai.kitsumate.onnx.tts.tests/Tests/Fixtures/OmniVoice"));
        }
        private static string OutputRoot() => Path.GetFullPath(Path.Combine(Application.dataPath, "..", "TestResults/OmniVoice"));

        private static void SetField(object target, string fieldName, object value)
        {
            FieldInfo field = target.GetType().GetField(fieldName, BindingFlags.Instance | BindingFlags.NonPublic);
            if (field == null) throw new MissingFieldException(target.GetType().FullName, fieldName);
            field.SetValue(target, value);
        }
    }
}
#endif

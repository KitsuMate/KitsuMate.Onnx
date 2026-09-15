using System;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Threading;
using System.Threading.Tasks;
using KitsuMate.Onnx.Download;
using KitsuMate.Onnx.Tts.NeuTts;
using KitsuMate.Tokenizers;
using NUnit.Framework;
using UnityEngine;

namespace KitsuMate.Onnx.Tts.Tests
{
    public sealed class NeuTtsTests
    {
        [TestCase("“It’s １２ o’clock.”", "\"It's 12 o'clock.\"")]
        [TestCase("Cafe\u0301", "Café")]
        public void NormalizesLikeUpstream(string input, string expected) => Assert.AreEqual(expected, NeuTtsPromptBuilder.Normalize(input));

        [Test]
        public void SamplingSuppressesEndToken()
        {
            var config = new NeuTtsGenerationConfig { TopK = 1 };
            Assert.AreEqual(1, NeuTtsEngineRuntime.Sample(new[] { 10f, 2f, 1f }, config, new System.Random(42), 0));
        }

        [TestCase(1)]
        [TestCase(50)]
        [TestCase(257)]
        [TestCase(1000)]
        public void CandidateSelectionMatchesStableFullSort(int topK)
        {
            var random = new System.Random(42);
            // Ties must retain token order to preserve seeded sampling.
            var logits = Enumerable.Range(0, 700).Select(_ => (float)random.Next(-20, 20)).ToArray();
            foreach (int suppressed in new[] { -1, 0, 350, 699 })
            {
                var expected = Enumerable.Range(0, logits.Length).Where(i => i != suppressed)
                    .OrderByDescending(i => logits[i]).Take(topK).ToArray();
                CollectionAssert.AreEqual(expected, NeuTtsEngineRuntime.TopCandidates(logits, topK, suppressed));
            }
        }

        [Test]
        public void InvalidSamplingIsRejected()
        {
            Assert.Throws<ArgumentException>(() => new NeuTtsGenerationConfig { Temperature = float.NaN }.Validate());
            Assert.Throws<ArgumentException>(() => new NeuTtsGenerationConfig { TopK = 0 }.Validate());
            Assert.Throws<ArgumentException>(() => new NeuTtsGenerationConfig { TopP = 0 }.Validate());
            Assert.Throws<ArgumentException>(() => new NeuTtsGenerationConfig { MinP = float.NaN }.Validate());
        }

        [TestCase(0.1f, 0f)]
        [TestCase(1f, 0.9f)]
        public void ProbabilityFiltersKeepTheLeadingToken(float topP, float minP)
        {
            var config = new NeuTtsGenerationConfig { TopK = 3, TopP = topP, MinP = minP };
            for (int seed = 0; seed < 10; seed++)
                Assert.AreEqual(0, NeuTtsEngineRuntime.Sample(new[] { 3f, 2f, 1f }, config, new System.Random(seed), -1));
        }

        [Test]
        public void MissingModelsAreRejected()
        {
            var set = ScriptableObject.CreateInstance<NeuTtsModelSet>();
            try { Assert.IsFalse(set.IsComplete); Assert.IsFalse(set.Validate(default).IsValid); }
            finally { UnityEngine.Object.DestroyImmediate(set); }
        }

        [Serializable] private sealed class Fixtures { public Case[] cases; }
        [Serializable] private sealed class Case { public string text, speaker, emotion; public long[] ids; }
        private static string Artifacts => Environment.GetEnvironmentVariable("KITSUMATE_NEUTTS_ARTIFACTS") ??
            Path.GetFullPath(Path.Combine(Application.dataPath, "../.benchmark-onnx/neutts/artifacts"));

        [Test]
        public void PromptsMatchPinnedUpstream()
        {
            if (!File.Exists(Path.Combine(Artifacts, "prompt-fixtures.json"))) Assert.Ignore("Prepare NeuTTS artifacts with reference_parity.py.");
            var metadata = JsonUtility.FromJson<NeuTtsMetadata>(File.ReadAllText(Path.Combine(Artifacts, "neutts.json")));
            metadata.Validate();
            var tokenizer = Tokenizer.FromTokenizerJson(File.ReadAllBytes(Path.Combine(Artifacts, "tokenizer.json")));
            var fixtures = JsonUtility.FromJson<Fixtures>(File.ReadAllText(Path.Combine(Artifacts, "prompt-fixtures.json")));
            Assert.GreaterOrEqual(fixtures.cases.Length, 28);
            foreach (var fixture in fixtures.cases)
                CollectionAssert.AreEqual(fixture.ids, NeuTtsPromptBuilder.Build(tokenizer, metadata, fixture.text,
                    new NeuTtsGenerationConfig { Speaker = fixture.speaker, Emotion = fixture.emotion }), fixture.speaker + "/" + fixture.emotion);
            Assert.Throws<ArgumentException>(() => NeuTtsPromptBuilder.Build(tokenizer, metadata, "Hello",
                new NeuTtsGenerationConfig { Speaker = "unknown" }));
        }

        [TestCase("fp32", OnnxExecutionProvider.Cpu)]
        [TestCase("fp16", OnnxExecutionProvider.Cpu)]
        [TestCase("int8", OnnxExecutionProvider.Cpu)]
        [TestCase("int4", OnnxExecutionProvider.Cpu)]
        [TestCase("fp32", OnnxExecutionProvider.WebGpu)]
        [Category("NeuTtsIntegration")]
        public async Task SynthesizesAndCancels(string profile, OnnxExecutionProvider provider)
        {
            if (!File.Exists(Path.Combine(Artifacts, "onnx/codec_decoder.onnx"))) Assert.Ignore("Prepare NeuTTS artifacts first.");
            var backendType = Type.GetType("KitsuMate.Onnx.OnnxRuntimeBackend, KitsuMate.Onnx.Backend.OnnxRuntime");
            Assert.IsNotNull(backendType);
            var backend = (OnnxBackend)ScriptableObject.CreateInstance(backendType);
            backendType.GetMethod("SetProviderOrder").Invoke(backend, new object[] { new[] { provider } });
            var set = ScriptableObject.CreateInstance<NeuTtsModelSet>();
            var engine = ScriptableObject.CreateInstance<NeuTtsEngine>();
            var tokenizer = new TextAsset(File.ReadAllText(Path.Combine(Artifacts, "tokenizer.json")));
            var metadata = new TextAsset(File.ReadAllText(Path.Combine(Artifacts, "neutts.json")));
            void Field(object target, string name, object value) => target.GetType().GetField(name, BindingFlags.NonPublic | BindingFlags.Instance).SetValue(target, value);
            Field(set, "tokenizer", (TextFileReference)tokenizer); Field(set, "metadata", (TextFileReference)metadata); Field(engine, "modelSet", set);
            void Configure(OnnxModelReference model, string path)
            {
                var schema = OnnxLightweightMetadataReader.Read(path);
                model.ConfigureFile(path, "", schema.Inputs, schema.Outputs);
            }
            Configure(set.Backbone, Path.Combine(Artifacts, $"onnx/backbone_{profile}.onnx"));
            Configure(set.CodecDecoder, Path.Combine(Artifacts, "onnx/codec_decoder.onnx"));
            InferenceEngineRuntime<TtsRequest,TtsResult> runtime = null;
            try
            {
                runtime = await engine.CreateRuntimeAsync(backend);
                await Fails<ArgumentException>(() => runtime.RunAsync(new TtsRequest("")));
                await Fails<ArgumentException>(() => runtime.RunAsync(new TtsRequest("Hello") { MaxNewTokens = 0 }));
                await Fails<ArgumentException>(() => runtime.RunAsync(new TtsRequest("Hello") { VoiceInstruction = "a custom voice" }));
                await Fails<ArgumentException>(() => runtime.RunAsync(new TtsRequest("Hello") { NeuTts = new NeuTtsGenerationConfig { Emotion = "unknown" } }));
                await Fails<ArgumentException>(() => runtime.RunAsync(new TtsRequest(string.Concat(Enumerable.Repeat("Hello world. ", 800)))));
                var request = new TtsRequest("Hello, this is a local voice test.")
                    { MaxNewTokens = 350, NeuTts = new NeuTtsGenerationConfig { UseSeed = true, Seed = 42 } };
                var result = await runtime.RunAsync(request);
                Assert.AreEqual(24000, result.SampleRate);
                Assert.Greater(result.Samples.Length, 24000);
                foreach (float sample in result.Samples) Assert.IsTrue(float.IsFinite(sample));
                var diagnostics = ((NeuTtsEngineRuntime)runtime).SessionDiagnostics;
                Assert.AreEqual(provider, diagnostics[0].ActiveProvider);
                Assert.AreEqual(OnnxExecutionProvider.Cpu, diagnostics[1].ActiveProvider,
                    "NeuCodec must use CPU to avoid corrupted WebGPU audio.");
                using var cancellation = new CancellationTokenSource(); cancellation.Cancel();
                await Fails<OperationCanceledException>(() => runtime.RunAsync(request, cancellation.Token));
                var repeated = await runtime.RunAsync(request);
                CollectionAssert.AreEqual(result.Samples, repeated.Samples);
                string output = Path.Combine(Artifacts, $"unity-{profile}-{provider}.json");
                File.WriteAllText(output, JsonUtility.ToJson(new IntegrationResult { provider = provider.ToString(), sampleRate = result.SampleRate,
                    samples = result.Samples.Length, tokens = result.TokensGenerated, repeated = true, cancellation = true }, true));
            }
            finally
            {
                runtime?.Dispose();
                UnityEngine.Object.DestroyImmediate(engine); UnityEngine.Object.DestroyImmediate(set);
                UnityEngine.Object.DestroyImmediate(tokenizer); UnityEngine.Object.DestroyImmediate(metadata);
                UnityEngine.Object.DestroyImmediate(backend);
            }
        }
        private static async Task Fails<T>(Func<Task<TtsResult>> action) where T : Exception
        {
            try { await action(); }
            catch (T) { return; }
            Assert.Fail("Expected " + typeof(T).Name);
        }
        [Serializable] private sealed class IntegrationResult { public string provider; public int sampleRate, samples, tokens; public bool repeated, cancellation; }
    }
}

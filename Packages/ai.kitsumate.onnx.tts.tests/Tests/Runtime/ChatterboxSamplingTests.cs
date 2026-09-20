using System;
using KitsuMate.Onnx.Tts.Chatterbox;
using NUnit.Framework;

namespace KitsuMate.Onnx.Tts.Tests
{
    public sealed class ChatterboxSamplingTests
    {
        [Test]
        public void V3PreservesCaseAndPolishNormalization()
        {
            var text = ChatterboxEngineRuntime.PrepareV3Text("  Zażółć   GĘŚLĄ jaźń  ");
            Assert.That(text, Is.EqualTo("Zażółć GĘŚLĄ jaźń."));
            Assert.That(new LanguagePreprocessor().ProcessV3(text, "pl"),
                Is.EqualTo(text.Normalize(System.Text.NormalizationForm.FormKD)));
            Assert.Throws<ArgumentException>(() => ChatterboxEngineRuntime.PrepareV3Text("  "));
        }

        [Test]
        public void GuidanceUsesLastPositionOfEachBatch()
        {
            var combined = ChatterboxSampling.Combine(new[] { 99f, 99f, 2f, 4f, 99f, 99f, 0f, 6f }, 2, 2, 0.5f);
            Assert.That(combined, Is.EqualTo(new[] { 3f, 3f }));
            Assert.That(ChatterboxSampling.Combine(new[] { 99f, 99f, 2f, 4f }, 2, 1, 0), Is.EqualTo(new[] { 2f, 4f }));
        }

        [Test]
        public void ZeroTemperatureIsGreedy()
        {
            Assert.That(ChatterboxSampling.Sample(new[] { -2f, 4f, 3f }, new ChatterboxGenerationConfig { Temperature = 0 }, new Random(1)), Is.EqualTo(1));
        }

        [Test]
        public void MinPRemovesLowProbabilityTokens()
        {
            var config = new ChatterboxGenerationConfig { Temperature = 1, MinP = 0.5f, TopP = 1 };
            var random = new Random(1);
            for (int i = 0; i < 100; i++)
                Assert.That(ChatterboxSampling.Sample(new[] { 0f, -1f, -2f }, config, random), Is.Zero);
        }

        [Test]
        public void TopPIsCalculatedAfterMinP()
        {
            // Weights 1, .6, .4: min-p removes .4, then top-p=.6 retains only 1.
            var config = new ChatterboxGenerationConfig { Temperature = 1, MinP = 0.5f, TopP = 0.6f };
            var random = new Random(1);
            for (int i = 0; i < 100; i++)
                Assert.That(ChatterboxSampling.Sample(new[] { 0f, (float)Math.Log(.6), (float)Math.Log(.4) }, config, random), Is.Zero);
        }

        [Test]
        public void InvalidSettingsAreRejected()
        {
            Assert.Throws<ArgumentException>(() => new ChatterboxGenerationConfig { Guidance = float.NaN }.Validate());
            Assert.Throws<ArgumentException>(() => new ChatterboxGenerationConfig { Temperature = -1 }.Validate());
            Assert.Throws<ArgumentException>(() => new ChatterboxGenerationConfig { TopP = 0 }.Validate());
            Assert.Throws<ArgumentException>(() => new ChatterboxGenerationConfig { MinP = 2 }.Validate());
        }

        [Test]
        public void SplitModelSetHasFiveFixedGraphRoles()
        {
            var split = UnityEngine.ScriptableObject.CreateInstance<ChatterboxSplitModelSet>();
            try
            {
                Assert.That(split.DownloadGraphRoles.Count, Is.EqualTo(5));
                Assert.That(split.DownloadGraphRoles[1].role, Is.EqualTo("embedding-language-model"));
                Assert.That(split.DownloadGraphRoles[4].fileStem, Is.EqualTo("vocoder"));
                Assert.That(split.GetAllModels().Length, Is.EqualTo(5));
            }
            finally { UnityEngine.Object.DestroyImmediate(split); }
        }

        [Test]
        public void MissingAssignedGraphDoesNotFallBackToInstallation()
        {
            var split = UnityEngine.ScriptableObject.CreateInstance<ChatterboxSplitModelSet>();
            try
            {
                split.Download.repository = "owner/repository";
                split.SpeechEncoder.ConfigureFile(System.IO.Path.Combine(System.IO.Path.GetTempPath(),
                    Guid.NewGuid().ToString("N") + ".onnx"), "", null, null);
                Assert.That(split.SpeechEncoder.IsAvailable, Is.False);
                Assert.That(split.UsesInstallation, Is.False);
            }
            finally { UnityEngine.Object.DestroyImmediate(split); }
        }

        [Test]
        public void SplitFlowScheduleUsesReferenceCosineSteps()
        {
            Assert.That(ChatterboxSplitEngineRuntime.FlowTime(0, 6), Is.EqualTo(0).Within(1e-6));
            Assert.That(ChatterboxSplitEngineRuntime.FlowTime(3, 6),
                Is.EqualTo(1 - Math.Cos(Math.PI / 4)).Within(1e-6));
            Assert.That(ChatterboxSplitEngineRuntime.FlowTime(6, 6), Is.EqualTo(1).Within(1e-6));
        }

        [Test]
        public void SplitFlowNoiseUsesTheConfiguredSeed()
        {
            var shape = new[] { 1, 80, 32 };
            float[] first = ChatterboxSplitEngineRuntime.GenerateFlowNoise(shape, 42);
            Assert.That(first, Is.EqualTo(ChatterboxSplitEngineRuntime.GenerateFlowNoise(shape, 42)));
            Assert.That(first, Is.Not.EqualTo(ChatterboxSplitEngineRuntime.GenerateFlowNoise(shape, 43)));
            Assert.That(first, Has.Length.EqualTo(2560));
            Assert.That(Array.TrueForAll(first, float.IsFinite), Is.True);
        }

        [Test]
        public void VoicePreparationDownmixesAndResamplesForBothPipelines()
        {
            var clip = UnityEngine.AudioClip.Create("stereo-12k", 2, 2, 12000, false);
            try
            {
                Assert.That(clip.SetData(new[] { 0f, 2f, 2f, 4f }, 0), Is.True);
                float[] samples = ChatterboxEngineRuntime.PrepareVoiceSamples(clip);
                Assert.That(samples, Has.Length.EqualTo(4));
                Assert.That(samples[0], Is.EqualTo(1f).Within(1e-6));
                Assert.That(samples[1], Is.EqualTo(2f).Within(1e-6));
                Assert.That(samples[2], Is.EqualTo(3f).Within(1e-6));
                Assert.That(samples[3], Is.EqualTo(3f).Within(1e-6));
            }
            finally { UnityEngine.Object.DestroyImmediate(clip); }
        }

        [Test]
        public void VoicePreparationAllowsSilenceAndRejectsAudioTooShortToResample()
        {
            var silence = UnityEngine.AudioClip.Create("silence", 4, 1, ChatterboxConstants.SampleRate, false);
            var tooShort = UnityEngine.AudioClip.Create("too-short", 1, 1, 48000, false);
            try
            {
                Assert.That(ChatterboxEngineRuntime.PrepareVoiceSamples(silence),
                    Is.EqualTo(new[] { 0f, 0f, 0f, 0f }));
                Assert.Throws<ArgumentException>(() => ChatterboxEngineRuntime.PrepareVoiceSamples(tooShort));
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(silence);
                UnityEngine.Object.DestroyImmediate(tooShort);
            }
        }

    }
}

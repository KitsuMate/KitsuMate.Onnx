using System;
using System.Linq;
using KitsuMate.Onnx.Tts.OmniVoice;
using NUnit.Framework;
using UnityEngine;

namespace KitsuMate.Onnx.Tts.Tests
{
    public sealed class OmniVoicePromptTests
    {
        [Test]
        public void StyleText_CloneAndInstruction_UsesOfficialEnvelope()
        {
            Assert.That(OmniVoicePromptBuilder.StyleText("en", "female, whisper", true, true), Is.EqualTo(
                "<|denoise|><|lang_start|>en<|lang_end|><|instruct_start|>female, whisper<|instruct_end|>"));
        }

        [Test]
        public void StyleText_MissingValues_UsesNoneSentinel()
        {
            Assert.That(OmniVoicePromptBuilder.StyleText(null, null, true, false), Is.EqualTo(
                "<|lang_start|>None<|lang_end|><|instruct_start|>None<|instruct_end|>"));
        }

        [Test]
        public void WrappedText_TagsAndPronunciation_PreservesControls()
        {
            string value = OmniVoicePromptBuilder.WrappedText(
                "The [B EY1 S] player laughed [laughter] at NI3.\r\n", "Reference sentence.");
            Assert.That(value, Is.EqualTo(
                "<|text_start|>Reference sentence. The [B EY1 S] player laughed [laughter] at NI3.<|text_end|>"));
        }

        [Test]
        public void Schedule_AllSteps_UnmasksEveryCodebookToken()
        {
            int[] schedule = OmniVoiceEngineRuntime.Schedule(41, 32, 0.1f);
            Assert.That(schedule.Length, Is.EqualTo(32));
            Assert.That(schedule.All(value => value >= 0), Is.True);
            Assert.That(schedule.Sum(), Is.EqualTo(41 * OmniVoiceConstants.Codebooks));
            Assert.That(schedule[^1], Is.GreaterThan(0));
        }

        [Test]
        public void GenerationConfig_InvalidValues_Throws()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => new OmniVoiceGenerationConfig { Steps = 0 }.Validate());
            Assert.Throws<ArgumentOutOfRangeException>(() => new OmniVoiceGenerationConfig { TimeShift = 0f }.Validate());
        }

        [Test]
        public void EstimateFrames_DurationOverridesSpeedAndSpeedChangesAutomaticDuration()
        {
            int fixedFrames = OmniVoiceEngineRuntime.EstimateFrames(new TtsRequest("one two three")
                { DurationSeconds = 2f, Speed = 8f });
            int normalFrames = OmniVoiceEngineRuntime.EstimateFrames(new TtsRequest("one two three four")
                { Speed = 1f });
            int fastFrames = OmniVoiceEngineRuntime.EstimateFrames(new TtsRequest("one two three four")
                { Speed = 2f });

            Assert.That(fixedFrames, Is.EqualTo(2 * OmniVoiceConstants.FrameRate));
            Assert.That(fastFrames, Is.LessThan(normalFrames));
        }

        [Test]
        public void Predict_WithSameSeed_IsDeterministic()
        {
            float[] logits = Enumerable.Repeat(0f, OmniVoiceConstants.AudioVocabularySize * 2).ToArray();
            var config = new OmniVoiceGenerationConfig
            {
                GuidanceScale = 1f,
                PositionTemperature = 1f,
                ClassTemperature = 1f
            };

            var first = OmniVoiceEngineRuntime.Predict(logits, 0, OmniVoiceConstants.AudioVocabularySize,
                config, new System.Random(42));
            var second = OmniVoiceEngineRuntime.Predict(logits, 0, OmniVoiceConstants.AudioVocabularySize,
                config, new System.Random(42));

            Assert.That(second, Is.EqualTo(first));
        }

        [Test]
        public void ResolveMode_SelectsAutoDesignAndValidatedClone()
        {
            Assert.That(OmniVoicePromptBuilder.ResolveMode(new TtsRequest("auto")),
                Is.EqualTo(OmniVoiceVoiceMode.Auto));
            Assert.That(OmniVoicePromptBuilder.ResolveMode(new TtsRequest("design")
                { VoiceInstruction = "warm voice" }), Is.EqualTo(OmniVoiceVoiceMode.Design));
            Assert.Throws<ArgumentException>(() => OmniVoicePromptBuilder.ResolveMode(new TtsRequest("clone")
                { VoiceReferenceText = "reference" }));

            AudioClip clip = AudioClip.Create("reference", 960, 1, 24000, false);
            try
            {
                Assert.Throws<ArgumentException>(() => OmniVoicePromptBuilder.ResolveMode(new TtsRequest("clone")
                    { VoiceReference = clip }));
                Assert.That(OmniVoicePromptBuilder.ResolveMode(new TtsRequest("clone")
                    { VoiceReference = clip, VoiceReferenceText = "reference" }),
                    Is.EqualTo(OmniVoiceVoiceMode.Clone));
            }
            finally { UnityEngine.Object.DestroyImmediate(clip); }
        }

        [Test]
        public void ModelSet_TopologyAndCodecProfile_AreIndependent()
        {
            var set = ScriptableObject.CreateInstance<OmniVoiceModelSet>();
            var tokenizer = new TextAsset("{}");
            try
            {
                set.Configure(OmniVoiceBackboneTopology.Split, OmniVoiceTensorPrecision.Float32,
                    "CPU compact", tokenizer);
                Assert.That(set.GetAllModels(), Has.Length.EqualTo(7));
                Assert.That(set.CodecPrecision, Is.EqualTo(OmniVoiceTensorPrecision.Float32));
                Assert.That(set.Profile, Is.EqualTo("CPU compact"));

                set.Configure(OmniVoiceBackboneTopology.Merged, OmniVoiceTensorPrecision.Float32,
                    "Portable FP32", tokenizer);
                Assert.That(set.GetAllModels(), Has.Length.EqualTo(5));
                Assert.That(set.Profile, Is.EqualTo("Portable FP32"));
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(tokenizer);
                UnityEngine.Object.DestroyImmediate(set);
            }
        }
    }
}

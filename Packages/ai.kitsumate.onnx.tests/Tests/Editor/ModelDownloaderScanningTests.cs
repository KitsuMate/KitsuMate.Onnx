using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using KitsuMate.Onnx.Editor.Download;
using KitsuMate.Onnx.Download;
using NUnit.Framework;

namespace KitsuMate.Onnx.Tests
{
    public sealed class ModelDownloaderScanningTests
    {
        [TestCase("onnx/model.onnx", "model", "default")]
        [TestCase("onnx/model_fp32.onnx", "model", "fp32")]
        [TestCase("onnx/model_fp16.onnx", "model", "fp16")]
        [TestCase("onnx/model_q4.onnx", "model", "q4")]
        [TestCase("onnx/model_custom-precision.onnx", "model", "custom-precision")]
        public void DetectsArtifactTypeFromStandardFilename(string path, string stem, string expected)
        {
            Assert.That(HuggingFaceModelRepository.ArtifactType(path, stem), Is.EqualTo(expected));
        }

        [TestCase("default", true)]
        [TestCase("fp32", true)]
        [TestCase("FP32", true)]
        [TestCase("fp16", false)]
        [TestCase("q4", false)]
        [TestCase("custom", false)]
        public void SentisEligibilityUsesPositiveFp32Allowlist(string type, bool expected)
        {
            Assert.That(HuggingFaceModelRepository.IsSentisArtifact(type), Is.EqualTo(expected));
        }

        [Test]
        public async Task FixedChatterboxContractsSelectOnlyTheirOwnGraphs()
        {
            var info = new HfModelInfo
            {
                sha = "pinned-revision",
                siblings = new[]
                {
                    "onnx/speech_encoder_slim.onnx", "onnx/embed_tokens.onnx",
                    "onnx/language_model.onnx", "onnx/conditional_decoder.onnx",
                    "onnx/embedding_language_model_last.onnx", "onnx/flow_prepare_slim.onnx",
                    "onnx/flow_step_slim.onnx", "onnx/vocoder_slim.onnx",
                    "tokenizer.json", "default_voice.wav"
                }.Select(path => new HfSibling { rfilename = path, size = 1 }).ToArray()
            };
            var four = new[]
            {
                new ModelGraphRole("speech-encoder", "speech_encoder"),
                new ModelGraphRole("embed-tokens", "embed_tokens"),
                new ModelGraphRole("language-model", "language_model"),
                new ModelGraphRole("conditional-decoder", "conditional_decoder")
            };
            var five = new[]
            {
                new ModelGraphRole("speech-encoder", "speech_encoder"),
                new ModelGraphRole("embedding-language-model", "embedding_language_model"),
                new ModelGraphRole("flow-prepare", "flow_prepare"),
                new ModelGraphRole("flow-step", "flow_step"),
                new ModelGraphRole("vocoder", "vocoder")
            };
            var standard = await HuggingFaceModelRepository.ScanSnapshotAsync(
                new ModelDownloadRequest("owner/chatterbox", expectedFamily: "chatterbox", graphRoles: four), info);
            var split = await HuggingFaceModelRepository.ScanSnapshotAsync(
                new ModelDownloadRequest("owner/chatterbox", expectedFamily: "chatterbox", graphRoles: five), info);
            Assert.That(standard.Artifacts.Keys, Is.EquivalentTo(four.Select(role => role.role)));
            Assert.That(split.Artifacts.Keys, Is.EquivalentTo(five.Select(role => role.role)));
            Assert.That(split.Artifacts["flow-step"].Single().Model.Path,
                Is.EqualTo("onnx/flow_step_slim.onnx"));
        }

        [Test]
        public void FixedContractRejectsMissingLayout()
        {
            var info = new HfModelInfo
            {
                sha = "pinned-revision",
                siblings = new[] { new HfSibling { rfilename = "onnx/speech_encoder.onnx", size = 1 } }
            };
            var roles = new[] { new ModelGraphRole("speech-encoder", "speech_encoder"),
                new ModelGraphRole("vocoder", "vocoder") };
            Assert.ThrowsAsync<System.IO.InvalidDataException>(async () =>
                await HuggingFaceModelRepository.ScanSnapshotAsync(
                    new ModelDownloadRequest("owner/chatterbox", graphRoles: roles), info));
        }

        [Test]
        public void ScanRejectsCaseCollidingPaths()
        {
            var info = new HfModelInfo
            {
                sha = "pinned-revision",
                siblings = new[]
                {
                    new HfSibling { rfilename = "onnx/model.onnx", size = 1 },
                    new HfSibling { rfilename = "onnx/Model.onnx", size = 1 }
                }
            };
            Assert.ThrowsAsync<System.IO.InvalidDataException>(async () =>
                await HuggingFaceModelRepository.ScanSnapshotAsync(new ModelDownloadRequest("owner/model"), info));
        }

        [Test]
        public void SelectsEachWhisperArtifactIndependently()
        {
            DiscoveredArtifact encoderDefault = Artifact("encoder", "default", "onnx/encoder_model.onnx");
            DiscoveredArtifact encoderFp16 = Artifact("encoder", "fp16", "onnx/encoder_model_fp16.onnx");
            DiscoveredArtifact decoderDefault = Artifact("decoder", "default", "onnx/decoder_model.onnx");
            DiscoveredArtifact decoderMergedQ4 =
                Artifact("decoder", "q4", "onnx/decoder_model_merged_q4.onnx", true);
            DiscoveredArtifact cachedDefault =
                Artifact("decoder-with-past", "default", "onnx/decoder_with_past_model.onnx");
            var repository = new DiscoveredRepository("owner", "whisper", "whisper", "revision",
                new Dictionary<string, DiscoveredArtifact[]>
                {
                    ["encoder"] = new[] { encoderDefault, encoderFp16 },
                    ["decoder"] = new[] { decoderDefault, decoderMergedQ4 },
                    ["decoder-with-past"] = new[] { cachedDefault }
                },
                Array.Empty<DiscoveredFile>(), new[] { "encoder", "decoder" });

            DiscoveredArtifact[] merged = HuggingFaceModelRepository.SelectArtifacts(repository,
                new Dictionary<string, string>
                {
                    ["encoder"] = encoderFp16.Model.Path,
                    ["decoder"] = decoderMergedQ4.Model.Path
                });

            Assert.That(merged.Select(artifact => artifact.Model.Path), Is.EquivalentTo(new[]
            {
                encoderFp16.Model.Path, decoderMergedQ4.Model.Path
            }));

            DiscoveredArtifact[] split = HuggingFaceModelRepository.SelectArtifacts(repository,
                new Dictionary<string, string>
                {
                    ["encoder"] = encoderDefault.Model.Path,
                    ["decoder"] = decoderDefault.Model.Path,
                    ["decoder-with-past"] = cachedDefault.Model.Path
                });
            Assert.That(split.Select(artifact => artifact.Model.Path), Does.Contain(cachedDefault.Model.Path));
        }

        [Test]
        public void IncludesSelectedMelWithMergedWhisperDecoder()
        {
            var encoder = Artifact("encoder", "int8", "onnx/encoder_model_int8.onnx");
            var decoder = Artifact("decoder", "int8", "onnx/decoder_model_merged_int8.onnx", true);
            var mel = Artifact("mel", "default", "onnx/mel.onnx");
            var cached = Artifact("decoder-with-past", "int8", "onnx/decoder_with_past_model_int8.onnx");
            var repository = new DiscoveredRepository("owner", "whisper", "whisper", "revision",
                new Dictionary<string, DiscoveredArtifact[]>
                {
                    ["encoder"] = new[] { encoder }, ["decoder"] = new[] { decoder },
                    ["mel"] = new[] { mel }, ["decoder-with-past"] = new[] { cached }
                }, Array.Empty<DiscoveredFile>(), new[] { "encoder", "decoder" });
            var selection = new Dictionary<string, string>
            {
                ["encoder"] = encoder.Model.Path, ["decoder"] = decoder.Model.Path,
                ["mel"] = mel.Model.Path, ["decoder-with-past"] = cached.Model.Path
            };
            var selected = HuggingFaceModelRepository.SelectArtifacts(repository, selection);
            Assert.That(selected.Select(artifact => artifact.Role), Is.EquivalentTo(new[] { "encoder", "decoder", "mel" }));
        }

        [Test]
        public void RejectsUnsafeRepositoryPaths()
        {
            Assert.Throws<System.IO.InvalidDataException>(() =>
                HuggingFaceModelRepository.SafeRelativePath("../model.onnx"));
            Assert.Throws<System.IO.InvalidDataException>(() =>
                HuggingFaceModelRepository.SafeRelativePath("onnx//model.onnx"));
        }

        [Test]
        public void RemovedModelWideVariantApiIsAbsent()
        {
            Assert.That(typeof(ModelIdentity).GetField("Variant"), Is.Null);
            Assert.That(typeof(ModelDownloadRequest).GetProperty("Variant"), Is.Null);
            Assert.That(typeof(HuggingFaceModelRepository).GetMethod("GetVariantsAsync"), Is.Null);
        }

        private static DiscoveredArtifact Artifact(string role, string type, string path, bool merged = false)
        {
            return new DiscoveredArtifact(role, type, merged, new[]
            {
                new DiscoveredFile(role, path, new string('0', 64), 1)
            });
        }

        [Test]
        [Category("Integration")]
        public async Task ScansOnnxCommunityModelArtifacts()
        {
            var request = new ModelDownloadRequest(
                "onnx-community/all-MiniLM-L6-v2-ONNX",
                "aff7a1dc4e8a1ea593e6ea21e95c22ef0a25966f",
                "text-embedding");

            var discovery = await HuggingFaceModelRepository.GetArtifactsAsync(request);

            Assert.That(discovery.Revision, Is.EqualTo(request.Revision));
            Assert.That(discovery.Artifacts.Keys, Is.EqualTo(new[] { "model" }));
            Assert.That(discovery.Artifacts["model"], Does.Contain("onnx/model.onnx"));
            Assert.That(discovery.Artifacts["model"], Does.Contain("onnx/model_fp16.onnx"));
            Assert.That(discovery.Artifacts["model"], Does.Contain("onnx/model_q4.onnx"));
            Assert.That(discovery.Artifacts["model"], Does.Contain("onnx/model_q4f16.onnx"));
        }

        [Test]
        [Category("Integration")]
        public async Task ScansKitsuMateDefaultModelUsingStandardFilename()
        {
            const string revision = "d0c533e5999da1c893a0bba27d6336d423ba117d";
            var request = new ModelDownloadRequest(
                "KitsuMate/all-MiniLM-L6-v2-onnx", revision, "text-embedding");

            var discovery = await HuggingFaceModelRepository.GetArtifactsAsync(request);

            Assert.That(discovery.Revision, Is.EqualTo(revision));
            Assert.That(discovery.Artifacts["model"], Does.Contain("onnx/model.onnx"));
            Assert.That(discovery.Artifacts["model"], Does.Not.Contain("onnx/model_unity_fp32.onnx"));
        }

        [Test]
        [Category("Integration")]
        public async Task ScansOnnxCommunityWhisperArtifactsByRole()
        {
            var request = new ModelDownloadRequest(
                "onnx-community/whisper-tiny",
                "ff4177021cc41f7db950912b73ea4fdf7d01d8e7",
                "whisper");

            var discovery = await HuggingFaceModelRepository.GetArtifactsAsync(request);

            Assert.That(discovery.Artifacts.Keys,
                Is.SupersetOf(new[] { "encoder", "decoder", "decoder-with-past" }));
            Assert.That(discovery.Artifacts.Keys, Does.Not.Contain("mel"));
            Assert.That(discovery.Artifacts["encoder"], Does.Contain("onnx/encoder_model.onnx"));
            Assert.That(discovery.Artifacts["encoder"].Any(path => path.Contains("_fp16.")), Is.True);
            Assert.That(discovery.Artifacts["decoder"].Any(path => path.Contains("_merged_q4.")), Is.True);
            Assert.That(discovery.Artifacts["decoder-with-past"],
                Does.Contain("onnx/decoder_with_past_model.onnx"));
        }

        [Test]
        [Category("Integration")]
        public async Task ScansChatterboxTurboArtifactsIndependently()
        {
            var request = new ModelDownloadRequest(
                "KitsuMate/chatterbox-turbo-onnx",
                "7320dc3cfac446f5d689565d1f701fd1a0b8e516",
                "chatterbox");

            var discovery = await HuggingFaceModelRepository.GetArtifactsAsync(request);

            Assert.That(discovery.Artifacts.Keys, Is.EquivalentTo(new[]
            {
                "speech-encoder", "embed-tokens", "language-model", "conditional-decoder"
            }));
            Assert.That(discovery.Artifacts.Values.All(paths =>
                paths.Any(path => path.Contains("_q4f16."))), Is.True);
            Assert.That(discovery.Artifacts.Values.All(paths =>
                paths.Any(path => path.Contains("_fp16."))), Is.True);
            Assert.That(discovery.Artifacts.Values.All(paths =>
                paths.Any(path => path.Contains("_q4."))), Is.True);
            Assert.That(discovery.Artifacts.Values.All(paths =>
                paths.Any(path => path.Contains("_quantized."))), Is.True);
        }

        [Test]
        [Category("Integration")]
        public async Task PreservesChatterboxNanoMixedArtifactTypes()
        {
            var request = new ModelDownloadRequest(
                "KitsuMate/chatterbox-nano-onnx",
                "ea0149ce2ea3a222cd83458bf54c3dfb793e781f",
                "chatterbox");

            var discovery = await HuggingFaceModelRepository.GetArtifactsAsync(request);

            Assert.That(discovery.Artifacts["embed-tokens"].Single(), Does.Contain("_fp16."));
            Assert.That(discovery.Artifacts["speech-encoder"].Single(), Does.Contain("_q4f16."));
            Assert.That(discovery.Artifacts["language-model"].Single(), Does.Contain("_q4f16."));
            Assert.That(discovery.Artifacts["conditional-decoder"].Single(), Does.Contain("_q4."));
        }

        [Test]
        [Category("Integration")]
        public async Task ScansOnnxCommunityOmniVoiceSplitProfile()
        {
            var request = new ModelDownloadRequest(
                "onnx-community/OmniVoice-Onnx",
                "a7be7c65cc118137683f49eff0f80fdf9d5b5dbf",
                "omnivoice");

            var discovery = await HuggingFaceModelRepository.GetArtifactsAsync(request);

            Assert.That(discovery.Artifacts.Keys, Is.SupersetOf(new[]
            {
                "audio-embeddings", "language-decoder", "audio-heads", "acoustic-encoder",
                "semantic-encoder", "quantizer-encoder", "higgs-decoder"
            }));
            Assert.That(discovery.Artifacts["audio-embeddings"], Does.Contain("int4/audio_embeddings_encoder.onnx"));
            Assert.That(discovery.Artifacts["acoustic-encoder"], Does.Contain("audio_tokenizer/acoustic_encoder.onnx"));
            Assert.That(discovery.Artifacts["acoustic-encoder"], Does.Contain("audio_tokenizer/fp16/acoustic_encoder.onnx"));
        }

        [Test]
        [Category("Integration")]
        public async Task RejectsIncompleteGluschenkoOmniVoiceRepository()
        {
            var request = new ModelDownloadRequest(
                "gluschenko/omnivoice-onnx",
                "4d4bb31790c2de902de0d19645fd13cf2d71a88b",
                "omnivoice");

            System.IO.InvalidDataException failure = null;
            try { await HuggingFaceModelRepository.GetArtifactsAsync(request); }
            catch (System.IO.InvalidDataException exception) { failure = exception; }
            Assert.That(failure, Is.Not.Null);
        }

        [Test]
        [Category("Integration")]
        public async Task ScansCanonicalOmniVoiceArtifacts()
        {
            var repository = await HuggingFaceModelRepository.GetArtifactsAsync(new ModelDownloadRequest(
                "KitsuMate/omnivoice-onnx", "45d20c87b64f35c4ac203c5bac7ad97a3e60ca95", "omnivoice"));
            Assert.That(repository.Artifacts["merged-backbone"].Count, Is.EqualTo(2));
            Assert.That(repository.Artifacts["acoustic-encoder"].Single(), Does.Contain("codec-fp32"));
        }
    }
}

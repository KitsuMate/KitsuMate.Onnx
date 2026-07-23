using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using KitsuMate.Onnx.Editor.Download;
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
            Assert.That(ModelDownloader.ArtifactType(path, stem), Is.EqualTo(expected));
        }

        [TestCase("default", true)]
        [TestCase("fp32", true)]
        [TestCase("FP32", true)]
        [TestCase("fp16", false)]
        [TestCase("q4", false)]
        [TestCase("custom", false)]
        public void SentisEligibilityUsesPositiveFp32Allowlist(string type, bool expected)
        {
            Assert.That(ModelDownloader.IsSentisArtifact(type), Is.EqualTo(expected));
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

            DiscoveredArtifact[] merged = ModelDownloader.SelectArtifacts(repository,
                new Dictionary<string, string>
                {
                    ["encoder"] = encoderFp16.Model.Path,
                    ["decoder"] = decoderMergedQ4.Model.Path
                });

            Assert.That(merged.Select(artifact => artifact.Model.Path), Is.EquivalentTo(new[]
            {
                encoderFp16.Model.Path, decoderMergedQ4.Model.Path
            }));

            DiscoveredArtifact[] split = ModelDownloader.SelectArtifacts(repository,
                new Dictionary<string, string>
                {
                    ["encoder"] = encoderDefault.Model.Path,
                    ["decoder"] = decoderDefault.Model.Path,
                    ["decoder-with-past"] = cachedDefault.Model.Path
                });
            Assert.That(split.Select(artifact => artifact.Model.Path), Does.Contain(cachedDefault.Model.Path));
        }

        [Test]
        public void RejectsUnsafeRepositoryPaths()
        {
            Assert.Throws<System.IO.InvalidDataException>(() =>
                ModelDownloader.SafeRelativePath("../model.onnx"));
            Assert.Throws<System.IO.InvalidDataException>(() =>
                ModelDownloader.SafeRelativePath("onnx//model.onnx"));
        }

        [Test]
        public void RemovedModelWideVariantApiIsAbsent()
        {
            Assert.That(typeof(ModelIdentity).GetField("Variant"), Is.Null);
            Assert.That(typeof(ModelDownloadRequest).GetProperty("Variant"), Is.Null);
            Assert.That(typeof(ModelDownloader).GetMethod("GetVariantsAsync"), Is.Null);
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

            var discovery = await ModelDownloader.GetArtifactsAsync(request);

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
            const string revision = "9ec4eb6ad90ebff9e819a807468f37926836816f";
            var request = new ModelDownloadRequest(
                "KitsuMate/all-MiniLM-L6-v2-onnx", revision, "text-embedding");

            var discovery = await ModelDownloader.GetArtifactsAsync(request);

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

            var discovery = await ModelDownloader.GetArtifactsAsync(request);

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
                "89b9d3a64cd1ff1bfc2c90c13771f6550790a6aa",
                "chatterbox");

            var discovery = await ModelDownloader.GetArtifactsAsync(request);

            Assert.That(discovery.Artifacts.Keys, Is.EquivalentTo(new[]
            {
                "speech-encoder", "embed-tokens", "language-model", "conditional-decoder"
            }));
            Assert.That(discovery.Artifacts.Values.All(paths =>
                paths.Count == 1 && paths[0].Contains("_q4f16.")), Is.True);
        }

        [Test]
        [Category("Integration")]
        public async Task PreservesChatterboxNanoMixedArtifactTypes()
        {
            var request = new ModelDownloadRequest(
                "KitsuMate/chatterbox-nano-onnx",
                "b70ba9ceb90a146e93af372d805ec76baaa48b0b",
                "chatterbox");

            var discovery = await ModelDownloader.GetArtifactsAsync(request);

            Assert.That(discovery.Artifacts["embed-tokens"].Single(), Does.Contain("_fp16."));
            Assert.That(discovery.Artifacts["speech-encoder"].Single(), Does.Contain("_q4f16."));
            Assert.That(discovery.Artifacts["language-model"].Single(), Does.Contain("_q4f16."));
            Assert.That(discovery.Artifacts["conditional-decoder"].Single(), Does.Contain("_q4."));
        }
    }
}

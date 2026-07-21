using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using KitsuMate.Onnx.Asr.Editor;
using KitsuMate.Onnx.Asr.Whisper;
using KitsuMate.Onnx.Editor.Download;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;

namespace KitsuMate.Onnx.Asr.Tests
{
    public sealed class WhisperModelDownloaderTests
    {
        [Test]
        public void CompatibleVariant_RequiresAndResolvesMelAlongsideWhisperArtifacts()
        {
            var modelSet = ScriptableObject.CreateInstance<WhisperModelSet>();
            try
            {
                ModelDownloadRequest request = WhisperModelDownloader.CreateRequest(modelSet);
                ModelVariantDefinition variant = request.Variants[0];
                var files = new List<RepoFile>
                {
                    new("onnx/mel.onnx", "mel.onnx", 1_354_556),
                    new("onnx/encoder_model.onnx", "encoder_model.onnx", 17_593_091),
                    new("onnx/decoder_model_merged.onnx", "decoder_model_merged.onnx", 122_030_467),
                    new("tokenizer.json", "tokenizer.json", 1_035_584)
                };

                Assert.That(RepoFileClient.CanResolveAllFiles(files, variant.Files), Is.True);
                Dictionary<string, string> resolved = RepoFileClient.ResolveFiles(files, variant.Files);

                Assert.That(resolved["mel"], Is.EqualTo("onnx/mel.onnx"));
                Assert.That(resolved["encoder"], Is.EqualTo("onnx/encoder_model.onnx"));
                Assert.That(resolved["decoder"], Is.EqualTo("onnx/decoder_model_merged.onnx"));
                Assert.That(resolved["tokenizer"], Is.EqualTo("tokenizer.json"));
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(modelSet);
            }
        }

        [Test]
        public void CompatibleVariant_RejectsRepositoryWithoutMel()
        {
            var modelSet = ScriptableObject.CreateInstance<WhisperModelSet>();
            try
            {
                ModelVariantDefinition variant = WhisperModelDownloader.CreateRequest(modelSet).Variants[0];
                var files = new List<RepoFile>
                {
                    new("onnx/encoder_model.onnx", "encoder_model.onnx"),
                    new("onnx/decoder_model_merged.onnx", "decoder_model_merged.onnx"),
                    new("tokenizer.json", "tokenizer.json")
                };

                Assert.That(RepoFileClient.CanResolveAllFiles(files, variant.Files), Is.False);
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(modelSet);
            }
        }

        [UnityTest]
        public IEnumerator DownloadFile_DownloadsMelArtifact()
        {
            string directory = Path.Combine(Path.GetTempPath(), "KitsuMate.Onnx.WhisperDownloader", Guid.NewGuid().ToString("N"));
            string source = Path.Combine(directory, "mel.onnx");
            string destination = Path.Combine(directory, "installed", "mel.onnx");
            byte[] expected = { 0x08, 0x01, 0x12, 0x03, 0x6D, 0x65, 0x6C };
            bool completed = false;
            bool succeeded = false;
            string error = null;

            try
            {
                Directory.CreateDirectory(directory);
                File.WriteAllBytes(source, expected);

                yield return RepoFileClient.DownloadFile(
                    new Uri(source).AbsoluteUri,
                    destination,
                    token: null,
                    domain: null,
                    onProgress: null,
                    onComplete: (ok, message) => { succeeded = ok; error = message; completed = true; });

                Assert.That(completed, Is.True);
                Assert.That(succeeded, Is.True, error);
                CollectionAssert.AreEqual(expected, File.ReadAllBytes(destination));
            }
            finally
            {
                if (Directory.Exists(directory)) Directory.Delete(directory, recursive: true);
            }
        }
    }
}

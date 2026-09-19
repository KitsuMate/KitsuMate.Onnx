using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Threading.Tasks;
using KitsuMate.Onnx;
using KitsuMate.Onnx.Download;
using NUnit.Framework;

namespace KitsuMate.Onnx.Tts.Tests
{
    public sealed class ChatterboxDiscoveryTests
    {
        [Test]
        public async Task MixedRepositoryUsesSelectedGraphContract()
        {
            var files = new[]
            {
                "onnx/speech_encoder_slim.onnx", "onnx/embed_tokens.onnx",
                "onnx/language_model.onnx", "onnx/conditional_decoder.onnx",
                "onnx/embedding_language_model_last.onnx", "onnx/flow_prepare_slim.onnx",
                "onnx/flow_step_slim.onnx", "onnx/vocoder_slim.onnx",
                "tokenizer.json", "default_voice.wav"
            };
            var snapshot = new HfModelInfo { sha = "revision",
                siblings = files.Select(path => new HfSibling { rfilename = path, size = 1 }).ToArray() };
            var four = UnityEngine.ScriptableObject.CreateInstance<Chatterbox.ChatterboxModelSet>();
            var five = UnityEngine.ScriptableObject.CreateInstance<Chatterbox.ChatterboxSplitModelSet>();
            try
            {
                var standard = await HuggingFaceModelRepository.ScanSnapshotAsync(
                    new ModelDownloadRequest("owner/model", graphRoles: four.DownloadGraphRoles,
                        requiredCompanionRoles: four.DownloadRequiredCompanionRoles), snapshot);
                var split = await HuggingFaceModelRepository.ScanSnapshotAsync(
                    new ModelDownloadRequest("owner/model", graphRoles: five.DownloadGraphRoles,
                        requiredCompanionRoles: five.DownloadRequiredCompanionRoles), snapshot);
                Assert.That(standard.Artifacts.Keys, Is.EquivalentTo(four.DownloadGraphRoles.Select(role => role.role)));
                Assert.That(split.Artifacts.Keys, Is.EquivalentTo(five.DownloadGraphRoles.Select(role => role.role)));
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(four);
                UnityEngine.Object.DestroyImmediate(five);
            }
        }

        [Test]
        public async Task StagedV3FoldersGiveSplitSetItsOwnSpeechEncoder()
        {
            var paths = new[]
            {
                "onnx/speech_encoder.onnx", "onnx/embed_tokens.onnx", "onnx/language_model.onnx",
                "onnx/conditional_decoder_slim.onnx", "onnx/speech_encoder_slim.onnx",
                "onnx/embedding_language_model_last.onnx", "onnx/flow_prepare_slim.onnx",
                "onnx/flow_step_slim.onnx", "onnx/vocoder_slim.onnx",
                "tokenizer.json", "Cangjie5_TC.json"
            };
            var snapshot = new HfModelInfo { sha = "revision",
                siblings = paths.Select(path => new HfSibling { rfilename = path, size = 1 }).ToArray() };
            var four = UnityEngine.ScriptableObject.CreateInstance<Chatterbox.ChatterboxModelSet>();
            var split = UnityEngine.ScriptableObject.CreateInstance<Chatterbox.ChatterboxSplitModelSet>();
            try
            {
                var fourResult = await HuggingFaceModelRepository.ScanSnapshotAsync(
                    new ModelDownloadRequest("owner/model", graphRoles: four.DownloadGraphRoles,
                        requiredCompanionRoles: four.DownloadRequiredCompanionRoles), snapshot);
                var splitResult = await HuggingFaceModelRepository.ScanSnapshotAsync(
                    new ModelDownloadRequest("owner/model", graphRoles: split.DownloadGraphRoles,
                        requiredCompanionRoles: split.DownloadRequiredCompanionRoles), snapshot);
                Assert.That(fourResult.Artifacts["speech-encoder"][0].Model.Path,
                    Is.EqualTo("onnx/speech_encoder.onnx"));
                Assert.That(splitResult.Artifacts["speech-encoder"].Select(item => item.Model.Path),
                    Is.EquivalentTo(new[] { "onnx/speech_encoder_slim.onnx" }));
                Assert.That(splitResult.Artifacts.Keys, Is.EquivalentTo(split.DownloadGraphRoles.Select(role => role.role)));
                Assert.That(splitResult.CommonFiles.Select(file => file.Role), Contains.Item("cangjie"));
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(four);
                UnityEngine.Object.DestroyImmediate(split);
            }
        }

        [Test]
        public async Task ChatterboxRepositoriesWithoutBundledVoiceCanBeSelected()
        {
            var four = UnityEngine.ScriptableObject.CreateInstance<Chatterbox.ChatterboxModelSet>();
            var split = UnityEngine.ScriptableObject.CreateInstance<Chatterbox.ChatterboxSplitModelSet>();
            try
            {
                foreach (ModelSet set in new ModelSet[] { four, split })
                {
                    string[] files = set.DownloadGraphRoles.Select(role => "onnx/" + role.fileStem + ".onnx")
                        .Concat(new[] { "tokenizer.json" }).ToArray();
                    var snapshot = new HfModelInfo { sha = "revision",
                        siblings = files.Select(path => new HfSibling { rfilename = path, size = 1 }).ToArray() };
                    var discovered = await HuggingFaceModelRepository.ScanSnapshotAsync(
                        new ModelDownloadRequest("owner/model", graphRoles: set.DownloadGraphRoles,
                            requiredCompanionRoles: set.DownloadRequiredCompanionRoles), snapshot);
                    Assert.That(discovered.CommonFiles.Select(file => file.Role), Contains.Item("tokenizer"));
                    Assert.That(discovered.CommonFiles.Select(file => file.Role), Does.Not.Contain("voice"));
                }
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(four);
                UnityEngine.Object.DestroyImmediate(split);
            }
        }

        [Test]
        public void WrongLayoutIsRejected()
        {
            var snapshot = new HfModelInfo { sha = "revision", siblings = new[]
            {
                new HfSibling { rfilename = "onnx/speech_encoder.onnx", size = 1 },
                new HfSibling { rfilename = "onnx/embed_tokens.onnx", size = 1 },
                new HfSibling { rfilename = "onnx/language_model.onnx", size = 1 },
                new HfSibling { rfilename = "onnx/conditional_decoder.onnx", size = 1 }
            } };
            var split = UnityEngine.ScriptableObject.CreateInstance<Chatterbox.ChatterboxSplitModelSet>();
            try
            {
                Assert.ThrowsAsync<InvalidDataException>(async () =>
                    await HuggingFaceModelRepository.ScanSnapshotAsync(
                        new ModelDownloadRequest("owner/model", graphRoles: split.DownloadGraphRoles), snapshot));
            }
            finally { UnityEngine.Object.DestroyImmediate(split); }
        }

        [Test]
        public void FailedBindingRestoresPreviousInstallation()
        {
            string root = Path.Combine(Path.GetTempPath(), "chatterbox-store-" + Guid.NewGuid().ToString("N"));
            var store = new ModelInstallationStore(root, "model-sets/one");
            try
            {
                Directory.CreateDirectory(store.DirectoryPath);
                Directory.CreateDirectory(store.DirectoryPath + ".previous");
                File.WriteAllText(Path.Combine(store.DirectoryPath, "new.txt"), "new");
                File.WriteAllText(Path.Combine(store.DirectoryPath + ".previous", "working.txt"), "working");
                store.RestorePrevious();
                Assert.That(File.Exists(Path.Combine(store.DirectoryPath, "working.txt")), Is.True);
                Assert.That(File.Exists(Path.Combine(store.DirectoryPath, "new.txt")), Is.False);
                Assert.That(Directory.Exists(store.DirectoryPath + ".previous"), Is.False);
                var other = new ModelInstallationStore(root, "model-sets/two");
                Assert.That(other.DirectoryPath, Is.Not.EqualTo(store.DirectoryPath));
            }
            finally { if (Directory.Exists(root)) Directory.Delete(root, true); }
        }

        [Test]
        public void FailedFirstBindingRemovesIncompleteInstallation()
        {
            string root = Path.Combine(Path.GetTempPath(), "chatterbox-store-" + Guid.NewGuid().ToString("N"));
            var store = new ModelInstallationStore(root, "model-sets/one");
            try
            {
                Directory.CreateDirectory(store.DirectoryPath);
                File.WriteAllText(Path.Combine(store.DirectoryPath, "new.txt"), "new");
                store.RestorePrevious();
                Assert.That(Directory.Exists(store.DirectoryPath), Is.False);
            }
            finally { if (Directory.Exists(root)) Directory.Delete(root, true); }
        }

        [Test]
        public async Task LayoutUpdateCanRestoreOrCompletePreviousInstallation()
        {
            string fixture = Path.GetFullPath(Path.Combine(UnityEngine.Application.dataPath, "..",
                "KitsuMateOnnxFixtures", "text-embedding", "all-minilm"));
            string fixtureGraph = Path.Combine(fixture, "model_q4f16.onnx");
            byte[] graph = File.ReadAllBytes(fixtureGraph);
            string externalData = OnnxLightweightMetadataReader.Read(fixtureGraph).ExternalData.Single().Location;
            byte[] weights = File.ReadAllBytes(Path.Combine(fixture, externalData));
            string root = Path.Combine(Path.GetTempPath(), "chatterbox-layout-" + Guid.NewGuid().ToString("N"));
            var store = new ModelInstallationStore(root, "model");
            var artifacts = new Dictionary<string, List<DiscoveredArtifact>>(StringComparer.Ordinal);
            var supplied = new Dictionary<string, string>(StringComparer.Ordinal);

            void Add(string role, string path)
            {
                string sidecar = Path.Combine(Path.GetDirectoryName(path), externalData).Replace('\\', '/');
                string source = Path.Combine(root, "source", path);
                string sidecarSource = Path.Combine(root, "source", sidecar);
                Directory.CreateDirectory(Path.GetDirectoryName(source));
                Directory.CreateDirectory(Path.GetDirectoryName(sidecarSource));
                File.WriteAllBytes(source, graph);
                File.WriteAllBytes(sidecarSource, weights);
                supplied[path] = source;
                supplied[sidecar] = sidecarSource;
                if (!artifacts.TryGetValue(role, out var choices)) artifacts[role] = choices = new List<DiscoveredArtifact>();
                choices.Add(new DiscoveredArtifact(role, "fp32", false, new[]
                {
                    new DiscoveredFile(role, path, Hash(graph), graph.Length),
                    new DiscoveredFile(role, sidecar, Hash(weights), weights.Length)
                }));
            }

            try
            {
                Add("speech-encoder", "onnx/speech_encoder.onnx");
                Add("embed-tokens", "onnx/embed_tokens.onnx");
                Add("language-model", "onnx/language_model.onnx");
                Add("conditional-decoder", "onnx/conditional_decoder.onnx");
                Add("speech-encoder", "onnx/speech_encoder_slim.onnx");
                Add("embedding-language-model", "onnx/embedding_language_model_last.onnx");
                Add("flow-prepare", "onnx/flow_prepare_slim.onnx");
                Add("flow-step", "onnx/flow_step_slim.onnx");
                Add("vocoder", "onnx/vocoder_slim.onnx");
                string tokenizer = Path.Combine(root, "source", "tokenizer.json");
                File.WriteAllText(tokenizer, "{}");
                supplied["tokenizer.json"] = tokenizer;
                var common = new[] { new DiscoveredFile("tokenizer", "tokenizer.json", "", 2) };
                var repository = new DiscoveredRepository("owner", "chatterbox", "chatterbox", "revision",
                    artifacts.ToDictionary(item => item.Key, item => item.Value.ToArray()), common, Array.Empty<string>());
                var four = new Dictionary<string, string>
                {
                    ["speech-encoder"] = "onnx/speech_encoder.onnx",
                    ["embed-tokens"] = "onnx/embed_tokens.onnx",
                    ["language-model"] = "onnx/language_model.onnx",
                    ["conditional-decoder"] = "onnx/conditional_decoder.onnx"
                };
                var split = new Dictionary<string, string>
                {
                    ["speech-encoder"] = "onnx/speech_encoder_slim.onnx",
                    ["embedding-language-model"] = "onnx/embedding_language_model_last.onnx",
                    ["flow-prepare"] = "onnx/flow_prepare_slim.onnx",
                    ["flow-step"] = "onnx/flow_step_slim.onnx",
                    ["vocoder"] = "onnx/vocoder_slim.onnx"
                };

                await store.InstallAsync(repository, four, suppliedFiles: supplied);
                Assert.That(store.Read().Files.Count, Is.EqualTo(6));
                await store.InstallAsync(repository, split, suppliedFiles: supplied, retainPreviousUntilBinding: true);
                Assert.That(store.Read().Files.Count, Is.EqualTo(7));
                Assert.That(File.Exists(Path.Combine(store.DirectoryPath + ".previous", four["language-model"])), Is.True);
                store.RestorePrevious();
                Assert.That(store.Read().Files.Count, Is.EqualTo(6));
                Assert.That(File.Exists(Path.Combine(store.DirectoryPath, split["vocoder"])), Is.False);

                await store.InstallAsync(repository, split, suppliedFiles: supplied, retainPreviousUntilBinding: true);
                store.CompleteBinding();
                Assert.That(store.Read().Files.Count, Is.EqualTo(7));
                Assert.That(Directory.Exists(store.DirectoryPath + ".previous"), Is.False);
            }
            finally { if (Directory.Exists(root)) Directory.Delete(root, true); }
        }

        private static string Hash(byte[] bytes)
        {
            using var sha = SHA256.Create();
            return BitConverter.ToString(sha.ComputeHash(bytes)).Replace("-", "").ToLowerInvariant();
        }
    }
}

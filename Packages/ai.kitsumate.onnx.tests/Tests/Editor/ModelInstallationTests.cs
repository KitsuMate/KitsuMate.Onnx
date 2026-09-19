using System;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using KitsuMate.Onnx.Download;
using NUnit.Framework;
using UnityEngine;

namespace KitsuMate.Onnx.Tests
{
    public sealed class ModelInstallationTests
    {
        [Test]
        public void PersistentModelReferencesSerializeRelativePaths()
        {
            string path = Path.Combine(Application.persistentDataPath, "KitsuMateModels", "owner", "model", "graph.onnx");
            var reference = new OnnxModelReference();
            reference.ConfigureFile(path, "", null, null);
            Assert.That(reference.Root, Is.EqualTo(OnnxModelReference.FileRoot.PersistentData));
            Assert.That(reference.FilePath, Is.EqualTo("KitsuMateModels/owner/model/graph.onnx"));
            Assert.That(JsonUtility.ToJson(reference), Does.Not.Contain(Application.persistentDataPath));
            var restored = JsonUtility.FromJson<OnnxModelReference>(JsonUtility.ToJson(reference));
            Assert.That(restored.Root, Is.EqualTo(reference.Root));
            Assert.That(restored.FilePath, Is.EqualTo(reference.FilePath));
        }

        [Test]
        public void PartialInstallationReportsAndReusesOnlyCompleteFiles()
        {
            string root = Path.Combine(Path.GetTempPath(), "kitsumate-partial-" + Guid.NewGuid().ToString("N"));
            var store = new ModelInstallationStore(root, "owner/model");
            var repository = Repository();
            try
            {
                Directory.CreateDirectory(store.DirectoryPath);
                File.WriteAllText(Path.Combine(store.DirectoryPath, "weights.data"), "data");
                File.WriteAllText(Path.Combine(store.DirectoryPath, "tokenizer.json"), "{}");
                ModelInstallationStore.WriteRecord(new DownloadedModel(store.DirectoryPath,
                    new ModelIdentity("test", "model", "revision", ""), ModelInstallationStore.RequiredFiles(repository, Selection()), repository.Repository));
                File.Delete(Path.Combine(store.DirectoryPath, "tokenizer.json"));
                Assert.That(store.Read(), Is.Null);
                Assert.That(store.Read(true).Files.Count, Is.EqualTo(1));
                var available = store.AvailableFiles(repository, ModelInstallationStore.RequiredFiles(repository, Selection()));
                Assert.That(available.ContainsKey("weights.data"), Is.True);
                Assert.That(available.ContainsKey("tokenizer.json"), Is.False);
                Assert.That(store.AvailableFiles(Repository("different-owner"), ModelInstallationStore.RequiredFiles(repository, Selection())).Count, Is.Zero);
            }
            finally { if (Directory.Exists(root)) Directory.Delete(root, true); }
        }

        [Test]
        public async Task SuppliedFilesCompleteSetupWithVerifiedContent()
        {
            string root = Path.Combine(Path.GetTempPath(), "kitsumate-supplied-" + Guid.NewGuid().ToString("N"));
            var store = new ModelInstallationStore(root, "owner/model");
            try
            {
                Directory.CreateDirectory(root);
                string weights = Path.Combine(root, "provided.data"), text = Path.Combine(root, "provided.json");
                File.WriteAllText(weights, "data");
                File.WriteAllText(text, "{}");
                string oldStaging = store.DownloadDirectory(Repository("old-owner"));
                Directory.CreateDirectory(oldStaging);
                File.WriteAllText(Path.Combine(oldStaging, "old.data"), "old");
                string currentStaging = store.DownloadDirectory(Repository());
                Directory.CreateDirectory(currentStaging);
                File.WriteAllText(Path.Combine(currentStaging, "unselected.data"), "old");
                var supplied = new Dictionary<string, string> { ["weights.data"] = weights, ["tokenizer.json"] = text };
                await store.InstallAsync(Repository(), Selection(), suppliedFiles: supplied);
                Assert.That(store.Read().Files.Count, Is.EqualTo(2));
                Assert.That(store.Read().Repository, Is.EqualTo("owner/model"));
                Assert.That(File.ReadAllText(weights), Is.EqualTo("data"), "Caller-owned files must be preserved.");
                Assert.That(File.ReadAllText(store.Read().GetPath("tokenizer")), Is.EqualTo("{}"));
                Assert.That(Directory.Exists(oldStaging), Is.False);
                Assert.That(File.Exists(Path.Combine(store.DirectoryPath, "unselected.data")), Is.False);
            }
            finally { if (Directory.Exists(root)) Directory.Delete(root, true); }
        }

        [Test]
        public async Task CancelledSetupKeepsCompletedFilesAndExistingInstallation()
        {
            string root = Path.Combine(Path.GetTempPath(), "kitsumate-retry-" + Guid.NewGuid().ToString("N"));
            var store = new ModelInstallationStore(root, "owner/model");
            var repository = Repository();
            using var cancellation = new CancellationTokenSource();
            try
            {
                Directory.CreateDirectory(store.DirectoryPath);
                File.WriteAllText(Path.Combine(store.DirectoryPath, "keep.txt"), "old installation");
                string staging = store.DownloadDirectory(repository);
                Directory.CreateDirectory(staging);
                File.WriteAllText(Path.Combine(staging, "weights.data"), "data");
                File.WriteAllText(Path.Combine(staging, "tokenizer.json"), "{}");
                try
                {
                    await store.InstallAsync(repository, Selection(), new InlineProgress(_ => cancellation.Cancel()), cancellation.Token);
                    Assert.Fail("Expected cancellation.");
                }
                catch (OperationCanceledException) { }
                Assert.That(File.Exists(Path.Combine(store.DirectoryPath, "keep.txt")), Is.True);
                Assert.That(File.Exists(Path.Combine(staging, "weights.data")), Is.True);
                await store.InstallAsync(repository, Selection());
                Assert.That(store.Read().Files.Count, Is.EqualTo(2));
            }
            finally { if (Directory.Exists(root)) Directory.Delete(root, true); }
        }

        [Test]
        public async Task InterruptedBindingCanRestorePreviousOrKeepCurrentInstallation()
        {
            string root = Path.Combine(Path.GetTempPath(), "kitsumate-recovery-" + Guid.NewGuid().ToString("N"));
            var store = new ModelInstallationStore(root, "owner/model");
            var repository = Repository();
            try
            {
                Directory.CreateDirectory(root);
                string oldWeights = Path.Combine(root, "old.data");
                string oldTokenizer = Path.Combine(root, "old.json");
                File.WriteAllText(oldWeights, "data");
                File.WriteAllText(oldTokenizer, "{}");
                await store.InstallAsync(repository, Selection(), suppliedFiles: new Dictionary<string, string>
                    { ["weights.data"] = oldWeights, ["tokenizer.json"] = oldTokenizer });
                File.WriteAllText(Path.Combine(store.DirectoryPath, "marker.txt"), "previous");

                string newWeights = Path.Combine(root, "new.data");
                string newTokenizer = Path.Combine(root, "new.json");
                File.WriteAllText(newWeights, "data");
                File.WriteAllText(newTokenizer, "{}");
                await store.InstallAsync(repository, Selection(), suppliedFiles: new Dictionary<string, string>
                    { ["weights.data"] = newWeights, ["tokenizer.json"] = newTokenizer },
                    retainPreviousUntilBinding: true);
                Assert.That(store.HasPendingBinding, Is.True);
                Assert.That(File.Exists(Path.Combine(store.DirectoryPath, "marker.txt")), Is.False);

                store.RestorePrevious();
                Assert.That(store.HasPendingBinding, Is.False);
                Assert.That(File.ReadAllText(Path.Combine(store.DirectoryPath, "marker.txt")), Is.EqualTo("previous"));

                await store.InstallAsync(repository, Selection(), suppliedFiles: new Dictionary<string, string>
                    { ["weights.data"] = newWeights, ["tokenizer.json"] = newTokenizer },
                    retainPreviousUntilBinding: true);
                Assert.That(store.HasPendingBinding, Is.True);
                store.CompleteBinding();
                Assert.That(store.HasPendingBinding, Is.False);
                Assert.That(File.Exists(Path.Combine(store.DirectoryPath, "marker.txt")), Is.False);
                Assert.That(store.Read(), Is.Not.Null);
            }
            finally { if (Directory.Exists(root)) Directory.Delete(root, true); }
        }

        [Test]
        public void MovingFilesRemovesSourceAndPreservesDestinationOnConflict()
        {
            string root = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(root);
            try
            {
                string source = Path.Combine(root, "source");
                string destination = Path.Combine(root, "destination");
                File.WriteAllText(source, "model");
                ModelBuildFiles.Move(source, destination);
                Assert.That(File.Exists(source), Is.False);
                Assert.That(File.ReadAllText(destination), Is.EqualTo("model"));
                ModelBuildFiles.Move(destination, destination);
                File.WriteAllText(source, "replacement");
                Assert.Throws<IOException>(() => ModelBuildFiles.Move(source, destination));
                Assert.That(File.ReadAllText(source), Is.EqualTo("replacement"));
                Assert.That(File.ReadAllText(destination), Is.EqualTo("model"));
            }
            finally { Directory.Delete(root, true); }
        }

        [Test]
        public void BuildCopiesReuseCompleteFilesAndRepairChangedFiles()
        {
            string root = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(root);
            try
            {
                string source = Path.Combine(root, "source");
                string destination = Path.Combine(root, "copy");
                File.WriteAllText(source, "complete");
                Assert.That(ModelBuildFiles.Copy(source, destination), Is.True);
                Assert.That(ModelBuildFiles.Copy(source, destination), Is.False);
                File.WriteAllText(destination, "partial");
                Assert.That(ModelBuildFiles.Matches(source, destination), Is.False);
                Assert.That(ModelBuildFiles.Copy(source, destination), Is.True);
                File.SetLastWriteTimeUtc(source, File.GetLastWriteTimeUtc(source).AddMinutes(1));
                Assert.That(ModelBuildFiles.Matches(source, destination), Is.False);
                Assert.That(ModelBuildFiles.Copy(source, destination), Is.True);
            }
            finally { Directory.Delete(root, true); }
        }

        private static Dictionary<string, string> Selection() => new() { ["model"] = "weights.data" };
        private static DiscoveredRepository Repository(string owner = "owner") => new(owner, "model", "test", "revision",
            new Dictionary<string, DiscoveredArtifact[]> { ["model"] = new[] { new DiscoveredArtifact("model", "default", false,
                new[] { new DiscoveredFile("model", "weights.data", "3a6eb0790f39ac87c94f3856b2dd2c5d110e6811602261a9a923d3bb23adc8b7", 4) }) } },
            new[] { new DiscoveredFile("tokenizer", "tokenizer.json", "", 2) }, new[] { "model" });
        private sealed class InlineProgress : IProgress<float>
        {
            private readonly Action<float> report;
            public InlineProgress(Action<float> report) => this.report = report;
            public void Report(float value) => report(value);
        }

        [Test]
        public void InstallationRejectsPathsOutsideRoot()
        {
            Assert.Throws<InvalidDataException>(() => new ModelInstallationStore(Path.GetTempPath(), "../outside"));
        }

        [Test]
        public void InstallationLockProtectsOtherDownloadAndUninstall()
        {
            string root = Path.Combine(Path.GetTempPath(), "kitsumate-lock-" + Guid.NewGuid().ToString("N"));
            var first = new ModelInstallationStore(root, "model");
            var second = new ModelInstallationStore(root, "model");
            try
            {
                Directory.CreateDirectory(first.DirectoryPath);
                string existing = Path.Combine(first.DirectoryPath, "keep.txt");
                File.WriteAllText(existing, "existing installation");
                using (first.AcquireLock())
                {
                    Assert.Throws<IOException>(() => second.AcquireLock());
                    Assert.Throws<IOException>(() => second.Uninstall());
                    Assert.IsTrue(File.Exists(existing));
                }
                using (second.AcquireLock()) { }
                Assert.IsFalse(File.Exists(first.DirectoryPath + ".lock"));
            }
            finally { if (Directory.Exists(root)) Directory.Delete(root, true); }
        }

        [Test]
        public async Task CancelledTransferReleasesAndDeletesPartialFile()
        {
            string root = Path.Combine(Path.GetTempPath(), "kitsumate-cancel-" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(root);
            string path = Path.Combine(root, "weights.data");
            var listener = new System.Net.Sockets.TcpListener(System.Net.IPAddress.Loopback, 0);
            listener.Start();
            var finishServer = new TaskCompletionSource<bool>();
            var received = new TaskCompletionSource<bool>();
            async Task Serve()
            {
                using var client = await listener.AcceptTcpClientAsync();
                using var stream = client.GetStream();
                var request = new byte[4096];
                await stream.ReadAsync(request, 0, request.Length);
                byte[] response = System.Text.Encoding.ASCII.GetBytes("HTTP/1.1 200 OK\r\nContent-Length: 1000000\r\nConnection: close\r\n\r\nx");
                await stream.WriteAsync(response, 0, response.Length);
                await finishServer.Task;
            }
            Task server = Serve();
            using var cancellation = new CancellationTokenSource();
            try
            {
                int port = ((System.Net.IPEndPoint)listener.LocalEndpoint).Port;
                var method = typeof(ModelDownloader).GetMethod("DownloadFileAsync", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static);
                var transfer = (Task)method.Invoke(null, new object[] { $"http://127.0.0.1:{port}/weights", path,
                    new DiscoveredFile("weights", "weights.data", "", 1000000), null,
                    new Action<long>(_ => received.TrySetResult(true)), cancellation.Token });
                Assert.AreSame(received.Task, await Task.WhenAny(received.Task, Task.Delay(5000)), "Transfer never started.");
                cancellation.Cancel();
                Assert.AreSame(transfer, await Task.WhenAny(transfer, Task.Delay(5000)), "Cancellation did not interrupt the response stream.");
                try { await transfer; Assert.Fail("Expected cancellation."); }
                catch (OperationCanceledException) { }
                Assert.IsFalse(File.Exists(path + ".partial"));
                Assert.IsFalse(File.Exists(path));
            }
            finally
            {
                cancellation.Cancel();
                finishServer.TrySetResult(true);
                listener.Stop();
                await server;
                if (Directory.Exists(root)) Directory.Delete(root, true);
            }
        }

        [Test]
        public async Task ResolvedTextReferencesLeaveConfigurationUnchanged()
        {
            string root = Path.Combine(Path.GetTempPath(), "kitsumate-installation-" + Guid.NewGuid().ToString("N"));
            var set = ScriptableObject.CreateInstance<TestModelSet>();
            ResolvedModelSet resolved = null;
            try
            {
                set.Download.repository = "test/model";
                set.Download.installationFolder = "model";
                var store = set.Installation(root);
                Directory.CreateDirectory(store.DirectoryPath);
                File.WriteAllText(Path.Combine(store.DirectoryPath, "tokenizer.json"), "{}");
                ModelInstallationStore.WriteRecord(new DownloadedModel(store.DirectoryPath,
                    new ModelIdentity("test", "model", "revision", "hash"),
                    new[] { new DiscoveredFile("tokenizer", "tokenizer.json", "", 2) }));
                resolved = await set.ResolveAsync(root, CancellationToken.None);
                var copy = (TestModelSet)resolved.Model;
                TextFileReference tokenizer = copy.Tokenizer;
                Assert.That(copy, Is.Not.SameAs(set));
                Assert.That(set.Tokenizer?.IsAvailable ?? false, Is.False);
                Assert.That(tokenizer.text, Is.EqualTo("{}"));
                Assert.That(copy.Identity.Revision, Is.EqualTo("revision"));
                resolved.Dispose();
                Assert.That(tokenizer.IsAvailable, Is.True);
                Assert.That(copy == null, Is.True);
            }
            finally
            {
                resolved?.Dispose();
                UnityEngine.Object.DestroyImmediate(set);
                if (Directory.Exists(root)) Directory.Delete(root, true);
            }
        }

        private sealed class TestModelSet : StandardModelSet
        {
            public TextFileReference Tokenizer;
            public override string DisplayName => "Test";
            public override bool IsComplete => Tokenizer != null;
            public override IOnnxModelSource[] GetAllModels() => Array.Empty<IOnnxModelSource>();
            public override ModelValidationResult Validate(ModelValidationContext context) => new();
            protected override Task BindInstallationAsync(DownloadedModel installation, ResolvedModelSet resources, CancellationToken cancellationToken)
            {
                Tokenizer = resources.ReadText(installation, "tokenizer");
                return Task.CompletedTask;
            }
        }
    }
}

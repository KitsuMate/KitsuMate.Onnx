using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using NUnit.Framework;
using UnityEditor;
using UnityEngine;

namespace KitsuMate.Onnx.Tests
{
    public sealed class UnifiedInferenceArchitectureTests
    {
        [Test]
        public void Runtime_DoesNotDisposeCallerBackend()
        {
            var model = ScriptableObject.CreateInstance<FakeModelSet>();
            var engine = ScriptableObject.CreateInstance<FakeEngine>();
            var backend = ScriptableObject.CreateInstance<FakeBackend>();
            engine.Model = model;
            using (InferenceEngineRuntime<int, int> runtime = engine.CreateRuntimeAsync(backend).GetAwaiter().GetResult())
                Assert.AreEqual(4, runtime.RunAsync(3).GetAwaiter().GetResult());
            Assert.IsFalse(backend.WasDisposed);
            backend.Dispose();
            UnityEngine.Object.DestroyImmediate(engine); UnityEngine.Object.DestroyImmediate(model); UnityEngine.Object.DestroyImmediate(backend);
        }

        [Test]
        public void Runtime_SerializesConcurrentRuns()
        {
            var model = ScriptableObject.CreateInstance<FakeModelSet>(); var engine = ScriptableObject.CreateInstance<FakeEngine>(); var backend = ScriptableObject.CreateInstance<FakeBackend>(); engine.Model = model;
            using InferenceEngineRuntime<int, int> runtime = engine.CreateRuntimeAsync(backend).GetAwaiter().GetResult();
            Task<int> a = Invoke(runtime, 1), b = Invoke(runtime, 2); Task.WhenAll(a, b).GetAwaiter().GetResult();
            Assert.AreEqual(1, ((FakeRuntime)runtime).MaximumConcurrency);
            runtime.Dispose(); backend.Dispose(); UnityEngine.Object.DestroyImmediate(engine); UnityEngine.Object.DestroyImmediate(model); UnityEngine.Object.DestroyImmediate(backend);
        }

        [Test]
        public void Runtime_UsesEngineBackendByDefault_AndAllowsExplicitOverride()
        {
            var model = ScriptableObject.CreateInstance<FakeModelSet>();
            var engine = ScriptableObject.CreateInstance<FakeEngine>();
            var configured = ScriptableObject.CreateInstance<FakeBackend>();
            var overrideBackend = ScriptableObject.CreateInstance<FakeBackend>();
            engine.Model = model;
            var serialized = new SerializedObject(engine);
            serialized.FindProperty("backend").objectReferenceValue = configured;
            serialized.ApplyModifiedPropertiesWithoutUndo();

            using (var runtime = (FakeRuntime)engine.CreateRuntimeAsync().GetAwaiter().GetResult())
                Assert.AreSame(configured, runtime.LoadedBackend);
            using (var runtime = (FakeRuntime)engine.CreateRuntimeAsync(overrideBackend).GetAwaiter().GetResult())
                Assert.AreSame(overrideBackend, runtime.LoadedBackend);

            Assert.IsFalse(configured.WasDisposed);
            Assert.IsFalse(overrideBackend.WasDisposed);
            configured.Dispose();
            overrideBackend.Dispose();
            UnityEngine.Object.DestroyImmediate(engine);
            UnityEngine.Object.DestroyImmediate(model);
            UnityEngine.Object.DestroyImmediate(configured);
            UnityEngine.Object.DestroyImmediate(overrideBackend);
        }

        [Test]
        public void Tensor_ValidatesShapeBufferTypeAndDisposedAccess()
        {
            Assert.Throws<ArgumentException>(() => new OnnxTensor(new[] { 2, 2 }, OnnxTensorElementType.Float, new float[3]));
            Assert.Throws<ArgumentException>(() => new OnnxTensor(new[] { 1 }, OnnxTensorElementType.Int32, new long[1]));
            Assert.Throws<ArgumentOutOfRangeException>(() => new OnnxTensor(new[] { -1 }, OnnxTensorElementType.Float, Array.Empty<float>()));
            var tensor = OnnxTensor.FromArray(new[] { 1f }, new[] { 1 });
            tensor.Dispose();
            Assert.Throws<ObjectDisposedException>(() => _ = tensor.Data);
            Assert.Throws<ObjectDisposedException>(() => tensor.AsFloatArray());
        }

        [Test]
        public void AndroidStager_RejectsTraversalAndInvalidHash()
        {
            var stager = new AndroidModelStager(new CountingFetcher(new byte[] { 1 }));
            var environment = new OnnxRuntimeEnvironment("models", "source", Path.GetTempPath());
            Assert.ThrowsAsync<OnnxModelPreparationException>(() => stager.StageAsync("../model.onnx", new string('0', 64), environment, default));
            Assert.ThrowsAsync<OnnxModelPreparationException>(() => stager.StageAsync("model.onnx", "", environment, default));
        }

        [Test]
        public async Task AndroidStager_IsSingleFlightAndChecksumFirst()
        {
            byte[] content = { 1, 2, 3, 4 };
            string hash;
            using (SHA256 algorithm = SHA256.Create())
                hash = string.Concat(algorithm.ComputeHash(content).Select(value => value.ToString("x2")));
            string root = Path.Combine(Path.GetTempPath(), "kitsumate-stager-" + Guid.NewGuid().ToString("N"));
            var fetcher = new CountingFetcher(content);
            var stager = new AndroidModelStager(fetcher);
            var environment = new OnnxRuntimeEnvironment("models", "source", root);
            try
            {
                Task<string> first = stager.StageAsync("nested/model.onnx", hash, environment, default);
                Task<string> second = stager.StageAsync("nested/model.onnx", hash, environment, default);
                string[] paths = await Task.WhenAll(first, second);
                Assert.AreEqual(paths[0], paths[1]);
                Assert.AreEqual(1, fetcher.FetchCount);
                CollectionAssert.AreEqual(content, File.ReadAllBytes(paths[0]));
            }
            finally
            {
                if (Directory.Exists(root)) Directory.Delete(root, true);
            }
        }

        [Test]
        public async Task Runtime_UnloadIsSharedAndWaitsForActiveInference()
        {
            var runtime = new DrainRuntime();
            var backend = ScriptableObject.CreateInstance<FakeBackend>();
            await runtime.LoadAsync(backend, default, default);
            Task<int> run = runtime.RunAsync(1);
            await runtime.Entered.Task;
            Task firstUnload = runtime.UnloadAsync();
            Task secondUnload = runtime.UnloadAsync();
            Assert.AreSame(firstUnload, secondUnload);
            Assert.IsFalse(runtime.Unloaded);
            runtime.Release.TrySetResult(true);
            Assert.AreEqual(2, await run);
            await firstUnload;
            Assert.IsTrue(runtime.Unloaded);
            runtime.Dispose();
            UnityEngine.Object.DestroyImmediate(backend);
        }

        private static Task<int> Invoke(InferenceEngineRuntime<int, int> runtime, int value) => runtime.RunAsync(value);

        private sealed class FakeEngine : InferenceEngine<int, int>
        {
            public FakeModelSet Model;
            public override ModelSet ModelSet => Model;
            protected override InferenceEngineRuntime<int, int> CreateRuntime() => new FakeRuntime();
        }
        private sealed class FakeRuntime : InferenceEngineRuntime<int, int>
        {
            private int concurrency; public int MaximumConcurrency { get; private set; }
            public OnnxBackend LoadedBackend { get; private set; }
            protected override Task OnLoadAsync(CancellationToken cancellationToken) { LoadedBackend = Backend; return Task.CompletedTask; }
            protected override async Task<int> OnRunAsync(int request, CancellationToken cancellationToken)
            {
                int current = Interlocked.Increment(ref concurrency); MaximumConcurrency = Math.Max(MaximumConcurrency, current);
                await Task.Delay(20, cancellationToken).ConfigureAwait(false); Interlocked.Decrement(ref concurrency); return request + 1;
            }
            protected override Task OnUnloadAsync(CancellationToken cancellationToken) => Task.CompletedTask;
        }
        private sealed class DrainRuntime : InferenceEngineRuntime<int, int>
        {
            public readonly TaskCompletionSource<bool> Entered = new(TaskCreationOptions.RunContinuationsAsynchronously);
            public readonly TaskCompletionSource<bool> Release = new(TaskCreationOptions.RunContinuationsAsynchronously);
            public bool Unloaded { get; private set; }
            protected override Task OnLoadAsync(CancellationToken cancellationToken) => Task.CompletedTask;
            protected override async Task<int> OnRunAsync(int request, CancellationToken cancellationToken)
            {
                Entered.TrySetResult(true);
                await Release.Task;
                return request + 1;
            }
            protected override Task OnUnloadAsync(CancellationToken cancellationToken)
            {
                Unloaded = true;
                return Task.CompletedTask;
            }
        }

        private sealed class CountingFetcher : IOnnxModelContentFetcher
        {
            private readonly byte[] content;
            public CountingFetcher(byte[] content) => this.content = content;
            public int FetchCount { get; private set; }
            public async Task FetchAsync(string sourceUri, string destinationPath, CancellationToken cancellationToken)
            {
                FetchCount++;
                await Task.Yield();
                cancellationToken.ThrowIfCancellationRequested();
                File.WriteAllBytes(destinationPath, content);
            }
        }
        private sealed class FakeModelSet : StandardModelSet
        {
            public override string DisplayName => "Fake"; public override bool IsComplete => true; public override IOnnxModelSource[] GetAllModels() => Array.Empty<IOnnxModelSource>();
            public override ModelValidationResult Validate(ModelValidationContext context) => new ModelValidationResult();
        }
        private sealed class FakeBackend : OnnxBackend
        {
            public bool WasDisposed; public override string DisplayName => "Fake"; public override IReadOnlyList<RuntimePlatform> SupportedPlatforms => Array.Empty<RuntimePlatform>(); public override bool IsAvailable => true; public override int Priority => 0;
            public override IOnnxSession CreateSession(byte[] modelData, OnnxSessionOptions options) => throw new NotSupportedException();
            public override IOnnxSession CreateSession(string modelPath, OnnxSessionOptions options) => throw new NotSupportedException();
            public override void Dispose() => WasDisposed = true;
        }
    }
}

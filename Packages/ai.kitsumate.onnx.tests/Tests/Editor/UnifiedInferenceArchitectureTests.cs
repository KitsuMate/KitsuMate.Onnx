using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
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

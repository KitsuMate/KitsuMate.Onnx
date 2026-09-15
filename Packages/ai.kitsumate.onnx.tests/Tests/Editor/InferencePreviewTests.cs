using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using KitsuMate.Onnx.Editor;
using NUnit.Framework;
using UnityEditor;
using UnityEngine;

namespace KitsuMate.Onnx.Tests
{
    public sealed class InferencePreviewTests
    {
        private PreviewEngine engine;
        private PreviewModel model;
        private PreviewBackend backend;
        private InferencePreviewSession<int, int> session;
        private static Task<int> Input(CancellationToken token) => Task.FromResult(7);

        [SetUp] public void SetUp()
        {
            engine = ScriptableObject.CreateInstance<PreviewEngine>();
            model = ScriptableObject.CreateInstance<PreviewModel>();
            backend = ScriptableObject.CreateInstance<PreviewBackend>();
            engine.Model = model;
            session = new InferencePreviewSession<int, int>();
        }
        [TearDown] public async Task TearDown()
        {
            await session.CloseAsync();
            UnityEngine.Object.DestroyImmediate(engine);
            UnityEngine.Object.DestroyImmediate(model);
            UnityEngine.Object.DestroyImmediate(backend);
        }
        private Task<InferenceEngineRuntime<int, int>> Load(CancellationToken token) => engine.CreateRuntimeAsync(backend, token);

        [Test] public async Task ReusesRuntimeAndReloadsChangedConfiguration()
        {
            Assert.AreEqual(8, await session.RunAsync("a", Load, Input));
            var first = engine.Last;
            await session.RunAsync("a", Load, Input);
            Assert.AreEqual(1, engine.Loads);
            await session.RunAsync("b", Load, Input);
            Assert.AreEqual(2, engine.Loads);
            Assert.IsTrue(first.Disposed);
            Assert.IsFalse(backend.Disposed);
        }
        [Test] public async Task CancelDuringRunDiscardsLateResultAndAllowsRetry()
        {
            engine.RunGate = new TaskCompletionSource<bool>();
            var running = session.RunAsync("a", Load, Input);
            Assert.IsTrue(session.IsBusy);
            session.Cancel();
            Assert.AreEqual("Cancelling", session.Status);
            Assert.IsFalse(running.IsCompleted);
            engine.RunGate.SetResult(true);
            try { await running; Assert.Fail("Cancelled result was returned."); } catch (OperationCanceledException) { }
            Assert.IsFalse(session.IsLoaded);
            Assert.IsTrue(engine.Last.Disposed);
            engine.RunGate = null;
            Assert.AreEqual(8, await session.RunAsync("a", Load, Input));
        }
        [Test] public async Task CloseDuringLoadDisposesLateRuntime()
        {
            var gate = new TaskCompletionSource<bool>();
            async Task<InferenceEngineRuntime<int, int>> LateLoad(CancellationToken token)
            {
                await gate.Task;
                return await Load(CancellationToken.None);
            }
            var running = session.RunAsync("a", LateLoad, Input);
            await session.CloseAsync();
            gate.SetResult(true);
            try { await running; Assert.Fail("Closed preview returned a result."); } catch (OperationCanceledException) { }
            Assert.IsTrue(engine.Last.Disposed);
            Assert.IsFalse(session.IsBusy);
        }
        [Test] public async Task CancelDuringLoadAllowsRetry()
        {
            var gate = new TaskCompletionSource<bool>();
            async Task<InferenceEngineRuntime<int, int>> LateLoad(CancellationToken token)
            {
                await gate.Task;
                return await Load(CancellationToken.None);
            }
            var running = session.RunAsync("a", LateLoad, Input);
            session.Cancel(); gate.SetResult(true);
            try { await running; Assert.Fail(); } catch (OperationCanceledException) { }
            Assert.IsTrue(engine.Last.Disposed);
            Assert.AreEqual(8, await session.RunAsync("a", Load, Input));
        }
        [Test] public async Task CloseDuringRunWaitsForOperationCleanup()
        {
            engine.RunGate = new TaskCompletionSource<bool>();
            var running = session.RunAsync("a", Load, Input);
            await session.CloseAsync();
            Assert.IsFalse(engine.Last.Disposed);
            engine.RunGate.SetResult(true);
            try { await running; Assert.Fail(); } catch (OperationCanceledException) { }
            Assert.IsTrue(engine.Last.Disposed);
        }
        [Test] public async Task RejectsConcurrentRuns()
        {
            engine.RunGate = new TaskCompletionSource<bool>();
            var first = session.RunAsync("a", Load, Input);
            try { await session.RunAsync("a", Load, Input); Assert.Fail(); } catch (InvalidOperationException) { }
            engine.RunGate.SetResult(true);
            await first;
        }
        [Test] public async Task FailedLoadCanBeRetried()
        {
            Task<InferenceEngineRuntime<int, int>> Fail(CancellationToken token) => throw new InvalidOperationException("load failed");
            try { await session.RunAsync("a", Fail, Input); Assert.Fail(); } catch (InvalidOperationException) { }
            Assert.IsFalse(session.IsBusy);
            Assert.AreEqual(8, await session.RunAsync("a", Load, Input));
        }
        [Test] public async Task FailedRunDisposesRuntimeAndCanRetry()
        {
            engine.FailRun = true;
            try { await session.RunAsync("a", Load, Input); Assert.Fail(); } catch (InvalidOperationException) { }
            Assert.IsTrue(engine.Last.Disposed);
            engine.FailRun = false;
            Assert.AreEqual(8, await session.RunAsync("a", Load, Input));
        }
        [Test] public async Task UnloadReleasesRuntimeButNotBackend()
        {
            await session.RunAsync("a", Load, Input);
            await session.UnloadAsync();
            Assert.IsTrue(engine.Last.Disposed);
            Assert.IsFalse(backend.Disposed);
            Assert.IsFalse(session.IsLoaded);
        }
        [Serializable] private sealed class Preferences { public string Text; public string Asset; }
        [Test] public void PreferencesRoundTripWithoutChangingEngine()
        {
            string folder = "Assets/InferencePreviewTest-" + Guid.NewGuid().ToString("N");
            AssetDatabase.CreateFolder("Assets", folder.Substring(7));
            AssetDatabase.CreateAsset(engine, folder + "/engine.asset");
            string before = EditorJsonUtility.ToJson(engine);
            string key = InferenceTestPreferences.Key(engine);
            try
            {
                var input = new Preferences { Text = "hello", Asset = AssetDatabase.AssetPathToGUID(folder + "/engine.asset") };
                InferenceTestPreferences.Save(engine, input);
                var restored = new Preferences();
                InferenceTestPreferences.Load(engine, restored);
                Assert.AreEqual(input.Text, restored.Text);
                Assert.AreSame(engine, InferenceTestPreferences.Asset<PreviewEngine>(restored.Asset));
                Assert.AreEqual(before, EditorJsonUtility.ToJson(engine));
            }
            finally { EditorPrefs.DeleteKey(key); AssetDatabase.DeleteAsset(folder); }
        }
        [Test] public void FingerprintIncludesUnsavedModelAndBackendChanges()
        {
            var serialized = new SerializedObject(engine);
            serialized.FindProperty("backend").objectReferenceValue = backend;
            serialized.ApplyModifiedPropertiesWithoutUndo();
            string first = InferenceEngineEditor<int, int>.Fingerprint(engine);
            model.Version++;
            string second = InferenceEngineEditor<int, int>.Fingerprint(engine);
            Assert.AreNotEqual(first, second);
            backend.Version++;
            Assert.AreNotEqual(second, InferenceEngineEditor<int, int>.Fingerprint(engine));
        }

        private sealed class PreviewEngine : InferenceEngine<int, int>
        {
            public PreviewModel Model;
            [NonSerialized] public int Loads;
            [NonSerialized] public bool FailRun;
            [NonSerialized] public TaskCompletionSource<bool> RunGate;
            [NonSerialized] public PreviewRuntime Last;
            public override ModelSet ModelSet => Model;
            protected override InferenceEngineRuntime<int, int> CreateRuntime(ModelSet resolvedModelSet) { Loads++; return Last = new PreviewRuntime(RunGate, FailRun); }
        }
        private sealed class PreviewRuntime : InferenceEngineRuntime<int, int>
        {
            private readonly TaskCompletionSource<bool> gate;
            private readonly bool fail;
            public bool Disposed;
            public PreviewRuntime(TaskCompletionSource<bool> gate, bool fail) { this.gate = gate; this.fail = fail; }
            protected override Task OnLoadAsync(CancellationToken token) => Task.CompletedTask;
            protected override async Task<int> OnRunAsync(int value, CancellationToken token)
            {
                if (gate != null) await gate.Task;
                if (fail) throw new InvalidOperationException("run failed");
                return value + 1;
            }
            protected override Task OnUnloadAsync(CancellationToken token) => Task.CompletedTask;
            protected override void OnDispose() => Disposed = true;
        }
        private sealed class PreviewModel : StandardModelSet
        {
            public int Version;
            public override string DisplayName => "Preview test";
            public override bool IsComplete => true;
            public override IOnnxModelSource[] GetAllModels() => Array.Empty<IOnnxModelSource>();
            public override ModelValidationResult Validate(ModelValidationContext context) => new();
        }
        private sealed class PreviewBackend : OnnxBackend
        {
            public int Version;
            public bool Disposed;
            public override string DisplayName => "Preview test";
            public override IReadOnlyList<RuntimePlatform> SupportedPlatforms => Array.Empty<RuntimePlatform>();
            public override bool IsAvailable => true;
            public override int Priority => 0;
            public override IOnnxSession CreateSession(byte[] bytes, OnnxSessionOptions options) => throw new NotSupportedException();
            public override IOnnxSession CreateSession(string path, OnnxSessionOptions options) => throw new NotSupportedException();
            public override void Dispose() => Disposed = true;
        }
    }
}

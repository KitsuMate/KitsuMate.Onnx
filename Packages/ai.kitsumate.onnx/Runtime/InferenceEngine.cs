using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

namespace KitsuMate.Onnx
{
    public enum EngineLoadState { Unloaded, Loading, Loaded, Unloading, Failed }

    public interface IInferenceRuntimeProvider
    {
        Task<InferenceEngineRuntime<TRequest, TResult>> GetRuntimeAsync<TRequest, TResult>(
            InferenceEngine<TRequest, TResult> engine, OnnxBackend backend, CancellationToken cancellationToken = default);
    }

    public abstract class InferenceEngineBase : ScriptableObject
    {
        [SerializeField] private OnnxBackend backend;

        public abstract ModelSet ModelSet { get; }
        public OnnxBackend Backend => backend;
        public virtual bool SupportsParallelInference => false;
    }

    public abstract class InferenceEngine<TRequest, TResult> : InferenceEngineBase
    {
        public Task<InferenceEngineRuntime<TRequest, TResult>> CreateRuntimeAsync(
            CancellationToken cancellationToken = default)
        {
            if (Backend == null)
                throw new InvalidOperationException($"{GetType().Name} has no backend assigned.");
            return CreateRuntimeAsync(Backend, cancellationToken);
        }

        /// <summary>
        /// Creates a runtime using an explicit caller-owned backend override. Normal consumers should use
        /// <see cref="CreateRuntimeAsync(CancellationToken)"/>; this overload is intended for tests,
        /// diagnostics, and application-level runtime selection.
        /// </summary>
        public async Task<InferenceEngineRuntime<TRequest, TResult>> CreateRuntimeAsync(
            OnnxBackend backend, CancellationToken cancellationToken = default)
        {
            if (backend == null) throw new ArgumentNullException(nameof(backend));
            if (!backend.IsAvailable) throw new InvalidOperationException($"Backend '{backend.DisplayName}' is unavailable.");
            if (ModelSet == null) throw new InvalidOperationException($"{GetType().Name} has no model set assigned.");
            ModelValidationResult validation = ModelSet.Validate(new ModelValidationContext(backend));
            if (!validation.IsValid) throw new ModelValidationException(validation);
            InferenceEngineRuntime<TRequest, TResult> runtime = CreateRuntime();
            if (runtime == null) throw new InvalidOperationException($"{GetType().Name} returned no runtime.");
            try
            {
                await runtime.LoadAsync(backend, ModelSet.Identity, cancellationToken).ConfigureAwait(false);
                return runtime;
            }
            catch
            {
                runtime.Dispose();
                throw;
            }
        }

        protected abstract InferenceEngineRuntime<TRequest, TResult> CreateRuntime();
    }

    public abstract class InferenceEngineRuntimeBase : IDisposable
    {
        private static readonly List<InferenceEngineRuntimeBase> Live = new();
        public static IReadOnlyList<InferenceEngineRuntimeBase> LiveRuntimes => Live;
        public abstract EngineLoadState LoadState { get; }
        public abstract void Dispose();
        protected void Track() { lock (Live) if (!Live.Contains(this)) Live.Add(this); }
        protected void Untrack() { lock (Live) Live.Remove(this); }
        public static void DisposeAll()
        {
            InferenceEngineRuntimeBase[] copy;
            lock (Live) copy = Live.ToArray();
            foreach (InferenceEngineRuntimeBase runtime in copy) runtime.Dispose();
        }
    }

    public abstract class InferenceEngineRuntime<TRequest, TResult> : InferenceEngineRuntimeBase
    {
        private readonly SemaphoreSlim runGate;
        private readonly CancellationTokenSource lifetime = new();
        private EngineLoadState state;
        private OnnxBackend backend;
        private ModelIdentity identity;
        private Exception lastLoadError;
        private bool disposed;

        protected InferenceEngineRuntime(bool supportsParallelInference = false)
        {
            runGate = supportsParallelInference ? null : new SemaphoreSlim(1, 1);
        }

        public override EngineLoadState LoadState => state;
        public bool IsLoaded => state == EngineLoadState.Loaded;
        public Exception LastLoadError => lastLoadError;
        public ModelIdentity LoadedModelIdentity => identity;
        protected OnnxBackend Backend => backend;
        protected CancellationToken LifetimeToken => lifetime.Token;

        internal async Task LoadAsync(OnnxBackend selectedBackend, ModelIdentity modelIdentity, CancellationToken cancellationToken)
        {
            ThrowIfDisposed();
            if (state != EngineLoadState.Unloaded) throw new InvalidOperationException($"Runtime cannot load from state {state}.");
            backend = selectedBackend;
            identity = modelIdentity;
            state = EngineLoadState.Loading;
            lastLoadError = null;
            using var linked = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken, lifetime.Token);
            try
            {
                await OnLoadAsync(linked.Token).ConfigureAwait(false);
                state = EngineLoadState.Loaded;
                Track();
            }
            catch (Exception exception)
            {
                lastLoadError = exception;
                state = EngineLoadState.Failed;
                throw;
            }
        }

        public async Task<TResult> RunAsync(TRequest request, CancellationToken cancellationToken = default)
        {
            ThrowIfDisposed();
            if (state != EngineLoadState.Loaded) throw new InvalidOperationException("Runtime is not loaded.");
            using var linked = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken, lifetime.Token);
            if (runGate != null) await runGate.WaitAsync(linked.Token).ConfigureAwait(false);
            try { return await OnRunAsync(request, linked.Token).ConfigureAwait(false); }
            finally { runGate?.Release(); }
        }


        public async Task UnloadAsync(CancellationToken cancellationToken = default)
        {
            if (state == EngineLoadState.Unloaded || state == EngineLoadState.Unloading) return;
            state = EngineLoadState.Unloading;
            lifetime.Cancel();
            try { await OnUnloadAsync(cancellationToken).ConfigureAwait(false); }
            finally { state = EngineLoadState.Unloaded; Untrack(); }
        }

        protected abstract Task OnLoadAsync(CancellationToken cancellationToken);
        protected abstract Task<TResult> OnRunAsync(TRequest request, CancellationToken cancellationToken);
        protected abstract Task OnUnloadAsync(CancellationToken cancellationToken);

        public override void Dispose()
        {
            if (disposed) return;
            disposed = true;
            lifetime.Cancel();
            try { OnDispose(); }
            finally
            {
                state = EngineLoadState.Unloaded;
                Untrack();
                runGate?.Dispose();
                lifetime.Dispose();
                backend = null; // caller-owned; never dispose it
            }
        }

        protected virtual void OnDispose() { }
        private void ThrowIfDisposed() { if (disposed) throw new ObjectDisposedException(GetType().Name); }
    }

    public abstract class ThreadedInferenceEngineRuntime<TRequest, TResult> : InferenceEngineRuntime<TRequest, TResult>
    {
        protected ThreadedInferenceEngineRuntime(bool supportsParallelInference = false) : base(supportsParallelInference) { }
        public bool VerboseLogging { get; set; }

        protected sealed override Task OnLoadAsync(CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();
            OnLoadMainThread(Backend);
            if (Backend.RequiresMainThread)
            {
                OnLoadBackground(Backend);
                return Task.CompletedTask;
            }
            return Task.Run(() =>
            {
                cancellationToken.ThrowIfCancellationRequested();
                OnLoadBackground(Backend);
            }, cancellationToken);
        }

        protected sealed override Task<TResult> OnRunAsync(TRequest request, CancellationToken cancellationToken)
        {
            OnPrepareInput(request);
            if (Backend.RequiresMainThread)
            {
                cancellationToken.ThrowIfCancellationRequested();
                return Task.FromResult(OnRun(request, cancellationToken));
            }
            return Task.Run(() =>
            {
                cancellationToken.ThrowIfCancellationRequested();
                return OnRun(request, cancellationToken);
            }, cancellationToken);
        }

        protected sealed override Task OnUnloadAsync(CancellationToken cancellationToken) { OnUnload(); return Task.CompletedTask; }
        protected override void OnDispose() => OnUnload();
        protected virtual void OnLoadMainThread(OnnxBackend backend) { }
        protected virtual void OnLoadBackground(OnnxBackend backend) { }
        protected virtual void OnPrepareInput(TRequest request) { }
        protected void ThrowIfCancellationRequested() => LifetimeToken.ThrowIfCancellationRequested();
        protected abstract TResult OnRun(TRequest request, CancellationToken cancellationToken);
        protected abstract void OnUnload();
    }
}

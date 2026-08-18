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
            await Awaitable.MainThreadAsync();
            cancellationToken.ThrowIfCancellationRequested();
            if (backend == null) throw new ArgumentNullException(nameof(backend));
            if (!backend.IsAvailable) throw new InvalidOperationException($"Backend '{backend.DisplayName}' is unavailable.");
            if (ModelSet == null) throw new InvalidOperationException($"{GetType().Name} has no model set assigned.");
            OnnxSettings settings = OnnxSettings.Load();
            OnnxRuntimeEnvironment environment = OnnxSettings.CaptureEnvironment(settings);
            foreach (IOnnxModelSource source in ModelSet.GetAllModels())
            {
                cancellationToken.ThrowIfCancellationRequested();
                if (source is OnnxModelReference reference)
                    await reference.PrepareForRuntimeAsync(environment, cancellationToken);
            }
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
        private readonly object lifecycleLock = new();
        private readonly SemaphoreSlim runGate;
        private readonly CancellationTokenSource lifetime = new();
        private EngineLoadState state;
        private OnnxBackend backend;
        private ModelIdentity identity;
        private Exception lastLoadError;
        private bool disposed;
        private int activeOperations;
        private TaskCompletionSource<bool> operationsDrained = CompletedDrain();
        private Task unloadTask;

        protected InferenceEngineRuntime(bool supportsParallelInference = false)
        {
            runGate = supportsParallelInference ? null : new SemaphoreSlim(1, 1);
        }

        public override EngineLoadState LoadState { get { lock (lifecycleLock) return state; } }
        public bool IsLoaded => LoadState == EngineLoadState.Loaded;
        public Exception LastLoadError => lastLoadError;
        public ModelIdentity LoadedModelIdentity => identity;
        protected OnnxBackend Backend => backend;
        protected CancellationToken LifetimeToken => lifetime.Token;

        internal async Task LoadAsync(OnnxBackend selectedBackend, ModelIdentity modelIdentity, CancellationToken cancellationToken)
        {
            lock (lifecycleLock)
            {
                ThrowIfDisposed();
                if (state != EngineLoadState.Unloaded) throw new OnnxRuntimeLifecycleException($"Runtime cannot load from state {state}.");
                backend = selectedBackend;
                identity = modelIdentity;
                state = EngineLoadState.Loading;
                lastLoadError = null;
            }
            using var linked = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken, lifetime.Token);
            try
            {
                await OnLoadAsync(linked.Token).ConfigureAwait(false);
                lock (lifecycleLock)
                {
                    ThrowIfDisposed();
                    state = EngineLoadState.Loaded;
                }
                Track();
            }
            catch (Exception exception)
            {
                lock (lifecycleLock)
                {
                    lastLoadError = exception;
                    if (!disposed) state = EngineLoadState.Failed;
                }
                throw;
            }
        }

        public async Task<TResult> RunAsync(TRequest request, CancellationToken cancellationToken = default)
        {
            BeginOperation();
            using var linked = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken, lifetime.Token);
            bool gateEntered = false;
            try
            {
                if (runGate != null)
                {
                    await runGate.WaitAsync(linked.Token).ConfigureAwait(false);
                    gateEntered = true;
                }
                return await OnRunAsync(request, linked.Token).ConfigureAwait(false);
            }
            finally
            {
                if (gateEntered) runGate.Release();
                EndOperation();
            }
        }


        public Task UnloadAsync(CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            lock (lifecycleLock)
            {
                if (state == EngineLoadState.Unloaded) return Task.CompletedTask;
                return unloadTask ??= UnloadCoreAsync();
            }
        }

        private async Task UnloadCoreAsync()
        {
            Task drain;
            lock (lifecycleLock)
            {
                if (state == EngineLoadState.Unloaded) return;
                state = EngineLoadState.Unloading;
                lifetime.Cancel();
                drain = operationsDrained.Task;
            }

            await drain.ConfigureAwait(false);
            try
            {
                await OnUnloadAsync(CancellationToken.None).ConfigureAwait(false);
            }
            finally
            {
                lock (lifecycleLock) state = EngineLoadState.Unloaded;
                Untrack();
            }
        }

        protected abstract Task OnLoadAsync(CancellationToken cancellationToken);
        protected abstract Task<TResult> OnRunAsync(TRequest request, CancellationToken cancellationToken);
        protected abstract Task OnUnloadAsync(CancellationToken cancellationToken);

        public override void Dispose()
        {
            Task cleanup;
            lock (lifecycleLock)
            {
                if (disposed) return;
                disposed = true;
                lifetime.Cancel();
                cleanup = unloadTask ??= UnloadCoreAsync();
            }

            try
            {
                cleanup.GetAwaiter().GetResult();
                OnDispose();
            }
            finally
            {
                lock (lifecycleLock) state = EngineLoadState.Unloaded;
                Untrack();
                runGate?.Dispose();
                lifetime.Dispose();
                backend = null; // caller-owned; never dispose it
            }
        }

        protected virtual void OnDispose() { }
        private void ThrowIfDisposed() { if (disposed) throw new ObjectDisposedException(GetType().Name); }

        private void BeginOperation()
        {
            lock (lifecycleLock)
            {
                ThrowIfDisposed();
                if (state != EngineLoadState.Loaded)
                    throw new OnnxRuntimeLifecycleException($"Runtime cannot run from state {state}.");
                if (activeOperations++ == 0)
                    operationsDrained = new TaskCompletionSource<bool>(TaskCreationOptions.RunContinuationsAsynchronously);
            }
        }

        private void EndOperation()
        {
            TaskCompletionSource<bool> drained = null;
            lock (lifecycleLock)
            {
                if (--activeOperations == 0) drained = operationsDrained;
            }
            drained?.TrySetResult(true);
        }

        private static TaskCompletionSource<bool> CompletedDrain()
        {
            var source = new TaskCompletionSource<bool>(TaskCreationOptions.RunContinuationsAsynchronously);
            source.SetResult(true);
            return source;
        }
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

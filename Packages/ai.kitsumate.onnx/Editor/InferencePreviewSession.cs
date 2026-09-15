using System;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;

namespace KitsuMate.Onnx.Editor
{
    /// <summary>One Inspector's runtime. Never owns or disposes the engine's backend.</summary>
    public sealed class InferencePreviewSession<TRequest, TResult>
    {
        private InferenceEngineRuntime<TRequest, TResult> runtime;
        private CancellationTokenSource operation;
        private string configuration;
        private bool closed;
        public bool IsBusy => operation != null;
        public bool IsLoaded => runtime != null && runtime.IsLoaded;
        public string Status { get; private set; } = "Unloaded";
        public double LoadMilliseconds { get; private set; }
        public double RunMilliseconds { get; private set; }

        public async Task<TResult> RunAsync(string fingerprint,
            Func<CancellationToken, Task<InferenceEngineRuntime<TRequest, TResult>>> load,
            Func<CancellationToken, Task<TRequest>> request)
        {
            if (closed) throw new ObjectDisposedException(nameof(InferencePreviewSession<TRequest, TResult>));
            if (IsBusy) throw new InvalidOperationException("An inference test is already running.");
            operation = new CancellationTokenSource();
            CancellationToken token = operation.Token;
            try
            {
                if (!IsLoaded || configuration != fingerprint)
                {
                    await ReleaseRuntimeAsync();
                    Status = "Loading";
                    var timer = Stopwatch.StartNew();
                    runtime = await load(token);
                    LoadMilliseconds = timer.Elapsed.TotalMilliseconds;
                    token.ThrowIfCancellationRequested();
                    configuration = fingerprint;
                }
                Status = "Preparing input";
                TRequest input = await request(token);
                token.ThrowIfCancellationRequested();
                Status = "Running";
                var runTimer = Stopwatch.StartNew();
                TResult result = await runtime.RunAsync(input, token);
                RunMilliseconds = runTimer.Elapsed.TotalMilliseconds;
                token.ThrowIfCancellationRequested();
                Status = "Ready";
                return result;
            }
            catch (OperationCanceledException) { Status = "Cancelled"; throw; }
            catch { Status = "Failed"; await ReleaseRuntimeAsync(); throw; }
            finally
            {
                try { if (closed || token.IsCancellationRequested) await ReleaseRuntimeAsync(); }
                finally { operation.Dispose(); operation = null; }
            }
        }

        public void Cancel()
        {
            if (operation == null) return;
            Status = "Cancelling";
            operation.Cancel();
        }

        public async Task UnloadAsync()
        {
            if (IsBusy) throw new InvalidOperationException("Wait for inference to finish before unloading.");
            operation = new CancellationTokenSource();
            Status = "Unloading";
            try { await ReleaseRuntimeAsync(); Status = "Unloaded"; }
            finally { operation.Dispose(); operation = null; }
        }

        public async Task CloseAsync()
        {
            closed = true;
            Cancel();
            // A pending load or run owns cleanup until its continuation finishes.
            if (!IsBusy) await ReleaseRuntimeAsync();
        }

        private async Task ReleaseRuntimeAsync()
        {
            var previous = runtime;
            runtime = null;
            configuration = null;
            if (previous == null) return;
            try { await previous.UnloadAsync(); }
            finally { previous.Dispose(); }
        }
    }
}

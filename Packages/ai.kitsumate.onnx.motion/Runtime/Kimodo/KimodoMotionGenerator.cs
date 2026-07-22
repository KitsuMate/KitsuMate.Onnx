using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading;
using KitsuMate.Onnx;
using UnityEngine;

namespace KitsuMate.Onnx.Motion.Kimodo
{
    /// <summary>
    /// First Kimodo v1 implementation. It consumes a pre-generated LLM2Vec embedding,
    /// runs the fixed batch-3 FP16 denoiser, applies separated CFG/DDIM on the host,
    /// and decodes a Unity Humanoid-compatible result.
    /// </summary>
    internal sealed class KimodoMotionGenerator : IDisposable
    {
        private readonly IOnnxModelSource _model;
        private readonly OnnxBackend _backend;
        private readonly bool _enableDiagnostics;
        private readonly object _lifecycleLock = new();
        private IOnnxSession _session;
        private int _running;
        private bool _disposed;

        public bool IsInitialized => _session != null && !_disposed;
        public KimodoModelCapabilities Capabilities => KimodoConstraintCompiler.SomaRpV11Capabilities;

        internal KimodoMotionGenerator(IOnnxModelSource model, OnnxBackend backend, bool enableDiagnostics = false)
        {
            _model = model ?? throw new ArgumentNullException(nameof(model));
            _backend = backend ?? throw new ArgumentNullException(nameof(backend));
            _enableDiagnostics = enableDiagnostics;
        }

        public async Awaitable InitializeAsync(CancellationToken cancellationToken = default)
        {
            await Awaitable.BackgroundThreadAsync();
            try
            {
                Initialize(cancellationToken);
            }
            finally
            {
                await Awaitable.MainThreadAsync();
            }
        }

        public async Awaitable<KimodoHumanoidMotion> GenerateAsync(
            KimodoTextEmbedding embedding,
            KimodoGenerationRequest request,
            CancellationToken cancellationToken = default)
        {
            await Awaitable.BackgroundThreadAsync();
            try
            {
                return Generate(embedding, request, cancellationToken);
            }
            finally
            {
                await Awaitable.MainThreadAsync();
            }
        }

        public async Awaitable<KimodoHumanoidMotion> GenerateAsync(
            KimodoTextEmbedding embedding,
            KimodoConditioning conditioning,
            KimodoGenerationRequest request,
            CancellationToken cancellationToken = default)
        {
            await Awaitable.BackgroundThreadAsync();
            try
            {
                return Generate(embedding, conditioning, request, cancellationToken);
            }
            finally
            {
                await Awaitable.MainThreadAsync();
            }
        }

        public void Initialize(CancellationToken cancellationToken = default)
        {
            lock (_lifecycleLock)
            {
                ThrowIfDisposed();
                if (_session != null) return;
                cancellationToken.ThrowIfCancellationRequested();
                var session = _backend.CreateSession(_model);
                try
                {
                    ValidateContract(session);
                    _session = session;
                }
                catch
                {
                    session.Dispose();
                    throw;
                }
            }
        }

        public KimodoHumanoidMotion Generate(
            KimodoTextEmbedding embedding,
            KimodoGenerationRequest request,
            CancellationToken cancellationToken = default)
        {
            request ??= new KimodoGenerationRequest();
            if (request.Constraints.HasConstraints)
                throw new InvalidOperationException("Compile semantic constraints and use the conditioning overload.");
            return Generate(embedding, KimodoConditioning.Empty, request, cancellationToken);
        }

        public KimodoHumanoidMotion Generate(
            KimodoTextEmbedding embedding,
            KimodoConditioning conditioning,
            KimodoGenerationRequest request,
            CancellationToken cancellationToken = default)
        {
            ThrowIfDisposed();
            if (_session == null) throw new InvalidOperationException("Initialize the Kimodo generator before use.");
            if (embedding == null) throw new ArgumentNullException(nameof(embedding));
            if (conditioning == null) throw new ArgumentNullException(nameof(conditioning));
            request ??= new KimodoGenerationRequest();
            if (request.FrameCount != KimodoTensorContract.Frames)
                throw new NotSupportedException($"This model requires exactly {KimodoTensorContract.Frames} frames.");
            if (conditioning.FrameCount != request.FrameCount)
                throw new ArgumentException("Conditioning frame count must match the generation request.", nameof(conditioning));
            if (Interlocked.CompareExchange(ref _running, 1, 0) != 0)
                throw new InvalidOperationException("Concurrent Kimodo generation is not supported by one generator instance.");

            try
            {
                return GenerateCore(embedding, conditioning, request, cancellationToken);
            }
            finally
            {
                Volatile.Write(ref _running, 0);
            }
        }

        private KimodoHumanoidMotion GenerateCore(
            KimodoTextEmbedding embedding,
            KimodoConditioning conditioning,
            KimodoGenerationRequest request,
            CancellationToken cancellationToken)
        {
            int motionLength = KimodoTensorContract.Frames * KimodoTensorContract.MotionDimension;
            int batchMotionLength = KimodoTensorContract.Batch * motionLength;
            var current = request.InitialNoiseOverride != null
                ? ValidateAndCopyInitialNoise(request.InitialNoiseOverride, motionLength)
                : CreateGaussianNoise(motionLength, request.Seed);
            var next = new float[motionLength];
            var predicted = new float[motionLength];
            var batchMotion = new float[batchMotionLength];
            var motionValid = new bool[KimodoTensorContract.Batch * KimodoTensorContract.Frames];
            Array.Fill(motionValid, true);
            var text = new float[KimodoTensorContract.Batch * KimodoTensorContract.TextTokens * KimodoTensorContract.TextDimension];
            var timesteps = new long[KimodoTensorContract.Batch];
            var headings = new[] { request.FirstHeadingRadians, request.FirstHeadingRadians, request.FirstHeadingRadians };
            var constraintMask = new bool[batchMotionLength];
            var observedMotion = new float[batchMotionLength];
            KimodoCfgBatchBuilder.Build(embedding, conditioning, text, constraintMask, observedMotion);
            var schedule = new KimodoDiffusionSchedule(request.DenoisingSteps);

            var inputs = new Dictionary<string, OnnxTensor>(7)
            {
                [KimodoTensorContract.Motion] = null,
                [KimodoTensorContract.MotionValid] = OnnxTensor.FromArray(motionValid, new[] { 3, 60 }),
                [KimodoTensorContract.TextEmbedding] = OnnxTensor.FromArray(text, new[] { 3, 50, 4096 }),
                [KimodoTensorContract.Timestep] = null,
                [KimodoTensorContract.FirstHeading] = OnnxTensor.FromArray(headings, new[] { 3 }),
                [KimodoTensorContract.ConstraintMask] = OnnxTensor.FromArray(constraintMask, new[] { 3, 60, 369 }),
                [KimodoTensorContract.ObservedMotion] = OnnxTensor.FromArray(observedMotion, new[] { 3, 60, 369 }),
            };

            var inferenceWatch = Stopwatch.StartNew();
            for (int samplingIndex = request.DenoisingSteps - 1; samplingIndex >= 0; samplingIndex--)
            {
                cancellationToken.ThrowIfCancellationRequested();
                Array.Copy(current, 0, batchMotion, 0, motionLength);
                Array.Copy(current, 0, batchMotion, motionLength, motionLength);
                Array.Copy(current, 0, batchMotion, motionLength * 2, motionLength);
                long mappedTimestep = schedule.GetModelTimestep(samplingIndex);
                timesteps[0] = timesteps[1] = timesteps[2] = mappedTimestep;
                inputs[KimodoTensorContract.Motion] = OnnxTensor.FromArray(batchMotion, new[] { 3, 60, 369 });
                inputs[KimodoTensorContract.Timestep] = OnnxTensor.FromArray(timesteps, new[] { 3 });

#pragma warning disable CS0618
                var outputs = _session.Run(inputs);
#pragma warning restore CS0618
                if (!outputs.TryGetValue(KimodoTensorContract.PredictedCleanMotion, out var output))
                    throw new InvalidOperationException("Kimodo denoiser did not produce predicted_clean_motion.");
                var batchPrediction = output.AsFloatArray();
                if (batchPrediction.Length != batchMotionLength)
                    throw new InvalidOperationException("Kimodo denoiser output has the wrong shape.");

                int constraintOffset = motionLength;
                int unconditionalOffset = motionLength * 2;
                for (int i = 0; i < motionLength; i++)
                {
                    float unconditional = batchPrediction[unconditionalOffset + i];
                    predicted[i] = unconditional +
                                   request.TextGuidance * (batchPrediction[i] - unconditional) +
                                   request.ConstraintGuidance * (batchPrediction[constraintOffset + i] - unconditional);
                }

                schedule.Step(current, predicted, samplingIndex, next);
                (current, next) = (next, current);
            }
            inferenceWatch.Stop();

            for (int i = 0; i < current.Length; i++)
            {
                if (float.IsNaN(current[i]) || float.IsInfinity(current[i]))
                    throw new InvalidOperationException($"Kimodo produced a non-finite motion value at index {i}.");
            }

            var decodeWatch = Stopwatch.StartNew();
            var retainedMotion = _enableDiagnostics ? (float[])current.Clone() : null;
            var decoded = KimodoMotionDecoder.Decode(current, KimodoTensorContract.Frames, diagnostics: null);
            decodeWatch.Stop();
            if (!_enableDiagnostics) return decoded;

            var diagnostics = new KimodoGenerationDiagnostics(
                inferenceWatch.ElapsedMilliseconds,
                decodeWatch.ElapsedMilliseconds,
                retainedMotion);
            return new KimodoHumanoidMotion(
                decoded.FrameCount,
                decoded.FramesPerSecond,
                decoded.BoneRotationDeltas,
                decoded.RootPositions,
                decoded.RootRotations,
                decoded.BoneAvailability,
                diagnostics,
                decoded.SmoothedRootPositions,
                decoded.BoneGlobalRotationDeltas);
        }

        private static float[] CreateGaussianNoise(int length, int seed)
        {
            var random = new System.Random(seed);
            var result = new float[length];
            int index = 0;
            while (index < length)
            {
                double u1 = Math.Max(double.Epsilon, random.NextDouble());
                double u2 = random.NextDouble();
                double radius = Math.Sqrt(-2.0 * Math.Log(u1));
                double angle = 2.0 * Math.PI * u2;
                result[index++] = (float)(radius * Math.Cos(angle));
                if (index < length) result[index++] = (float)(radius * Math.Sin(angle));
            }
            return result;
        }

        private static float[] ValidateAndCopyInitialNoise(float[] source, int expectedLength)
        {
            if (source.Length != expectedLength)
                throw new ArgumentException($"Initial noise must contain {expectedLength} values.");
            var copy = (float[])source.Clone();
            for (int i = 0; i < copy.Length; i++)
            {
                if (float.IsNaN(copy[i]) || float.IsInfinity(copy[i]))
                    throw new ArgumentException($"Initial noise contains a non-finite value at index {i}.");
            }
            return copy;
        }

        private static void ValidateContract(IOnnxSession session)
        {
            string[] requiredInputs =
            {
                KimodoTensorContract.Motion,
                KimodoTensorContract.MotionValid,
                KimodoTensorContract.TextEmbedding,
                KimodoTensorContract.Timestep,
                KimodoTensorContract.FirstHeading,
                KimodoTensorContract.ConstraintMask,
                KimodoTensorContract.ObservedMotion,
            };
            foreach (string name in requiredInputs)
            {
                if (!Contains(session.InputNames, name))
                    throw new InvalidOperationException($"Kimodo model is missing required input '{name}'.");
            }
            if (!Contains(session.OutputNames, KimodoTensorContract.PredictedCleanMotion))
                throw new InvalidOperationException("Kimodo model is missing predicted_clean_motion output.");
        }

        private static bool Contains(IReadOnlyList<string> names, string expected)
        {
            for (int i = 0; i < names.Count; i++) if (names[i] == expected) return true;
            return false;
        }

        public void Dispose()
        {
            lock (_lifecycleLock)
            {
                if (_disposed) return;
                if (Volatile.Read(ref _running) != 0)
                    throw new InvalidOperationException("Cannot dispose a Kimodo generator while inference is running.");
                _disposed = true;
                _session?.Dispose();
                _session = null;
            }
        }

        private void ThrowIfDisposed()
        {
            if (_disposed) throw new ObjectDisposedException(nameof(KimodoMotionGenerator));
        }
    }
}

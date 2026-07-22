using System;

namespace KitsuMate.Onnx.Motion
{
    /// <summary>Normalized, model-ready Kimodo constraint tensors for one sample.</summary>
    public sealed class KimodoConditioning
    {
        public const int DefaultFrameCount = 60;
        public const int FeatureCount = 369;

        private readonly float[] _observedMotion;
        private readonly bool[] _motionMask;

        public int FrameCount { get; }
        public ReadOnlyMemory<float> ObservedMotion => _observedMotion;
        public ReadOnlyMemory<bool> MotionMask => _motionMask;
        public bool HasConstraints { get; }

        public static KimodoConditioning Empty { get; } = new KimodoConditioning(
            new float[DefaultFrameCount * FeatureCount],
            new bool[DefaultFrameCount * FeatureCount],
            DefaultFrameCount,
            copy: false);

        public KimodoConditioning(float[] observedMotion, bool[] motionMask, int frameCount = DefaultFrameCount, bool copy = true)
        {
            if (observedMotion == null) throw new ArgumentNullException(nameof(observedMotion));
            if (motionMask == null) throw new ArgumentNullException(nameof(motionMask));
            if (frameCount <= 0) throw new ArgumentOutOfRangeException(nameof(frameCount));
            int expected = checked(frameCount * FeatureCount);
            if (observedMotion.Length != expected || motionMask.Length != expected)
                throw new ArgumentException($"Conditioning arrays must each contain {expected} values.");

            bool any = false;
            for (int i = 0; i < expected; i++)
            {
                if (float.IsNaN(observedMotion[i]) || float.IsInfinity(observedMotion[i]))
                    throw new ArgumentException($"Observed motion contains a non-finite value at index {i}.");
                any |= motionMask[i];
            }
            FrameCount = frameCount;
            _observedMotion = copy ? (float[])observedMotion.Clone() : observedMotion;
            _motionMask = copy ? (bool[])motionMask.Clone() : motionMask;
            HasConstraints = any;
        }
    }
}

using System;

namespace KitsuMate.Onnx.Motion
{
    /// <summary>Kimodo v1 generation controls and optional semantic constraints.</summary>
    public sealed class KimodoGenerationRequest
    {
        public int Seed { get; }
        public int DenoisingSteps { get; }
        public float TextGuidance { get; }
        public float ConstraintGuidance { get; }
        public float FirstHeadingRadians { get; }
        public int FrameCount { get; }
        public KimodoConstraintSet Constraints { get; }

        // Test/reference hook. Production generation leaves this null and uses Seed.
        internal float[] InitialNoiseOverride { get; set; }

        public KimodoGenerationRequest(
            int seed = 0,
            int denoisingSteps = 100,
            float textGuidance = 2f,
            float constraintGuidance = 2f,
            float firstHeadingRadians = 0f,
            int frameCount = 60,
            KimodoConstraintSet constraints = null)
        {
            if (denoisingSteps < 2 || denoisingSteps > 1000)
                throw new ArgumentOutOfRangeException(nameof(denoisingSteps), "Denoising steps must be in [2, 1000].");
            if (frameCount < 2)
                throw new ArgumentOutOfRangeException(nameof(frameCount));
            if (!IsFinite(textGuidance) || !IsFinite(constraintGuidance) || !IsFinite(firstHeadingRadians))
                throw new ArgumentException("Guidance and heading values must be finite.");

            Seed = seed;
            DenoisingSteps = denoisingSteps;
            TextGuidance = textGuidance;
            ConstraintGuidance = constraintGuidance;
            FirstHeadingRadians = firstHeadingRadians;
            FrameCount = frameCount;
            Constraints = constraints ?? KimodoConstraintSet.Empty;
        }

        private static bool IsFinite(float value) => !float.IsNaN(value) && !float.IsInfinity(value);
    }

}

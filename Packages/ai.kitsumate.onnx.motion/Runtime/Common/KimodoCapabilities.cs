using System;

namespace KitsuMate.Onnx.Motion
{
    [Flags]
    public enum KimodoConstraintCapabilities
    {
        None = 0,
        RootPosition2D = 1 << 0,
        RootHeading = 1 << 1,
        FullBodyPose = 1 << 2,
        LeftHand = 1 << 3,
        RightHand = 1 << 4,
        LeftFoot = 1 << 5,
        RightFoot = 1 << 6,
    }

    public sealed class KimodoModelCapabilities
    {
        public int FrameCount { get; }
        public float FramesPerSecond { get; }
        public int MotionFeatureCount { get; }
        public int InternalJointCount { get; }
        public int TextEmbeddingDimension { get; }
        public bool SupportsSeparatedGuidance { get; }
        public KimodoConstraintCapabilities Constraints { get; }

        public KimodoModelCapabilities(
            int frameCount,
            float framesPerSecond,
            int motionFeatureCount,
            int internalJointCount,
            int textEmbeddingDimension,
            bool supportsSeparatedGuidance,
            KimodoConstraintCapabilities constraints)
        {
            FrameCount = frameCount;
            FramesPerSecond = framesPerSecond;
            MotionFeatureCount = motionFeatureCount;
            InternalJointCount = internalJointCount;
            TextEmbeddingDimension = textEmbeddingDimension;
            SupportsSeparatedGuidance = supportsSeparatedGuidance;
            Constraints = constraints;
        }
    }
}

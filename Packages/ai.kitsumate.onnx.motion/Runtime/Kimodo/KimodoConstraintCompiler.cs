using System;
using UnityEngine;

namespace KitsuMate.Onnx.Motion.Kimodo
{
    /// <summary>Compiles canonical SOMA-30 constraints into Kimodo's normalized 369-feature tensors.</summary>
    public sealed class KimodoConstraintCompiler : IKimodoConstraintCompiler
    {
        private const int SmoothRootOffset = 0;
        private const int HeadingOffset = 3;
        private const int LocalPositionOffset = 5;
        private const int GlobalRotationOffset = 95;
        private const float ConflictTolerance = 1e-5f;

        public static KimodoModelCapabilities SomaRpV11Capabilities { get; } = new KimodoModelCapabilities(
            KimodoTensorContract.Frames,
            KimodoTensorContract.FramesPerSecond,
            KimodoTensorContract.MotionDimension,
            KimodoTensorContract.JointCount,
            KimodoTensorContract.TextDimension,
            supportsSeparatedGuidance: true,
            KimodoConstraintCapabilities.RootPosition2D |
            KimodoConstraintCapabilities.RootHeading |
            KimodoConstraintCapabilities.FullBodyPose |
            KimodoConstraintCapabilities.LeftHand |
            KimodoConstraintCapabilities.RightHand |
            KimodoConstraintCapabilities.LeftFoot |
            KimodoConstraintCapabilities.RightFoot);

        public KimodoConditioning Compile(KimodoConstraintSet constraints, int frameCount = KimodoConditioning.DefaultFrameCount)
        {
            constraints ??= KimodoConstraintSet.Empty;
            if (frameCount != KimodoTensorContract.Frames)
                throw new NotSupportedException($"Kimodo SOMA RP v1.1 requires {KimodoTensorContract.Frames} frames.");

            int length = frameCount * KimodoTensorContract.MotionDimension;
            var raw = new float[length];
            var mask = new bool[length];
            var explicitRootPositions = new bool[frameCount];
            var explicitHeadings = new bool[frameCount];

            // Explicit root controls are authored independently from the reference pose carried by
            // full-body/end-effector constraints. Compile them first so they authoritatively replace
            // the derived context features regardless of constraint ordering.
            foreach (IKimodoConstraint constraint in constraints.Constraints)
            {
                if (constraint.Space != KimodoConstraintSpace.CanonicalYUpMeters)
                    throw new NotSupportedException($"Unsupported constraint space {constraint.Space}.");
                if (constraint is KimodoRootConstraint root)
                    CompileRoot(root, raw, mask, frameCount, explicitRootPositions, explicitHeadings);
            }

            foreach (IKimodoConstraint constraint in constraints.Constraints)
            {
                switch (constraint)
                {
                    case KimodoRootConstraint:
                        break;
                    case KimodoFullBodyConstraint fullBody:
                        CompileFullBody(fullBody, raw, mask, frameCount,
                            explicitRootPositions, explicitHeadings);
                        break;
                    case KimodoEndEffectorConstraint endEffector:
                        CompileEndEffector(endEffector, raw, mask, frameCount,
                            explicitRootPositions, explicitHeadings);
                        break;
                    default:
                        throw new NotSupportedException($"Unsupported Kimodo constraint type {constraint.GetType().Name}.");
                }
            }

            for (int frame = 0; frame < frameCount; frame++)
            {
                int offset = frame * KimodoTensorContract.MotionDimension;
                for (int feature = 0; feature < KimodoTensorContract.MotionDimension; feature++)
                    raw[offset + feature] = (raw[offset + feature] - KimodoSomaRuntimeData.Mean[feature]) /
                                            KimodoSomaRuntimeData.Scale[feature];
            }
            return new KimodoConditioning(raw, mask, frameCount, copy: false);
        }

        private static void CompileRoot(
            KimodoRootConstraint constraint,
            float[] raw,
            bool[] mask,
            int frames,
            bool[] explicitRootPositions,
            bool[] explicitHeadings)
        {
            ReadOnlySpan<int> indices = constraint.FrameIndices.Span;
            ReadOnlySpan<Vector2> positions = constraint.PositionsXZ.Span;
            ReadOnlySpan<Vector2> headings = constraint.Headings.Span;
            for (int i = 0; i < indices.Length; i++)
            {
                int frame = ValidateFrame(indices[i], frames);
                Set(raw, mask, frame, SmoothRootOffset, positions[i].x);
                Set(raw, mask, frame, SmoothRootOffset + 2, positions[i].y);
                explicitRootPositions[frame] = true;
                if (constraint.HasHeadings)
                {
                    Vector2 heading = NormalizeHeading(headings[i]);
                    Set(raw, mask, frame, HeadingOffset, heading.x);
                    Set(raw, mask, frame, HeadingOffset + 1, heading.y);
                    explicitHeadings[frame] = true;
                }
            }
        }

        private static void CompileFullBody(
            KimodoFullBodyConstraint constraint,
            float[] raw,
            bool[] mask,
            int frames,
            bool[] explicitRootPositions,
            bool[] explicitHeadings)
        {
            ReadOnlySpan<int> indices = constraint.FrameIndices.Span;
            ReadOnlySpan<Vector3> positions = constraint.GlobalJointPositions.Span;
            ReadOnlySpan<Vector2> smoothRoots = constraint.SmoothedRootPositionsXZ.Span;
            for (int sample = 0; sample < indices.Length; sample++)
            {
                int frame = ValidateFrame(indices[sample], frames);
                int poseOffset = sample * KimodoTensorContract.JointCount;
                Vector3 root = positions[poseOffset + (int)KimodoJoint.Hips];
                Vector2 smoothRoot = constraint.HasSmoothedRootPositions
                    ? smoothRoots[sample]
                    : new Vector2(root.x, root.z);
                SetRootAndHeading(raw, mask, frame, positions, poseOffset, smoothRoot, root.y,
                    explicitRootPositions[frame], explicitHeadings[frame]);
                for (int joint = 0; joint < KimodoTensorContract.JointCount; joint++)
                    SetLocalPosition(raw, mask, frame, joint, positions[poseOffset + joint], smoothRoot);
            }
        }

        private static void CompileEndEffector(
            KimodoEndEffectorConstraint constraint,
            float[] raw,
            bool[] mask,
            int frames,
            bool[] explicitRootPositions,
            bool[] explicitHeadings)
        {
            ReadOnlySpan<int> indices = constraint.FrameIndices.Span;
            ReadOnlySpan<Vector3> positions = constraint.GlobalJointPositions.Span;
            ReadOnlySpan<Quaternion> rotations = constraint.GlobalJointRotations.Span;
            ReadOnlySpan<Vector2> smoothRoots = constraint.SmoothedRootPositionsXZ.Span;
            for (int sample = 0; sample < indices.Length; sample++)
            {
                int frame = ValidateFrame(indices[sample], frames);
                int poseOffset = sample * KimodoTensorContract.JointCount;
                Vector3 root = positions[poseOffset + (int)KimodoJoint.Hips];
                Vector2 smoothRoot = constraint.HasSmoothedRootPositions
                    ? smoothRoots[sample]
                    : new Vector2(root.x, root.z);
                SetRootAndHeading(raw, mask, frame, positions, poseOffset, smoothRoot, root.y,
                    explicitRootPositions[frame], explicitHeadings[frame]);

                if ((constraint.Effectors & KimodoEndEffectors.LeftHand) != 0)
                    SetEffector(raw, mask, frame, positions, rotations, poseOffset, smoothRoot, 13, 15);
                if ((constraint.Effectors & KimodoEndEffectors.RightHand) != 0)
                    SetEffector(raw, mask, frame, positions, rotations, poseOffset, smoothRoot, 19, 21);
                if ((constraint.Effectors & KimodoEndEffectors.LeftFoot) != 0)
                    SetEffector(raw, mask, frame, positions, rotations, poseOffset, smoothRoot, 24, 25);
                if ((constraint.Effectors & KimodoEndEffectors.RightFoot) != 0)
                    SetEffector(raw, mask, frame, positions, rotations, poseOffset, smoothRoot, 28, 29);
            }
        }

        private static void SetRootAndHeading(
            float[] raw, bool[] mask, int frame, ReadOnlySpan<Vector3> positions, int poseOffset,
            Vector2 smoothRoot, float rootY, bool hasExplicitRootPosition, bool hasExplicitHeading)
        {
            if (!hasExplicitRootPosition)
            {
                Set(raw, mask, frame, SmoothRootOffset, smoothRoot.x);
                Set(raw, mask, frame, SmoothRootOffset + 2, smoothRoot.y);
            }
            Set(raw, mask, frame, SmoothRootOffset + 1, rootY);
            if (hasExplicitHeading) return;
            Vector3 rightHip = positions[poseOffset + (int)KimodoJoint.RightLeg];
            Vector3 leftHip = positions[poseOffset + (int)KimodoJoint.LeftLeg];
            Vector3 difference = rightHip - leftHip;
            float angle = Mathf.Atan2(difference.z, -difference.x);
            Set(raw, mask, frame, HeadingOffset, Mathf.Cos(angle));
            Set(raw, mask, frame, HeadingOffset + 1, Mathf.Sin(angle));
        }

        private static void SetEffector(
            float[] raw, bool[] mask, int frame,
            ReadOnlySpan<Vector3> positions, ReadOnlySpan<Quaternion> rotations,
            int poseOffset, Vector2 smoothRoot, int baseJoint, int endJoint)
        {
            SetLocalPosition(raw, mask, frame, baseJoint, positions[poseOffset + baseJoint], smoothRoot);
            SetLocalPosition(raw, mask, frame, endJoint, positions[poseOffset + endJoint], smoothRoot);
            SetGlobalRotation(raw, mask, frame, baseJoint, rotations[poseOffset + baseJoint]);
        }

        private static void SetLocalPosition(float[] raw, bool[] mask, int frame, int joint, Vector3 global, Vector2 smoothRoot)
        {
            int feature = LocalPositionOffset + joint * 3;
            Set(raw, mask, frame, feature, global.x - smoothRoot.x);
            Set(raw, mask, frame, feature + 1, global.y);
            Set(raw, mask, frame, feature + 2, global.z - smoothRoot.y);
        }

        private static void SetGlobalRotation(float[] raw, bool[] mask, int frame, int joint, Quaternion quaternion)
        {
            float magnitude = Mathf.Sqrt(quaternion.x * quaternion.x + quaternion.y * quaternion.y +
                                        quaternion.z * quaternion.z + quaternion.w * quaternion.w);
            if (magnitude < 1e-8f) throw new ArgumentException("End-effector rotations must be non-zero quaternions.");
            float x = quaternion.x / magnitude, y = quaternion.y / magnitude;
            float z = quaternion.z / magnitude, w = quaternion.w / magnitude;
            int feature = GlobalRotationOffset + joint * 6;
            // First two columns of the standard quaternion rotation matrix.
            Set(raw, mask, frame, feature, 1f - 2f * (y * y + z * z));
            Set(raw, mask, frame, feature + 1, 2f * (x * y + z * w));
            Set(raw, mask, frame, feature + 2, 2f * (x * z - y * w));
            Set(raw, mask, frame, feature + 3, 2f * (x * y - z * w));
            Set(raw, mask, frame, feature + 4, 1f - 2f * (x * x + z * z));
            Set(raw, mask, frame, feature + 5, 2f * (y * z + x * w));
        }

        private static Vector2 NormalizeHeading(Vector2 heading)
        {
            float magnitude = heading.magnitude;
            if (magnitude < 1e-8f) throw new ArgumentException("Root heading vectors must be non-zero.");
            return heading / magnitude;
        }

        private static int ValidateFrame(int frame, int frames)
        {
            if ((uint)frame >= (uint)frames) throw new ArgumentOutOfRangeException(nameof(frame), $"Constraint frame {frame} is outside [0,{frames - 1}].");
            return frame;
        }

        private static void Set(float[] raw, bool[] mask, int frame, int feature, float value)
        {
            int index = frame * KimodoTensorContract.MotionDimension + feature;
            if (mask[index] && Mathf.Abs(raw[index] - value) > ConflictTolerance)
                throw new InvalidOperationException($"Conflicting Kimodo constraints at frame {frame}, feature {feature}.");
            raw[index] = value;
            mask[index] = true;
        }
    }
}

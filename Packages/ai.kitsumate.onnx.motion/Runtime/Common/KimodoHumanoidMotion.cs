using System;
using UnityEngine;

namespace KitsuMate.Onnx.Motion
{
    /// <summary>Frame-major Unity Humanoid motion decoded from Kimodo SOMA output.</summary>
    public sealed class KimodoHumanoidMotion
    {
        public int FrameCount { get; }
        public float FramesPerSecond { get; }
        public int BoneCount { get; }
        public Quaternion[] BoneRotationDeltas { get; }
        /// <summary>Optional heading-relative global SOMA rotations mapped to Humanoid bones.</summary>
        public Quaternion[] BoneGlobalRotationDeltas { get; }
        /// <summary>Unity-space planar trajectory corresponding directly to Kimodo's constrained smoothed-root features.</summary>
        public Vector3[] SmoothedRootPositions { get; }
        public Vector3[] RootPositions { get; }
        public Quaternion[] RootRotations { get; }
        public bool[] BoneAvailability { get; }
        public KimodoGenerationDiagnostics Diagnostics { get; }

        public KimodoHumanoidMotion(
            int frameCount,
            float framesPerSecond,
            Quaternion[] boneRotationDeltas,
            Vector3[] rootPositions,
            Quaternion[] rootRotations,
            bool[] boneAvailability,
            KimodoGenerationDiagnostics diagnostics = null,
            Vector3[] smoothedRootPositions = null,
            Quaternion[] boneGlobalRotationDeltas = null)
        {
            FrameCount = frameCount;
            FramesPerSecond = framesPerSecond;
            BoneCount = (int)HumanBodyBones.LastBone;
            BoneRotationDeltas = boneRotationDeltas ?? throw new ArgumentNullException(nameof(boneRotationDeltas));
            BoneGlobalRotationDeltas = boneGlobalRotationDeltas;
            RootPositions = rootPositions ?? throw new ArgumentNullException(nameof(rootPositions));
            SmoothedRootPositions = smoothedRootPositions ?? rootPositions;
            RootRotations = rootRotations ?? throw new ArgumentNullException(nameof(rootRotations));
            BoneAvailability = boneAvailability ?? throw new ArgumentNullException(nameof(boneAvailability));
            Diagnostics = diagnostics;

            if (boneRotationDeltas.Length != frameCount * BoneCount)
                throw new ArgumentException("Bone rotation array has the wrong length.", nameof(boneRotationDeltas));
            if (boneGlobalRotationDeltas != null && boneGlobalRotationDeltas.Length != frameCount * BoneCount)
                throw new ArgumentException("Global bone rotation array has the wrong length.", nameof(boneGlobalRotationDeltas));
            if (rootPositions.Length != frameCount || rootRotations.Length != frameCount)
                throw new ArgumentException("Root arrays must contain one value per frame.");
            if (SmoothedRootPositions.Length != frameCount)
                throw new ArgumentException("Smoothed-root array must contain one value per frame.", nameof(smoothedRootPositions));
            if (boneAvailability.Length != BoneCount)
                throw new ArgumentException("Bone availability array has the wrong length.", nameof(boneAvailability));
        }

        public Quaternion GetBoneRotation(int frame, HumanBodyBones bone)
        {
            if ((uint)frame >= (uint)FrameCount) throw new ArgumentOutOfRangeException(nameof(frame));
            int boneIndex = (int)bone;
            if ((uint)boneIndex >= (uint)BoneCount) throw new ArgumentOutOfRangeException(nameof(bone));
            return BoneRotationDeltas[frame * BoneCount + boneIndex];
        }

        public Quaternion GetBoneGlobalRotation(int frame, HumanBodyBones bone)
        {
            if (BoneGlobalRotationDeltas == null) throw new InvalidOperationException("This motion has no global bone rotations.");
            if ((uint)frame >= (uint)FrameCount) throw new ArgumentOutOfRangeException(nameof(frame));
            int boneIndex = (int)bone;
            if ((uint)boneIndex >= (uint)BoneCount) throw new ArgumentOutOfRangeException(nameof(bone));
            return BoneGlobalRotationDeltas[frame * BoneCount + boneIndex];
        }
    }

    public sealed class KimodoGenerationDiagnostics
    {
        public long InferenceMilliseconds { get; }
        public long DecodeMilliseconds { get; }
        public float[] NormalizedMotion { get; }

        public KimodoGenerationDiagnostics(long inferenceMilliseconds, long decodeMilliseconds, float[] normalizedMotion)
        {
            InferenceMilliseconds = inferenceMilliseconds;
            DecodeMilliseconds = decodeMilliseconds;
            NormalizedMotion = normalizedMotion;
        }
    }
}

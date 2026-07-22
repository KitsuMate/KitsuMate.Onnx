using System;
using System.Collections.Generic;
using UnityEngine;

namespace KitsuMate.Onnx.Motion
{
    public enum KimodoConstraintSpace
    {
        CanonicalYUpMeters = 0,
    }

    public enum KimodoJoint
    {
        Hips, Spine1, Spine2, Chest, Neck1, Neck2, Head, Jaw, LeftEye, RightEye,
        LeftShoulder, LeftArm, LeftForeArm, LeftHand, LeftHandThumbEnd, LeftHandMiddleEnd,
        RightShoulder, RightArm, RightForeArm, RightHand, RightHandThumbEnd, RightHandMiddleEnd,
        LeftLeg, LeftShin, LeftFoot, LeftToeBase,
        RightLeg, RightShin, RightFoot, RightToeBase,
    }

    [Flags]
    public enum KimodoEndEffectors
    {
        None = 0,
        LeftHand = 1 << 0,
        RightHand = 1 << 1,
        LeftFoot = 1 << 2,
        RightFoot = 1 << 3,
    }

    public interface IKimodoConstraint
    {
        KimodoConstraintSpace Space { get; }
        ReadOnlyMemory<int> FrameIndices { get; }
    }

    public sealed class KimodoRootConstraint : IKimodoConstraint
    {
        private readonly int[] _frames;
        private readonly Vector2[] _positions;
        private readonly Vector2[] _headings;

        public KimodoConstraintSpace Space => KimodoConstraintSpace.CanonicalYUpMeters;
        public ReadOnlyMemory<int> FrameIndices => _frames;
        public ReadOnlyMemory<Vector2> PositionsXZ => _positions;
        public ReadOnlyMemory<Vector2> Headings => _headings;
        public bool HasHeadings => _headings != null;

        public KimodoRootConstraint(int[] frameIndices, Vector2[] positionsXZ, Vector2[] headings = null)
        {
            ValidateFramesAndCount(frameIndices, positionsXZ?.Length ?? -1);
            if (headings != null && headings.Length != frameIndices.Length)
                throw new ArgumentException("Heading count must match frame count.", nameof(headings));
            _frames = (int[])frameIndices.Clone();
            _positions = (Vector2[])positionsXZ.Clone();
            _headings = headings == null ? null : (Vector2[])headings.Clone();
            ValidateFinite(_positions, nameof(positionsXZ));
            if (_headings != null) ValidateFinite(_headings, nameof(headings));
        }

        internal static void ValidateFramesAndCount(int[] frames, int count)
        {
            if (frames == null) throw new ArgumentNullException(nameof(frames));
            if (count != frames.Length) throw new ArgumentException("Value count must match frame count.");
            for (int i = 0; i < frames.Length; i++) if (frames[i] < 0) throw new ArgumentOutOfRangeException(nameof(frames));
        }

        internal static void ValidateFinite(Vector2[] values, string name)
        {
            for (int i = 0; i < values.Length; i++)
                if (!Finite(values[i].x) || !Finite(values[i].y)) throw new ArgumentException($"{name} contains a non-finite value.", name);
        }

        internal static void ValidateFinite(Vector3[] values, string name)
        {
            for (int i = 0; i < values.Length; i++)
                if (!Finite(values[i].x) || !Finite(values[i].y) || !Finite(values[i].z)) throw new ArgumentException($"{name} contains a non-finite value.", name);
        }

        internal static void ValidateFinite(Quaternion[] values, string name)
        {
            for (int i = 0; i < values.Length; i++)
                if (!Finite(values[i].x) || !Finite(values[i].y) || !Finite(values[i].z) || !Finite(values[i].w)) throw new ArgumentException($"{name} contains a non-finite value.", name);
        }

        private static bool Finite(float value) => !float.IsNaN(value) && !float.IsInfinity(value);
    }

    /// <summary>One or more full-body keyframes expressed as SOMA-30 global joint positions.</summary>
    public sealed class KimodoFullBodyConstraint : IKimodoConstraint
    {
        private readonly int[] _frames;
        private readonly Vector3[] _globalPositions;
        private readonly Vector2[] _smoothRoot;

        public KimodoConstraintSpace Space => KimodoConstraintSpace.CanonicalYUpMeters;
        public ReadOnlyMemory<int> FrameIndices => _frames;
        public ReadOnlyMemory<Vector3> GlobalJointPositions => _globalPositions;
        public ReadOnlyMemory<Vector2> SmoothedRootPositionsXZ => _smoothRoot;
        public bool HasSmoothedRootPositions => _smoothRoot != null;

        public KimodoFullBodyConstraint(int[] frameIndices, Vector3[] globalJointPositions, Vector2[] smoothedRootPositionsXZ = null)
        {
            KimodoRootConstraint.ValidateFramesAndCount(frameIndices, globalJointPositions == null ? -1 : globalJointPositions.Length / 30);
            if (globalJointPositions.Length != frameIndices.Length * 30)
                throw new ArgumentException("Full-body positions must have shape [N,30].", nameof(globalJointPositions));
            if (smoothedRootPositionsXZ != null && smoothedRootPositionsXZ.Length != frameIndices.Length)
                throw new ArgumentException("Smoothed-root count must match frame count.", nameof(smoothedRootPositionsXZ));
            _frames = (int[])frameIndices.Clone();
            _globalPositions = (Vector3[])globalJointPositions.Clone();
            _smoothRoot = smoothedRootPositionsXZ == null ? null : (Vector2[])smoothedRootPositionsXZ.Clone();
            KimodoRootConstraint.ValidateFinite(_globalPositions, nameof(globalJointPositions));
            if (_smoothRoot != null) KimodoRootConstraint.ValidateFinite(_smoothRoot, nameof(smoothedRootPositionsXZ));
        }
    }

    /// <summary>End-effector keyframes backed by complete SOMA-30 global pose data.</summary>
    public sealed class KimodoEndEffectorConstraint : IKimodoConstraint
    {
        private readonly int[] _frames;
        private readonly Vector3[] _globalPositions;
        private readonly Quaternion[] _globalRotations;
        private readonly Vector2[] _smoothRoot;

        public KimodoConstraintSpace Space => KimodoConstraintSpace.CanonicalYUpMeters;
        public ReadOnlyMemory<int> FrameIndices => _frames;
        public KimodoEndEffectors Effectors { get; }
        public ReadOnlyMemory<Vector3> GlobalJointPositions => _globalPositions;
        public ReadOnlyMemory<Quaternion> GlobalJointRotations => _globalRotations;
        public ReadOnlyMemory<Vector2> SmoothedRootPositionsXZ => _smoothRoot;
        public bool HasSmoothedRootPositions => _smoothRoot != null;

        public KimodoEndEffectorConstraint(
            int[] frameIndices,
            KimodoEndEffectors effectors,
            Vector3[] globalJointPositions,
            Quaternion[] globalJointRotations,
            Vector2[] smoothedRootPositionsXZ = null)
        {
            if (effectors == KimodoEndEffectors.None) throw new ArgumentException("Select at least one end effector.", nameof(effectors));
            int frames = frameIndices?.Length ?? -1;
            KimodoRootConstraint.ValidateFramesAndCount(frameIndices, globalJointPositions == null ? -1 : globalJointPositions.Length / 30);
            if (globalJointPositions.Length != frames * 30 || globalJointRotations == null || globalJointRotations.Length != frames * 30)
                throw new ArgumentException("End-effector pose arrays must have shape [N,30].");
            if (smoothedRootPositionsXZ != null && smoothedRootPositionsXZ.Length != frames)
                throw new ArgumentException("Smoothed-root count must match frame count.", nameof(smoothedRootPositionsXZ));
            _frames = (int[])frameIndices.Clone();
            Effectors = effectors;
            _globalPositions = (Vector3[])globalJointPositions.Clone();
            _globalRotations = (Quaternion[])globalJointRotations.Clone();
            _smoothRoot = smoothedRootPositionsXZ == null ? null : (Vector2[])smoothedRootPositionsXZ.Clone();
            KimodoRootConstraint.ValidateFinite(_globalPositions, nameof(globalJointPositions));
            KimodoRootConstraint.ValidateFinite(_globalRotations, nameof(globalJointRotations));
            if (_smoothRoot != null) KimodoRootConstraint.ValidateFinite(_smoothRoot, nameof(smoothedRootPositionsXZ));
        }
    }

    public sealed class KimodoConstraintSet
    {
        private readonly IKimodoConstraint[] _constraints;
        public static KimodoConstraintSet Empty { get; } = new KimodoConstraintSet(Array.Empty<IKimodoConstraint>());
        public IReadOnlyList<IKimodoConstraint> Constraints => _constraints;
        public bool HasConstraints => _constraints.Length != 0;

        public KimodoConstraintSet(params IKimodoConstraint[] constraints)
        {
            if (constraints == null) throw new ArgumentNullException(nameof(constraints));
            _constraints = (IKimodoConstraint[])constraints.Clone();
            for (int i = 0; i < _constraints.Length; i++)
                if (_constraints[i] == null) throw new ArgumentException("Constraint sets cannot contain null entries.", nameof(constraints));
        }
    }
}

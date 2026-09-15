using System;
using System.Threading;
using UnityEngine;

namespace KitsuMate.Onnx.Motion.Kimodo
{
    /// <summary>Canonical SOMA kinematics, constraint correction and planted-foot cleanup.</summary>
    internal static class KimodoMotionProcessing
    {
        private static readonly int[] Effectors = { 13, 19, 24, 28 };

        internal static float Value(float[] motion, int index) =>
            motion[index] * KimodoSomaRuntimeData.Scale[index % 369] + KimodoSomaRuntimeData.Mean[index % 369];

        private static void Set(float[] motion, int index, float value) =>
            motion[index] = (value - KimodoSomaRuntimeData.Mean[index % 369]) / KimodoSomaRuntimeData.Scale[index % 369];

        internal static Quaternion Rotation(float[] motion, int offset)
        {
            Vector3 x = new Vector3(Value(motion, offset), Value(motion, offset + 1), Value(motion, offset + 2)).normalized;
            Vector3 y = new Vector3(Value(motion, offset + 3), Value(motion, offset + 4), Value(motion, offset + 5));
            y = (y - Vector3.Dot(x, y) * x).normalized;
            Vector3 z = Vector3.Cross(x, y);
            return z.sqrMagnitude < 1e-8f ? Quaternion.identity : Quaternion.LookRotation(z, y);
        }

        private static void SetRotation(float[] motion, int offset, Quaternion rotation)
        {
            Vector3 x = rotation * Vector3.right, y = rotation * Vector3.up;
            SetVector(motion, offset, x);
            SetVector(motion, offset + 3, y);
        }

        private static void SetVector(float[] motion, int offset, Vector3 value)
        {
            Set(motion, offset, value.x); Set(motion, offset + 1, value.y); Set(motion, offset + 2, value.z);
        }

        private static Vector3 Position(float[] motion, int offset, int joint) => new Vector3(
            Value(motion, offset) + Value(motion, offset + 5 + joint * 3),
            Value(motion, offset + 6 + joint * 3),
            Value(motion, offset + 2) + Value(motion, offset + 7 + joint * 3));

        private static void Forward(Vector3[] positions, Quaternion[] rotations)
        {
            for (int joint = 1; joint < 30; joint++)
            {
                int parent = KimodoSomaRuntimeData.Parents[joint];
                positions[joint] = positions[parent] + rotations[parent] * KimodoSomaRuntimeData.RestOffsets[joint];
            }
        }

        internal static void Correct(float[] motion, KimodoConditioning conditioning, CancellationToken token)
        {
            int frames = motion.Length / 369;
            float[] targets = conditioning.ObservedMotion.ToArray();
            ReadOnlySpan<bool> mask = conditioning.MotionMask.Span;
            var positions = new Vector3[30];
            var rotations = new Quaternion[30];
            var anchors = new Vector3[2];
            var anchorRotations = new Quaternion[2];
            var planted = new bool[2];
            var lastPositions = new Vector3[30];
            for (int frame = 0; frame < frames; frame++)
            {
                token.ThrowIfCancellationRequested();
                int offset = frame * 369;
                // Root/path corrections move the entire pose rather than deforming its local positions.
                for (int feature = 0; feature < 5; feature++)
                    if (mask[offset + feature]) motion[offset + feature] = targets[offset + feature];
                positions[0] = Position(motion, offset, 0);
                if (mask[offset + 1]) positions[0].y = Value(targets, offset + 1);
                for (int joint = 0; joint < 30; joint++) rotations[joint] = Rotation(motion, offset + 95 + joint * 6);
                Forward(positions, rotations);
                if (mask[offset + 3] && mask[offset + 4])
                {
                    float heading = Mathf.Atan2(Value(targets, offset + 4), Value(targets, offset + 3));
                    RotateBranch(0, Quaternion.AngleAxis((heading - Heading(positions)) * Mathf.Rad2Deg, Vector3.up), rotations);
                    Forward(positions, rotations);
                }
                bool fullBody = mask[offset + 5] && mask[offset + 5 + 6 * 3];
                if (fullBody)
                {
                    positions[0] = Position(targets, offset, 0);
                    // Fit each branch once. A second child determines twist at hips, chest and head.
                    for (int parent = 0; parent < 30; parent++)
                    {
                        int first = -1;
                        for (int child = parent + 1; child < 30; child++)
                        {
                            if (KimodoSomaRuntimeData.Parents[child] != parent) continue;
                            Forward(positions, rotations);
                            Vector3 desired = Position(targets, offset, child) - positions[parent];
                            if (first < 0)
                            {
                                RotateBranch(parent, Quaternion.FromToRotation(positions[child] - positions[parent], desired), rotations);
                                first = child;
                            }
                            else
                            {
                                Vector3 axis = (positions[first] - positions[parent]).normalized;
                                Vector3 from = Vector3.ProjectOnPlane(positions[child] - positions[parent], axis);
                                Vector3 to = Vector3.ProjectOnPlane(desired, axis);
                                if (from.sqrMagnitude < 1e-8f || to.sqrMagnitude < 1e-8f) continue;
                                RotateBranch(parent, Quaternion.AngleAxis(Vector3.SignedAngle(from, to, axis), axis), rotations);
                                break;
                            }
                        }
                    }
                    Forward(positions, rotations);
                }
                float correctedHeading = Heading(positions);
                Set(motion, offset + 3, Mathf.Cos(correctedHeading));
                Set(motion, offset + 4, Mathf.Sin(correctedHeading));

                for (int e = 0; e < Effectors.Length; e++)
                {
                    int joint = Effectors[e];
                    bool constrained = mask[offset + 5 + joint * 3];
                    Vector3 target = constrained ? Position(targets, offset, joint) : positions[joint];
                    Quaternion rotation = mask[offset + 95 + joint * 6]
                        ? Rotation(targets, offset + 95 + joint * 6) : rotations[joint];
                    if (e >= 2)
                    {
                        int foot = e - 2;
                        bool contact = Value(motion, offset + 365 + foot * 2) > .5f ||
                            Value(motion, offset + 366 + foot * 2) > .5f;
                        if (!contact) planted[foot] = false;
                        else if (constrained || fullBody)
                        {
                            anchors[foot] = target;
                            anchorRotations[foot] = rotation;
                            planted[foot] = true;
                        }
                        if (contact && !constrained && !fullBody)
                        {
                            if (!planted[foot])
                            {
                                anchors[foot] = positions[joint];
                                anchorRotations[foot] = rotations[joint];
                                planted[foot] = true;
                            }
                            target = anchors[foot];
                            rotation = anchorRotations[foot];
                            constrained = true;
                        }
                    }
                    if (constrained)
                    {
                        if ((positions[joint] - target).sqrMagnitude > 1e-10f)
                            SolveLimb(joint, target, positions, rotations);
                        rotations[joint] = rotation;
                        Forward(positions, rotations);
                    }
                }
                for (int joint = 0; joint < 30; joint++)
                {
                    SetRotation(motion, offset + 95 + joint * 6, rotations[joint]);
                    SetVector(motion, offset + 5 + joint * 3, positions[joint] -
                        new Vector3(Value(motion, offset), 0f, Value(motion, offset + 2)));
                    if (frame > 0) SetVector(motion, offset - 369 + 275 + joint * 3,
                        (positions[joint] - lastPositions[joint]) * 30f);
                    lastPositions[joint] = positions[joint];
                }
                if (frame == frames - 1 && frame > 0)
                    Array.Copy(motion, offset - 369 + 275, motion, offset + 275, 90);
            }
        }

        private static float Heading(Vector3[] positions)
        {
            Vector3 hips = positions[26] - positions[22];
            return Mathf.Atan2(hips.z, -hips.x);
        }

        private static void SolveLimb(int end, Vector3 target, Vector3[] positions, Quaternion[] rotations)
        {
            int middle = KimodoSomaRuntimeData.Parents[end], root = KimodoSomaRuntimeData.Parents[middle];
            Vector3 a = positions[root], b = positions[middle], c = positions[end];
            float upper = Vector3.Distance(a, b), lower = Vector3.Distance(b, c);
            Vector3 direction = target - a;
            float distance = Mathf.Clamp(direction.magnitude, Mathf.Abs(upper - lower) + 1e-5f, upper + lower - 1e-5f);
            if (direction.sqrMagnitude < 1e-10f || upper < 1e-6f || lower < 1e-6f) return;
            direction.Normalize();
            Vector3 pole = b - a - Vector3.Dot(b - a, direction) * direction;
            if (pole.sqrMagnitude < 1e-8f) pole = Vector3.Cross(direction, Vector3.right);
            if (pole.sqrMagnitude < 1e-8f) pole = Vector3.Cross(direction, Vector3.up);
            float along = (upper * upper + distance * distance - lower * lower) / (2f * distance);
            Vector3 elbow = a + direction * along + pole.normalized * Mathf.Sqrt(Mathf.Max(0f, upper * upper - along * along));
            RotateBranch(root, Quaternion.FromToRotation(b - a, elbow - a), rotations);
            Forward(positions, rotations);
            RotateBranch(middle, Quaternion.FromToRotation(positions[end] - positions[middle],
                a + direction * distance - positions[middle]), rotations);
            Forward(positions, rotations);
        }

        private static void RotateBranch(int root, Quaternion delta, Quaternion[] rotations)
        {
            for (int joint = root; joint < 30; joint++)
            {
                int ancestor = joint;
                while (ancestor > root) ancestor = KimodoSomaRuntimeData.Parents[ancestor];
                if (ancestor == root) rotations[joint] = delta * rotations[joint];
            }
        }

        internal static float[] TransformHistory(float[] source, Vector3 translation, Quaternion rotation)
        {
            var result = (float[])source.Clone();
            for (int offset = 0; offset < result.Length; offset += 369)
            {
                Vector3 oldRoot = new Vector3(Value(source, offset), 0f, Value(source, offset + 2));
                Vector3 root = rotation * oldRoot + translation;
                Set(result, offset, root.x); Set(result, offset + 2, root.z);
                Set(result, offset + 1, Value(source, offset + 1) + translation.y);
                float heading = Mathf.Atan2(Value(source, offset + 4), Value(source, offset + 3));
                Vector3 forward = rotation * new Vector3(Mathf.Sin(heading), 0f, Mathf.Cos(heading));
                heading = Mathf.Atan2(forward.x, forward.z);
                Set(result, offset + 3, Mathf.Cos(heading)); Set(result, offset + 4, Mathf.Sin(heading));
                for (int joint = 0; joint < 30; joint++)
                {
                    Vector3 position = rotation * Position(source, offset, joint) + translation;
                    SetVector(result, offset + 5 + joint * 3, position - new Vector3(root.x, 0f, root.z));
                    SetRotation(result, offset + 95 + joint * 6, rotation * Rotation(source, offset + 95 + joint * 6));
                    int velocity = offset + 275 + joint * 3;
                    SetVector(result, velocity, rotation * new Vector3(Value(source, velocity), Value(source, velocity + 1), Value(source, velocity + 2)));
                }
            }
            return result;
        }
    }
}

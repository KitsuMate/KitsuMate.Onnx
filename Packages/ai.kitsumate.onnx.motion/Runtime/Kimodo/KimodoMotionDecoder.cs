using System;
using UnityEngine;

namespace KitsuMate.Onnx.Motion.Kimodo
{
    /// <summary>Decodes normalized SOMA-30 features into Unity Humanoid rotation deltas.</summary>
    internal static class KimodoMotionDecoder
    {
        private const int SmoothRootOffset = 0;
        private const int HeadingOffset = 3;
        private const int LocalPositionOffset = 5;
        private const int GlobalRotationOffset = 95;

        private static readonly int[] Parents =
        {
            -1, 0, 1, 2, 3, 4, 5, 6, 6, 6,
            3, 10, 11, 12, 13, 13,
            3, 16, 17, 18, 19, 19,
            0, 22, 23, 24,
            0, 26, 27, 28
        };

        private static readonly (int Joint, HumanBodyBones Bone)[] HumanoidMap =
        {
            (0, HumanBodyBones.Hips),
            (1, HumanBodyBones.Spine),
            (2, HumanBodyBones.Chest),
            (3, HumanBodyBones.UpperChest),
            (5, HumanBodyBones.Neck),
            (6, HumanBodyBones.Head),
            (7, HumanBodyBones.Jaw),
            (8, HumanBodyBones.LeftEye),
            (9, HumanBodyBones.RightEye),
            (10, HumanBodyBones.LeftShoulder),
            (11, HumanBodyBones.LeftUpperArm),
            (12, HumanBodyBones.LeftLowerArm),
            (13, HumanBodyBones.LeftHand),
            (16, HumanBodyBones.RightShoulder),
            (17, HumanBodyBones.RightUpperArm),
            (18, HumanBodyBones.RightLowerArm),
            (19, HumanBodyBones.RightHand),
            (22, HumanBodyBones.LeftUpperLeg),
            (23, HumanBodyBones.LeftLowerLeg),
            (24, HumanBodyBones.LeftFoot),
            (25, HumanBodyBones.LeftToes),
            (26, HumanBodyBones.RightUpperLeg),
            (27, HumanBodyBones.RightLowerLeg),
            (28, HumanBodyBones.RightFoot),
            (29, HumanBodyBones.RightToes),
        };

        public static KimodoHumanoidMotion Decode(
            float[] normalizedMotion,
            int frames,
            KimodoGenerationDiagnostics diagnostics)
        {
            if (normalizedMotion == null) throw new ArgumentNullException(nameof(normalizedMotion));
            if (normalizedMotion.Length != frames * KimodoTensorContract.MotionDimension)
                throw new ArgumentException("Normalized motion has the wrong shape.", nameof(normalizedMotion));

            int boneCount = (int)HumanBodyBones.LastBone;
            var rotations = new Quaternion[frames * boneCount];
            var globalRotations = new Quaternion[frames * boneCount];
            var smoothRoots = new Vector3[frames];
            var roots = new Vector3[frames];
            var rootRotations = new Quaternion[frames];
            var availability = new bool[boneCount];
            var global = new Matrix3[KimodoTensorContract.JointCount];
            var local = new Matrix3[KimodoTensorContract.JointCount];

            for (int i = 0; i < rotations.Length; i++) rotations[i] = Quaternion.identity;
            for (int i = 0; i < globalRotations.Length; i++) globalRotations[i] = Quaternion.identity;
            for (int i = 0; i < HumanoidMap.Length; i++) availability[(int)HumanoidMap[i].Bone] = true;

            for (int frame = 0; frame < frames; frame++)
            {
                int featureBase = frame * KimodoTensorContract.MotionDimension;
                float Feature(int index)
                {
                    int absolute = featureBase + index;
                    return normalizedMotion[absolute] * KimodoSomaRuntimeData.Scale[index] +
                           KimodoSomaRuntimeData.Mean[index];
                }

                float smoothX = Feature(SmoothRootOffset);
                float smoothZ = Feature(SmoothRootOffset + 2);
                smoothRoots[frame] = new Vector3(-smoothX, 0f, smoothZ);
                float hipsX = Feature(LocalPositionOffset);
                float hipsY = Feature(LocalPositionOffset + 1);
                float hipsZ = Feature(LocalPositionOffset + 2);
                roots[frame] = new Vector3(-(smoothX + hipsX), hipsY, smoothZ + hipsZ);

                float heading = Mathf.Atan2(Feature(HeadingOffset + 1), Feature(HeadingOffset));
                rootRotations[frame] = Quaternion.AngleAxis(-heading * Mathf.Rad2Deg, Vector3.up);

                for (int joint = 0; joint < KimodoTensorContract.JointCount; joint++)
                {
                    int offset = GlobalRotationOffset + joint * 6;
                    global[joint] = Matrix3.FromContinuous6D(
                        Feature(offset), Feature(offset + 1), Feature(offset + 2),
                        Feature(offset + 3), Feature(offset + 4), Feature(offset + 5));

                    int parent = Parents[joint];
                    local[joint] = parent < 0
                        ? Matrix3.Multiply(Matrix3.Yaw(-heading), global[joint])
                        : Matrix3.TransposeMultiply(global[parent], global[joint]);
                }

                int outputBase = frame * boneCount;
                for (int i = 0; i < HumanoidMap.Length; i++)
                {
                    var mapping = HumanoidMap[i];
                    rotations[outputBase + (int)mapping.Bone] = local[mapping.Joint].ReflectX().ToQuaternion();
                    globalRotations[outputBase + (int)mapping.Bone] =
                        Matrix3.Multiply(Matrix3.Yaw(-heading), global[mapping.Joint]).ReflectX().ToQuaternion();
                }
            }

            return new KimodoHumanoidMotion(
                frames,
                KimodoTensorContract.FramesPerSecond,
                rotations,
                roots,
                rootRotations,
                availability,
                diagnostics,
                smoothRoots,
                globalRotations);
        }

        private readonly struct Matrix3
        {
            private readonly float _m00, _m01, _m02;
            private readonly float _m10, _m11, _m12;
            private readonly float _m20, _m21, _m22;

            private Matrix3(
                float m00, float m01, float m02,
                float m10, float m11, float m12,
                float m20, float m21, float m22)
            {
                _m00 = m00; _m01 = m01; _m02 = m02;
                _m10 = m10; _m11 = m11; _m12 = m12;
                _m20 = m20; _m21 = m21; _m22 = m22;
            }

            public static Matrix3 FromContinuous6D(float xx, float xy, float xz, float yx, float yy, float yz)
            {
                Vector3 x = NormalizeSafe(new Vector3(xx, xy, xz), Vector3.right);
                Vector3 yRaw = new Vector3(yx, yy, yz);
                Vector3 z = NormalizeSafe(Vector3.Cross(x, yRaw), Vector3.forward);
                Vector3 y = NormalizeSafe(Vector3.Cross(z, x), Vector3.up);
                return FromColumns(x, y, z);
            }

            public static Matrix3 Yaw(float radians)
            {
                float c = Mathf.Cos(radians);
                float s = Mathf.Sin(radians);
                return new Matrix3(c, 0f, s, 0f, 1f, 0f, -s, 0f, c);
            }

            public static Matrix3 Multiply(Matrix3 a, Matrix3 b)
            {
                return new Matrix3(
                    a._m00*b._m00+a._m01*b._m10+a._m02*b._m20,
                    a._m00*b._m01+a._m01*b._m11+a._m02*b._m21,
                    a._m00*b._m02+a._m01*b._m12+a._m02*b._m22,
                    a._m10*b._m00+a._m11*b._m10+a._m12*b._m20,
                    a._m10*b._m01+a._m11*b._m11+a._m12*b._m21,
                    a._m10*b._m02+a._m11*b._m12+a._m12*b._m22,
                    a._m20*b._m00+a._m21*b._m10+a._m22*b._m20,
                    a._m20*b._m01+a._m21*b._m11+a._m22*b._m21,
                    a._m20*b._m02+a._m21*b._m12+a._m22*b._m22);
            }

            public static Matrix3 TransposeMultiply(Matrix3 parent, Matrix3 value)
            {
                return new Matrix3(
                    parent._m00*value._m00+parent._m10*value._m10+parent._m20*value._m20,
                    parent._m00*value._m01+parent._m10*value._m11+parent._m20*value._m21,
                    parent._m00*value._m02+parent._m10*value._m12+parent._m20*value._m22,
                    parent._m01*value._m00+parent._m11*value._m10+parent._m21*value._m20,
                    parent._m01*value._m01+parent._m11*value._m11+parent._m21*value._m21,
                    parent._m01*value._m02+parent._m11*value._m12+parent._m21*value._m22,
                    parent._m02*value._m00+parent._m12*value._m10+parent._m22*value._m20,
                    parent._m02*value._m01+parent._m12*value._m11+parent._m22*value._m21,
                    parent._m02*value._m02+parent._m12*value._m12+parent._m22*value._m22);
            }

            public Matrix3 ReflectX()
            {
                return new Matrix3(
                    _m00, -_m01, -_m02,
                    -_m10, _m11, _m12,
                    -_m20, _m21, _m22);
            }

            public Quaternion ToQuaternion()
            {
                var forward = NormalizeSafe(new Vector3(_m02, _m12, _m22), Vector3.forward);
                var up = NormalizeSafe(new Vector3(_m01, _m11, _m21), Vector3.up);
                return Quaternion.Normalize(Quaternion.LookRotation(forward, up));
            }

            private static Matrix3 FromColumns(Vector3 x, Vector3 y, Vector3 z)
            {
                return new Matrix3(x.x, y.x, z.x, x.y, y.y, z.y, x.z, y.z, z.z);
            }

            private static Vector3 NormalizeSafe(Vector3 value, Vector3 fallback)
            {
                float magnitude = value.magnitude;
                return magnitude > 1e-8f ? value / magnitude : fallback;
            }
        }
    }
}

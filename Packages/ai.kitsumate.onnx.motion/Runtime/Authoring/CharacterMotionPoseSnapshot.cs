using System;
using System.Globalization;
using UnityEngine;

namespace KitsuMate.Onnx.Motion
{
    [Serializable]
    public sealed class CharacterMotionPoseSnapshot
    {
        private const int CurrentRootSpaceVersion = 1;

        [SerializeField, HideInInspector] private Quaternion[] humanoidLocalRotations;
        [SerializeField, HideInInspector] private bool[] humanoidBoneAvailability;
        [SerializeField, HideInInspector] private Vector3[] somaPositionsRelativeToRoot;
        [SerializeField, HideInInspector] private Quaternion[] somaRotationsRelativeToRoot;
        [SerializeField, HideInInspector] private string avatarSignature;
        [SerializeField, HideInInspector] private int rootSpaceVersion;

        public bool IsCaptured =>
            humanoidLocalRotations?.Length == (int)HumanBodyBones.LastBone &&
            humanoidBoneAvailability?.Length == (int)HumanBodyBones.LastBone &&
            somaPositionsRelativeToRoot?.Length == 30 && somaRotationsRelativeToRoot?.Length == 30;
        public string AvatarSignature => avatarSignature ?? string.Empty;
        public bool UsesCurrentRootSpace => IsCaptured && rootSpaceVersion == CurrentRootSpaceVersion;
        public ReadOnlyMemory<Quaternion> HumanoidLocalRotations => humanoidLocalRotations ?? Array.Empty<Quaternion>();
        public ReadOnlyMemory<bool> HumanoidBoneAvailability => humanoidBoneAvailability ?? Array.Empty<bool>();
        public ReadOnlyMemory<Vector3> SomaPositionsRelativeToRoot => somaPositionsRelativeToRoot ?? Array.Empty<Vector3>();
        public ReadOnlyMemory<Quaternion> SomaRotationsRelativeToRoot => somaRotationsRelativeToRoot ?? Array.Empty<Quaternion>();

        public bool MatchesAvatar(string signature) => IsCaptured && string.Equals(AvatarSignature, signature ?? string.Empty, StringComparison.Ordinal);

        /// <summary>Captures the Animator's current Humanoid pose relative to a keyframe transform.</summary>
        public void Capture(Animator animator, Transform keyframeRoot)
            => Capture(animator, keyframeRoot, CharacterMotion.ComputeAvatarSignature(animator));

        internal void Capture(Animator animator, Transform keyframeRoot, string signature)
        {
            if (animator == null) throw new ArgumentNullException(nameof(animator));
            if (keyframeRoot == null) throw new ArgumentNullException(nameof(keyframeRoot));
            if (animator.avatar == null || !animator.avatar.isHuman)
                throw new ArgumentException("A valid Humanoid Animator is required.", nameof(animator));

            int count = (int)HumanBodyBones.LastBone;
            var rotations = new Quaternion[count];
            var availability = new bool[count];
            for (int i = 0; i < count; i++)
            {
                Transform bone = animator.GetBoneTransform((HumanBodyBones)i);
                rotations[i] = bone != null ? bone.localRotation : Quaternion.identity;
                availability[i] = bone != null;
            }

            var diagnostics = new CharacterMotionValidationResult();
            if (!HumanoidSomaMapper.TryCapture(animator.GetBoneTransform, out Vector3[] positions, out Quaternion[] somaRotations, diagnostics))
                throw new InvalidOperationException(string.Join("\n", diagnostics.Diagnostics));

            Quaternion inverse = Quaternion.Inverse(keyframeRoot.rotation);
            for (int i = 0; i < positions.Length; i++)
            {
                positions[i] = inverse * (positions[i] - keyframeRoot.position);
                somaRotations[i] = inverse * somaRotations[i];
            }

            Set(rotations, availability, positions, somaRotations, signature);
        }

        public string ComputeHash()
        {
            if (!IsCaptured) return string.Empty;
            var builder = new System.Text.StringBuilder(AvatarSignature).Append('|').Append(rootSpaceVersion);
            for (int i = 0; i < somaPositionsRelativeToRoot.Length; i++)
            {
                Vector3 p = somaPositionsRelativeToRoot[i]; Quaternion q = somaRotationsRelativeToRoot[i];
                builder.Append('|').Append(p.x.ToString("R", CultureInfo.InvariantCulture)).Append(',')
                    .Append(p.y.ToString("R", CultureInfo.InvariantCulture)).Append(',').Append(p.z.ToString("R", CultureInfo.InvariantCulture))
                    .Append('@').Append(q.x.ToString("R", CultureInfo.InvariantCulture)).Append(',')
                    .Append(q.y.ToString("R", CultureInfo.InvariantCulture)).Append(',').Append(q.z.ToString("R", CultureInfo.InvariantCulture))
                    .Append(',').Append(q.w.ToString("R", CultureInfo.InvariantCulture));
            }
            for (int i = 0; i < humanoidLocalRotations.Length; i++)
                if (humanoidBoneAvailability[i])
                {
                    Quaternion q = humanoidLocalRotations[i];
                    builder.Append('|').Append(i).Append('=').Append(q.x.ToString("R", CultureInfo.InvariantCulture)).Append(',')
                        .Append(q.y.ToString("R", CultureInfo.InvariantCulture)).Append(',').Append(q.z.ToString("R", CultureInfo.InvariantCulture))
                        .Append(',').Append(q.w.ToString("R", CultureInfo.InvariantCulture));
                }
            return Hash128.Compute(builder.ToString()).ToString();
        }

        public void Set(
            Quaternion[] humanoidRotations,
            bool[] humanoidAvailability,
            Vector3[] somaPositions,
            Quaternion[] somaRotations,
            string signature)
        {
            int bones = (int)HumanBodyBones.LastBone;
            if (humanoidRotations == null || humanoidRotations.Length != bones) throw new ArgumentException($"Expected {bones} Humanoid rotations.", nameof(humanoidRotations));
            if (humanoidAvailability == null || humanoidAvailability.Length != bones) throw new ArgumentException($"Expected {bones} Humanoid availability values.", nameof(humanoidAvailability));
            if (somaPositions == null || somaPositions.Length != 30) throw new ArgumentException("Expected 30 SOMA positions.", nameof(somaPositions));
            if (somaRotations == null || somaRotations.Length != 30) throw new ArgumentException("Expected 30 SOMA rotations.", nameof(somaRotations));
            ValidateFinite(somaPositions, somaRotations);
            humanoidLocalRotations = (Quaternion[])humanoidRotations.Clone();
            humanoidBoneAvailability = (bool[])humanoidAvailability.Clone();
            somaPositionsRelativeToRoot = (Vector3[])somaPositions.Clone();
            somaRotationsRelativeToRoot = (Quaternion[])somaRotations.Clone();
            avatarSignature = signature ?? string.Empty;
            rootSpaceVersion = CurrentRootSpaceVersion;
        }

        public void Clear()
        {
            humanoidLocalRotations = null;
            humanoidBoneAvailability = null;
            somaPositionsRelativeToRoot = null;
            somaRotationsRelativeToRoot = null;
            avatarSignature = null;
            rootSpaceVersion = 0;
        }

        private static void ValidateFinite(Vector3[] positions, Quaternion[] rotations)
        {
            for (int i = 0; i < positions.Length; i++)
            {
                Vector3 p = positions[i]; Quaternion q = rotations[i];
                if (!Finite(p.x) || !Finite(p.y) || !Finite(p.z) || !Finite(q.x) || !Finite(q.y) || !Finite(q.z) || !Finite(q.w))
                    throw new ArgumentException($"Pose contains a non-finite value at SOMA joint {i}.");
            }
        }

        private static bool Finite(float value) => !float.IsNaN(value) && !float.IsInfinity(value);
    }
}

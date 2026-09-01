using System;
using UnityEngine;

namespace KitsuMate.Onnx.Motion
{
    /// <summary>Maps an existing Humanoid transform pose into the exact public SOMA-30 joint order.</summary>
    public static class HumanoidSomaMapper
    {
        public static bool TryCapture(
            Func<HumanBodyBones, Transform> boneResolver,
            out Vector3[] positions,
            out Quaternion[] rotations,
            CharacterMotionValidationResult diagnostics = null)
        {
            positions = new Vector3[30];
            rotations = new Quaternion[30];
            return TryCapture(boneResolver, positions, rotations, diagnostics);
        }

        /// <summary>Fills caller-owned SOMA-30 buffers, avoiding allocations in editor rendering loops.</summary>
        public static bool TryCapture(
            Func<HumanBodyBones, Transform> boneResolver,
            Vector3[] positions,
            Quaternion[] rotations,
            CharacterMotionValidationResult diagnostics = null)
        {
            if (boneResolver == null) throw new ArgumentNullException(nameof(boneResolver));
            if (positions == null || positions.Length < 30)
                throw new ArgumentException("A 30-element position buffer is required.", nameof(positions));
            if (rotations == null || rotations.Length < 30)
                throw new ArgumentException("A 30-element rotation buffer is required.", nameof(rotations));
            diagnostics ??= new CharacterMotionValidationResult();

            Transform hips = Required(boneResolver, HumanBodyBones.Hips, diagnostics);
            Transform spine = Required(boneResolver, HumanBodyBones.Spine, diagnostics);
            Transform chest = Required(boneResolver, HumanBodyBones.Chest, diagnostics);
            Transform upperChest = boneResolver(HumanBodyBones.UpperChest) ?? chest;
            Transform neck = Required(boneResolver, HumanBodyBones.Neck, diagnostics);
            Transform head = Required(boneResolver, HumanBodyBones.Head, diagnostics);
            Transform lShoulder = Required(boneResolver, HumanBodyBones.LeftShoulder, diagnostics);
            Transform lUpperArm = Required(boneResolver, HumanBodyBones.LeftUpperArm, diagnostics);
            Transform lLowerArm = Required(boneResolver, HumanBodyBones.LeftLowerArm, diagnostics);
            Transform lHand = Required(boneResolver, HumanBodyBones.LeftHand, diagnostics);
            Transform rShoulder = Required(boneResolver, HumanBodyBones.RightShoulder, diagnostics);
            Transform rUpperArm = Required(boneResolver, HumanBodyBones.RightUpperArm, diagnostics);
            Transform rLowerArm = Required(boneResolver, HumanBodyBones.RightLowerArm, diagnostics);
            Transform rHand = Required(boneResolver, HumanBodyBones.RightHand, diagnostics);
            Transform lUpperLeg = Required(boneResolver, HumanBodyBones.LeftUpperLeg, diagnostics);
            Transform lLowerLeg = Required(boneResolver, HumanBodyBones.LeftLowerLeg, diagnostics);
            Transform lFoot = Required(boneResolver, HumanBodyBones.LeftFoot, diagnostics);
            Transform rUpperLeg = Required(boneResolver, HumanBodyBones.RightUpperLeg, diagnostics);
            Transform rLowerLeg = Required(boneResolver, HumanBodyBones.RightLowerLeg, diagnostics);
            Transform rFoot = Required(boneResolver, HumanBodyBones.RightFoot, diagnostics);
            if (!diagnostics.IsValid) return false;

            Set(0, hips); Set(1, spine); Set(2, chest); Set(3, upperChest);
            SetInterpolated(4, upperChest, neck, 0.5f); Set(5, neck); Set(6, head);
            SetOptionalOrFallback(7, boneResolver(HumanBodyBones.Jaw), head, head.position + head.up * -HeadScale(head, neck) * 0.2f, "jaw");
            SetOptionalOrFallback(8, boneResolver(HumanBodyBones.LeftEye), head, head.position + (-head.right * 0.03f + head.up * 0.04f + head.forward * 0.06f) * HeadScale(head, neck) / 0.1f, "left eye");
            SetOptionalOrFallback(9, boneResolver(HumanBodyBones.RightEye), head, head.position + (head.right * 0.03f + head.up * 0.04f + head.forward * 0.06f) * HeadScale(head, neck) / 0.1f, "right eye");

            Set(10, lShoulder); Set(11, lUpperArm); Set(12, lLowerArm); Set(13, lHand);
            SetEndpoint(14, boneResolver(HumanBodyBones.LeftThumbDistal), boneResolver(HumanBodyBones.LeftThumbIntermediate), lHand, "left thumb endpoint");
            SetEndpoint(15, boneResolver(HumanBodyBones.LeftMiddleDistal), boneResolver(HumanBodyBones.LeftMiddleIntermediate), lHand, "left middle endpoint");
            Set(16, rShoulder); Set(17, rUpperArm); Set(18, rLowerArm); Set(19, rHand);
            SetEndpoint(20, boneResolver(HumanBodyBones.RightThumbDistal), boneResolver(HumanBodyBones.RightThumbIntermediate), rHand, "right thumb endpoint");
            SetEndpoint(21, boneResolver(HumanBodyBones.RightMiddleDistal), boneResolver(HumanBodyBones.RightMiddleIntermediate), rHand, "right middle endpoint");

            Set(22, lUpperLeg); Set(23, lLowerLeg); Set(24, lFoot);
            SetToe(25, boneResolver(HumanBodyBones.LeftToes), lFoot, lLowerLeg, "left toes");
            Set(26, rUpperLeg); Set(27, rLowerLeg); Set(28, rFoot);
            SetToe(29, boneResolver(HumanBodyBones.RightToes), rFoot, rLowerLeg, "right toes");
            return diagnostics.IsValid;

            void Set(int index, Transform transform)
            {
                positions[index] = transform.position;
                rotations[index] = transform.rotation;
            }

            void SetInterpolated(int index, Transform a, Transform b, float t)
            {
                positions[index] = Vector3.Lerp(a.position, b.position, t);
                rotations[index] = Quaternion.Slerp(a.rotation, b.rotation, t);
                diagnostics.Warning("derived_joint", $"SOMA joint {(KimodoJoint)index} is interpolated from the Humanoid hierarchy.");
            }

            void SetOptionalOrFallback(int index, Transform value, Transform parent, Vector3 fallback, string label)
            {
                if (value != null) { Set(index, value); return; }
                positions[index] = fallback;
                rotations[index] = parent.rotation;
                diagnostics.Warning("optional_bone_fallback", $"The avatar has no {label}; a head-relative fallback is used.");
            }

            void SetEndpoint(int index, Transform distal, Transform intermediate, Transform hand, string label)
            {
                if (distal != null)
                {
                    Vector3 segment = intermediate != null ? distal.position - intermediate.position : distal.forward * 0.02f;
                    if (segment.sqrMagnitude < 1e-8f) segment = distal.forward * 0.02f;
                    positions[index] = distal.position + segment;
                    rotations[index] = distal.rotation;
                    return;
                }
                positions[index] = hand.position + hand.forward * 0.08f;
                rotations[index] = hand.rotation;
                diagnostics.Warning("optional_bone_fallback", $"The avatar has no {label}; a hand-relative fallback is used.");
            }

            void SetToe(int index, Transform toes, Transform foot, Transform shin, string label)
            {
                if (toes != null) { Set(index, toes); return; }
                // Keep the synthetic toe attached to foot orientation so rotating a
                // foot IK target also rotates both its Kimodo endpoint and visual bounds.
                Vector3 forward = Vector3.ProjectOnPlane(foot.forward, Vector3.up).normalized;
                if (forward.sqrMagnitude < 1e-8f)
                    forward = Vector3.ProjectOnPlane(foot.position - shin.position, Vector3.up).normalized;
                if (forward.sqrMagnitude < 1e-8f) forward = foot.forward;
                positions[index] = foot.position + forward * Mathf.Max(0.08f, Vector3.Distance(foot.position, shin.position) * 0.25f);
                rotations[index] = foot.rotation;
                diagnostics.Warning("optional_bone_fallback", $"The avatar has no {label}; a foot-relative fallback is used.");
            }
        }

        private static Transform Required(Func<HumanBodyBones, Transform> resolver, HumanBodyBones bone, CharacterMotionValidationResult diagnostics)
        {
            Transform result = resolver(bone);
            if (result == null) diagnostics.Error("missing_required_bone", $"The Humanoid avatar is missing {bone}.");
            return result;
        }

        private static float HeadScale(Transform head, Transform neck) => Mathf.Max(0.08f, Vector3.Distance(head.position, neck.position));
    }
}

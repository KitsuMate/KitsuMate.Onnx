using System;
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;

namespace KitsuMate.Onnx.Motion.Editor
{
    internal static class CharacterMotionAuthoringUtility
    {
        internal const string EditorControlsName = "EditorControls";

        internal static readonly CharacterMotionConstraintType[] EffectorTypes =
        {
            CharacterMotionConstraintType.LeftHand,
            CharacterMotionConstraintType.RightHand,
            CharacterMotionConstraintType.LeftFoot,
            CharacterMotionConstraintType.RightFoot,
        };

        internal static CharacterMotionSkeleton EnsureSkeleton(CharacterMotion motion)
        {
            if (motion == null) throw new ArgumentNullException(nameof(motion));
            Animator source = motion.TargetAnimator;
            if (source == null || source.avatar == null || !source.avatar.isHuman)
                throw new InvalidOperationException("Assign a valid Humanoid Animator before authoring poses.");

            CharacterMotionSkeleton skeleton = motion.GetComponentInChildren<CharacterMotionSkeleton>(true);
            if (skeleton == null || skeleton.Owner != motion || skeleton.GuideAnimator == null ||
                skeleton.GuideAnimator.avatar != source.avatar)
                skeleton = CharacterMotionSkeletonUtility.Rebuild(motion);
            skeleton.gameObject.hideFlags |= HideFlags.DontSaveInEditor | HideFlags.DontSaveInBuild;
            return skeleton;
        }

        internal static bool IsStale(CharacterMotionKeyframe keyframe, CharacterMotion motion)
            => IsStale(keyframe, motion, motion != null ? motion.AvatarSignature : string.Empty);

        private static bool IsStale(
            CharacterMotionKeyframe keyframe,
            CharacterMotion motion,
            string avatarSignature)
        {
            if (keyframe == null || motion == null || !keyframe.RequiresPose) return false;
            if (!keyframe.Pose.IsCaptured || !keyframe.Pose.UsesCurrentRootSpace ||
                !keyframe.Pose.MatchesAvatar(avatarSignature) ||
                !HasFinitePose(keyframe.Pose)) return true;
            foreach (CharacterMotionConstraintType type in EffectorTypes)
            {
                if ((keyframe.Constraints & type) == 0) continue;
                if (keyframe.GetEffector(type) == null) return true;
                if (HasFullPose(keyframe) && FindPole(keyframe, type) == null) return true;
            }
            return false;
        }

        internal static int CountStale(CharacterMotion motion)
        {
            if (motion == null) return 0;
            string avatarSignature = motion.AvatarSignature;
            int count = 0;
            foreach (CharacterMotionKeyframe keyframe in motion.GetKeyframes())
                if (IsStale(keyframe, motion, avatarSignature)) count++;
            return count;
        }

        internal static CharacterMotionSkeleton PrepareKeyframe(
            CharacterMotionKeyframe keyframe,
            bool captureMissing = true)
        {
            if (keyframe == null) throw new ArgumentNullException(nameof(keyframe));
            CharacterMotion motion = keyframe.GetComponentInParent<CharacterMotion>();
            if (motion == null) throw new InvalidOperationException("The keyframe must be a child of CharacterMotion.");
            CharacterMotionSkeleton skeleton = EnsureSkeleton(motion);
            bool captured = keyframe.Pose.IsCaptured;
            if (captured)
                skeleton.LoadPose(keyframe);
            else
            {
                RecordSkeleton(skeleton, "Initialize Character Motion Pose");
                skeleton.LoadBindPose(keyframe);
            }
            RepairEnabledControls(keyframe, skeleton);
            if (!captured && captureMissing) Capture(skeleton, keyframe, "Initialize Character Motion Pose");
            return skeleton;
        }

        internal static void SetConstraint(CharacterMotionKeyframe keyframe, CharacterMotionConstraintType type, bool enabled)
        {
            if (keyframe == null) return;
            int current = (int)keyframe.Constraints;
            int value = enabled ? current | (int)type : current & ~(int)type;
            if (value == current) return;

            CharacterMotionSkeleton skeleton = null;
            if (enabled && RequiresPose(type))
            {
                skeleton = PrepareKeyframe(keyframe, false);
                if (IsEffector(type))
                {
                    CharacterMotionIkChangeTracker.RunSuppressed(() =>
                        CharacterMotionSkeletonUtility.CreateEffector(keyframe, type));
                }
            }

            Undo.RecordObject(keyframe, "Change Character Motion Constraints");
            // Create a fresh snapshot after repairing controls. Reusing a SerializedObject
            // created before CreateEffector would write its stale null references back over
            // the newly assigned hand/foot fields when applying the constraint change.
            var serialized = new SerializedObject(keyframe);
            SerializedProperty constraints = serialized.FindProperty("constraints");
            constraints.intValue = value;
            serialized.ApplyModifiedProperties();
            EditorUtility.SetDirty(keyframe);
            PrefabUtility.RecordPrefabInstancePropertyModifications(keyframe);

            if (enabled && RequiresPose(type))
            {
                if (HasFullPose(keyframe))
                {
                    AlignControls(keyframe, skeleton, false);
                    ApplyEnabledIk(keyframe, skeleton);
                }
                Capture(skeleton, keyframe, "Enable Character Motion Constraint");
            }
            SceneView.RepaintAll();
        }

        internal static void ResetPose(CharacterMotionKeyframe keyframe)
        {
            CharacterMotionSkeleton skeleton = PrepareKeyframe(keyframe, false);
            RecordSkeleton(skeleton, "Reset Character Motion Pose");
            skeleton.LoadBindPose(keyframe);
            AlignControls(keyframe, skeleton, true);
            Capture(skeleton, keyframe, "Reset Character Motion Pose");
            SceneView.RepaintAll();
        }

        internal static int Recalculate(CharacterMotion motion)
        {
            if (motion == null) throw new ArgumentNullException(nameof(motion));
            CharacterMotionKeyframe[] keyframes = motion.GetKeyframes();
            var stale = new List<CharacterMotionKeyframe>();
            string avatarSignature = motion.AvatarSignature;
            foreach (CharacterMotionKeyframe keyframe in keyframes)
                if (IsStale(keyframe, motion, avatarSignature)) stale.Add(keyframe);
            if (stale.Count == 0) return 0;

            Undo.IncrementCurrentGroup();
            int undoGroup = Undo.GetCurrentGroup();
            Undo.SetCurrentGroupName("Recalculate Character Motion Poses");
            CharacterMotionKeyframe selected = Selection.activeGameObject != null
                ? Selection.activeGameObject.GetComponent<CharacterMotionKeyframe>()
                : null;
            CharacterMotionSkeleton skeleton = EnsureSkeleton(motion);
            try
            {
                foreach (CharacterMotionKeyframe keyframe in stale)
                {
                    RecordSkeleton(skeleton, "Recalculate Character Motion Poses");
                    if (keyframe.Pose.IsCaptured) skeleton.LoadPose(keyframe);
                    else skeleton.LoadBindPose(keyframe);
                    AlignControls(keyframe, skeleton, false);
                    ApplyEnabledIk(keyframe, skeleton);
                    Capture(skeleton, keyframe, "Recalculate Character Motion Poses");
                }
                if (selected != null && selected.GetComponentInParent<CharacterMotion>() == motion && selected.Pose.IsCaptured)
                    skeleton.LoadPose(selected);
            }
            finally
            {
                Undo.CollapseUndoOperations(undoGroup);
                SceneView.RepaintAll();
            }
            return stale.Count;
        }

        internal static void AlignControls(CharacterMotionKeyframe keyframe, CharacterMotionSkeleton skeleton, bool includeExisting)
            => CharacterMotionIkChangeTracker.RunSuppressed(() =>
                AlignControlsInternal(keyframe, skeleton, includeExisting));

        private static void AlignControlsInternal(CharacterMotionKeyframe keyframe, CharacterMotionSkeleton skeleton, bool includeExisting)
        {
            if (keyframe == null) throw new ArgumentNullException(nameof(keyframe));
            if (skeleton == null || skeleton.GuideAnimator == null)
                throw new InvalidOperationException("The Character Motion guide is missing or invalid.");
            foreach (CharacterMotionConstraintType type in EffectorTypes)
            {
                bool enabled = (keyframe.Constraints & type) != 0;
                Transform effector = keyframe.GetEffector(type);
                if (!enabled && (!includeExisting || effector == null)) continue;
                bool createdEffector = effector == null;
                // Do not use ??= for UnityEngine.Object. A destroyed serialized reference
                // compares equal to null through Unity's operator but is not CLR-null, so
                // ??= would leave the unusable reference in place.
                if (effector == null)
                    effector = CharacterMotionSkeletonUtility.CreateEffector(keyframe, type);
                if (effector == null)
                    throw new InvalidOperationException($"Unable to create the {type} control for frame {keyframe.Frame}.");
                Transform tip = skeleton.GuideAnimator.GetBoneTransform(TipBone(type));
                if (tip != null && (includeExisting || createdEffector))
                {
                    Undo.RecordObject(effector, "Align Character Motion IK Controls");
                    effector.SetPositionAndRotation(tip.position, tip.rotation);
                }
                bool createdPole = FindPole(keyframe, type) == null;
                bool requirePole = HasFullPose(keyframe) && enabled;
                if (requirePole || includeExisting && !createdPole)
                    EnsurePole(keyframe, type, skeleton, includeExisting || createdPole);
            }
        }

        internal static Transform EnsurePole(
            CharacterMotionKeyframe keyframe,
            CharacterMotionConstraintType type,
            CharacterMotionSkeleton skeleton,
            bool align)
        {
            if (keyframe == null) throw new ArgumentNullException(nameof(keyframe));
            if (skeleton == null || skeleton.GuideAnimator == null)
                throw new InvalidOperationException("The Character Motion guide is missing or invalid.");
            Transform controls = keyframe.transform.Find(EditorControlsName);
            if (controls == null)
            {
                var group = new GameObject(EditorControlsName);
                Undo.RegisterCreatedObjectUndo(group, "Create Character Motion IK Controls");
                group.tag = "EditorOnly";
                group.transform.SetParent(keyframe.transform, false);
                controls = group.transform;
            }
            string name = PoleName(type);
            Transform pole = controls.Find(name);
            if (pole == null)
            {
                var value = new GameObject(name);
                Undo.RegisterCreatedObjectUndo(value, "Create Character Motion IK Pole");
                value.tag = "EditorOnly";
                value.transform.SetParent(controls, false);
                pole = value.transform;
                align = true;
            }
            if (align) InitializePole(pole, skeleton, type);
            return pole;
        }

        internal static Transform FindPole(CharacterMotionKeyframe keyframe, CharacterMotionConstraintType type)
            => keyframe != null ? keyframe.transform.Find(EditorControlsName + "/" + PoleName(type)) : null;

        internal static void InitializePole(Transform pole, CharacterMotionSkeleton skeleton, CharacterMotionConstraintType type)
        {
            if (pole == null) throw new ArgumentNullException(nameof(pole));
            if (skeleton == null || skeleton.GuideAnimator == null)
                throw new InvalidOperationException("The Character Motion guide is missing or invalid.");
            GetChain(skeleton.GuideAnimator, type, out Transform root, out Transform middle, out Transform tip);
            if (root == null || middle == null || tip == null) return;
            Vector3 axis = (tip.position - root.position).normalized;
            Vector3 bend = Vector3.ProjectOnPlane(middle.position - root.position, axis);
            if (bend.sqrMagnitude < 1e-8f)
            {
                bend = Vector3.ProjectOnPlane(skeleton.transform.forward, axis);
                if (bend.sqrMagnitude < 1e-8f) bend = Vector3.ProjectOnPlane(Vector3.up, axis);
                if (bend.sqrMagnitude < 1e-8f) bend = Vector3.right;
            }
            float distance = Mathf.Max(Vector3.Distance(root.position, tip.position) * 0.65f, 0.15f);
            Undo.RecordObject(pole, "Align Character Motion IK Pole");
            pole.position = middle.position + bend.normalized * distance;
            pole.rotation = Quaternion.identity;
        }

        internal static bool ApplyIk(
            CharacterMotionKeyframe keyframe,
            CharacterMotionSkeleton skeleton,
            CharacterMotionConstraintType type)
        {
            if (!HasFullPose(keyframe)) return false;
            Transform target = keyframe.GetEffector(type);
            Transform pole = FindPole(keyframe, type);
            if (target == null || pole == null) return true;
            GetChain(skeleton.GuideAnimator, type, out Transform root, out Transform middle, out Transform tip);
            if (root == null || middle == null || tip == null) return true;
            return CharacterMotionTwoBoneIk.Solve(root, middle, tip, target.position, target.rotation, pole.position);
        }

        internal static void ApplyEnabledIk(CharacterMotionKeyframe keyframe, CharacterMotionSkeleton skeleton)
        {
            if (!HasFullPose(keyframe)) return;
            foreach (CharacterMotionConstraintType type in EffectorTypes)
                if ((keyframe.Constraints & type) != 0) ApplyIk(keyframe, skeleton, type);
        }

        internal static void Capture(CharacterMotionSkeleton skeleton, CharacterMotionKeyframe keyframe, string operation)
        {
            if (skeleton == null) throw new ArgumentNullException(nameof(skeleton));
            if (keyframe == null) throw new ArgumentNullException(nameof(keyframe));
            Undo.RecordObject(keyframe, operation);
            skeleton.CapturePose(keyframe);
            EditorUtility.SetDirty(keyframe);
        }

        internal static void RecordSkeleton(CharacterMotionSkeleton skeleton, string operation)
        {
            if (skeleton == null) throw new ArgumentNullException(nameof(skeleton));
            Transform[] transforms = skeleton.GetComponentsInChildren<Transform>(true);
            Undo.RecordObjects(Array.ConvertAll(transforms, value => (UnityEngine.Object)value), operation);
        }

        internal static bool IsEffector(CharacterMotionConstraintType type)
            => type == CharacterMotionConstraintType.LeftHand || type == CharacterMotionConstraintType.RightHand ||
               type == CharacterMotionConstraintType.LeftFoot || type == CharacterMotionConstraintType.RightFoot;

        internal static bool RequiresPose(CharacterMotionConstraintType type)
            => type == CharacterMotionConstraintType.FullBodyPose || IsEffector(type);

        internal static bool HasFullPose(CharacterMotionKeyframe keyframe)
            => keyframe != null &&
               (keyframe.Constraints & CharacterMotionConstraintType.FullBodyPose) != 0;

        private static void RepairEnabledControls(
            CharacterMotionKeyframe keyframe,
            CharacterMotionSkeleton skeleton)
        {
            CharacterMotionIkChangeTracker.RunSuppressed(() =>
            {
                foreach (CharacterMotionConstraintType type in EffectorTypes)
                {
                    if ((keyframe.Constraints & type) == 0) continue;
                    CharacterMotionSkeletonUtility.CreateEffector(keyframe, type);
                    if (HasFullPose(keyframe)) EnsurePole(keyframe, type, skeleton, false);
                }
            });
        }

        internal static HumanBodyBones TipBone(CharacterMotionConstraintType type) => type switch
        {
            CharacterMotionConstraintType.LeftHand => HumanBodyBones.LeftHand,
            CharacterMotionConstraintType.RightHand => HumanBodyBones.RightHand,
            CharacterMotionConstraintType.LeftFoot => HumanBodyBones.LeftFoot,
            CharacterMotionConstraintType.RightFoot => HumanBodyBones.RightFoot,
            _ => throw new ArgumentOutOfRangeException(nameof(type)),
        };

        internal static void GetChain(
            Animator animator,
            CharacterMotionConstraintType type,
            out Transform root,
            out Transform middle,
            out Transform tip)
        {
            (HumanBodyBones a, HumanBodyBones b, HumanBodyBones c) bones = type switch
            {
                CharacterMotionConstraintType.LeftHand => (HumanBodyBones.LeftUpperArm, HumanBodyBones.LeftLowerArm, HumanBodyBones.LeftHand),
                CharacterMotionConstraintType.RightHand => (HumanBodyBones.RightUpperArm, HumanBodyBones.RightLowerArm, HumanBodyBones.RightHand),
                CharacterMotionConstraintType.LeftFoot => (HumanBodyBones.LeftUpperLeg, HumanBodyBones.LeftLowerLeg, HumanBodyBones.LeftFoot),
                CharacterMotionConstraintType.RightFoot => (HumanBodyBones.RightUpperLeg, HumanBodyBones.RightLowerLeg, HumanBodyBones.RightFoot),
                _ => throw new ArgumentOutOfRangeException(nameof(type)),
            };
            root = animator.GetBoneTransform(bones.a);
            middle = animator.GetBoneTransform(bones.b);
            tip = animator.GetBoneTransform(bones.c);
        }

        private static string PoleName(CharacterMotionConstraintType type) => type switch
        {
            CharacterMotionConstraintType.LeftHand => "LeftElbowPole",
            CharacterMotionConstraintType.RightHand => "RightElbowPole",
            CharacterMotionConstraintType.LeftFoot => "LeftKneePole",
            CharacterMotionConstraintType.RightFoot => "RightKneePole",
            _ => throw new ArgumentOutOfRangeException(nameof(type)),
        };

        private static bool HasFinitePose(CharacterMotionPoseSnapshot pose)
        {
            foreach (Vector3 value in pose.SomaPositionsRelativeToRoot.Span)
                if (!CharacterMotionSpace.IsFinite(value)) return false;
            foreach (Quaternion value in pose.SomaRotationsRelativeToRoot.Span)
                if (!CharacterMotionSpace.IsFinite(value)) return false;
            foreach (Quaternion value in pose.HumanoidLocalRotations.Span)
                if (!CharacterMotionSpace.IsFinite(value)) return false;
            return true;
        }
    }

    internal static class CharacterMotionTwoBoneIk
    {
        internal static bool Solve(
            Transform root,
            Transform middle,
            Transform tip,
            Vector3 targetPosition,
            Quaternion targetRotation,
            Vector3 polePosition)
        {
            Vector3 rootPosition = root.position;
            float firstLength = Vector3.Distance(rootPosition, middle.position);
            float secondLength = Vector3.Distance(middle.position, tip.position);
            if (firstLength < 1e-6f || secondLength < 1e-6f) return true;

            Vector3 targetOffset = targetPosition - rootPosition;
            float requestedDistance = targetOffset.magnitude;
            Vector3 direction = requestedDistance > 1e-6f
                ? targetOffset / requestedDistance
                : (tip.position - rootPosition).normalized;
            if (direction.sqrMagnitude < 1e-8f) direction = root.forward;

            float minimum = Mathf.Abs(firstLength - secondLength) + 1e-5f;
            float maximum = firstLength + secondLength - 1e-5f;
            float distance = Mathf.Clamp(requestedDistance, minimum, maximum);
            bool invalid = requestedDistance < minimum || requestedDistance > maximum;

            Vector3 bend = Vector3.ProjectOnPlane(polePosition - rootPosition, direction);
            if (bend.sqrMagnitude < 1e-8f)
                bend = Vector3.ProjectOnPlane(middle.position - rootPosition, direction);
            if (bend.sqrMagnitude < 1e-8f) bend = Vector3.ProjectOnPlane(Vector3.up, direction);
            if (bend.sqrMagnitude < 1e-8f) bend = Vector3.ProjectOnPlane(Vector3.right, direction);
            bend.Normalize();

            float along = (firstLength * firstLength + distance * distance - secondLength * secondLength) /
                          (2f * distance);
            float height = Mathf.Sqrt(Mathf.Max(0f, firstLength * firstLength - along * along));
            Vector3 desiredMiddle = rootPosition + direction * along + bend * height;
            Vector3 desiredTip = rootPosition + direction * distance;

            Vector3 currentFirst = middle.position - rootPosition;
            Vector3 desiredFirst = desiredMiddle - rootPosition;
            if (currentFirst.sqrMagnitude > 1e-8f && desiredFirst.sqrMagnitude > 1e-8f)
                root.rotation = Quaternion.FromToRotation(currentFirst, desiredFirst) * root.rotation;

            Vector3 currentSecond = tip.position - middle.position;
            Vector3 desiredSecond = desiredTip - middle.position;
            if (currentSecond.sqrMagnitude > 1e-8f && desiredSecond.sqrMagnitude > 1e-8f)
                middle.rotation = Quaternion.FromToRotation(currentSecond, desiredSecond) * middle.rotation;
            tip.rotation = targetRotation;
            return invalid;
        }
    }
}

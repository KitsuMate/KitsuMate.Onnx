using System;
using UnityEditor;
using UnityEngine;

namespace KitsuMate.Onnx.Motion.Editor
{
    internal static class CharacterMotionSkeletonUtility
    {
        internal static CharacterMotionSkeleton Rebuild(CharacterMotion motion)
        {
            if (motion == null) throw new ArgumentNullException(nameof(motion));
            // Recover the source hierarchy before cloning if an animation preview is active.
            // This is especially important for scenes previewed by older package versions,
            // which sampled the target armature instead of the editor-only guide.
            if (AnimationMode.InAnimationMode()) AnimationMode.StopAnimationMode();
            CharacterMotionSkeleton existing = motion.GetComponentInChildren<CharacterMotionSkeleton>(true);
            if (existing != null) existing.StopPreview();
            Animator source = motion.TargetAnimator;
            if (source == null || source.avatar == null || !source.avatar.isHuman)
                throw new InvalidOperationException("Assign a valid Humanoid Animator before creating the motion skeleton.");
            Remove(motion);

            Transform hips = source.GetBoneTransform(HumanBodyBones.Hips);
            if (hips == null) throw new InvalidOperationException("The Humanoid avatar has no hips transform.");
            Transform armatureRoot = hips;
            while (armatureRoot.parent != null && armatureRoot.parent != source.transform)
                armatureRoot = armatureRoot.parent;

            var root = new GameObject("MotionSkeleton");
            Undo.RegisterCreatedObjectUndo(root, "Create Character Motion Skeleton");
            root.transform.SetParent(motion.transform, false);
            root.tag = "EditorOnly";
            root.hideFlags |= HideFlags.DontSaveInBuild;

            var guideAnimator = Undo.AddComponent<Animator>(root);
            guideAnimator.avatar = source.avatar;
            guideAnimator.applyRootMotion = false;
            CloneHierarchy(armatureRoot, root.transform);
            guideAnimator.Rebind();
            guideAnimator.Update(0f);
            // The guide is manipulated and sampled explicitly. Leaving its Animator active
            // lets Humanoid evaluation overwrite transform-curve previews on the next update.
            guideAnimator.enabled = false;

            var skeleton = Undo.AddComponent<CharacterMotionSkeleton>(root);
            skeleton.Initialize(motion, guideAnimator);
            Selection.activeGameObject = root;
            return skeleton;
        }

        internal static void Remove(CharacterMotion motion)
        {
            CharacterMotionSkeleton existing = motion.GetComponentInChildren<CharacterMotionSkeleton>(true);
            if (existing != null) Undo.DestroyObjectImmediate(existing.gameObject);
        }

        internal static Transform CreateEffector(CharacterMotionKeyframe keyframe, CharacterMotionConstraintType type)
        {
            Transform existing = keyframe.GetEffector(type);
            if (existing != null) return existing;
            var child = new GameObject(type.ToString());
            Undo.RegisterCreatedObjectUndo(child, $"Create {type} Handle");
            child.transform.SetParent(keyframe.transform, false);

            CharacterMotion motion = keyframe.GetComponentInParent<CharacterMotion>();
            CharacterMotionSkeleton skeleton = motion != null ? motion.GetComponentInChildren<CharacterMotionSkeleton>(true) : null;
            HumanBodyBones bone = type switch
            {
                CharacterMotionConstraintType.LeftHand => HumanBodyBones.LeftHand,
                CharacterMotionConstraintType.RightHand => HumanBodyBones.RightHand,
                CharacterMotionConstraintType.LeftFoot => HumanBodyBones.LeftFoot,
                CharacterMotionConstraintType.RightFoot => HumanBodyBones.RightFoot,
                _ => throw new ArgumentOutOfRangeException(nameof(type)),
            };
            Transform source = skeleton != null ? skeleton.GuideAnimator.GetBoneTransform(bone) : null;
            if (source != null) child.transform.SetPositionAndRotation(source.position, source.rotation);
            Undo.RecordObject(keyframe, "Assign Character Motion Effector");
            keyframe.SetEffector(type, child.transform);
            EditorUtility.SetDirty(keyframe);
            return child.transform;
        }

        private static Transform CloneHierarchy(Transform source, Transform parent)
        {
            var clone = new GameObject(source.name);
            clone.hideFlags |= HideFlags.DontSaveInBuild;
            clone.transform.SetParent(parent, false);
            clone.transform.localPosition = source.localPosition;
            clone.transform.localRotation = source.localRotation;
            clone.transform.localScale = source.localScale;
            for (int i = 0; i < source.childCount; i++) CloneHierarchy(source.GetChild(i), clone.transform);
            return clone.transform;
        }
    }
}

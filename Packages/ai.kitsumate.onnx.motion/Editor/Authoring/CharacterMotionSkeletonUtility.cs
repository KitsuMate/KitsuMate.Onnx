using System;
using System.Collections.Generic;
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
            root.hideFlags |= HideFlags.DontSaveInEditor | HideFlags.DontSaveInBuild;

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
            return skeleton;
        }

        internal static void Remove(CharacterMotion motion)
        {
            CharacterMotionSkeleton existing = motion.GetComponentInChildren<CharacterMotionSkeleton>(true);
            if (existing != null) Undo.DestroyObjectImmediate(existing.gameObject);
        }

        internal static Transform CreateEffector(CharacterMotionKeyframe keyframe, CharacterMotionConstraintType type)
        {
            if (keyframe == null) throw new ArgumentNullException(nameof(keyframe));
            HumanBodyBones bone = type switch
            {
                CharacterMotionConstraintType.LeftHand => HumanBodyBones.LeftHand,
                CharacterMotionConstraintType.RightHand => HumanBodyBones.RightHand,
                CharacterMotionConstraintType.LeftFoot => HumanBodyBones.LeftFoot,
                CharacterMotionConstraintType.RightFoot => HumanBodyBones.RightFoot,
                _ => throw new ArgumentOutOfRangeException(nameof(type)),
            };

            Transform existing = keyframe.GetEffector(type);
            string controlName = type.ToString();
            var namedChildren = new List<Transform>();
            for (int i = 0; i < keyframe.transform.childCount; i++)
            {
                Transform child = keyframe.transform.GetChild(i);
                if (child.name == controlName) namedChildren.Add(child);
            }

            Transform control = existing;
            bool created = false;
            if (control == null)
            {
                control = namedChildren.Count > 0 ? namedChildren[0] : null;
                if (control == null)
                {
                    var child = new GameObject(controlName);
                    Undo.RegisterCreatedObjectUndo(child, $"Create {type} Handle");
                    child.transform.SetParent(keyframe.transform, false);
                    control = child.transform;
                    created = true;
                }
            }

            // These names are reserved for generated controls. Preserve the referenced or
            // first reusable child, and remove only plain orphan duplicates. Objects with
            // components or children are left untouched to avoid deleting user content.
            foreach (Transform candidate in namedChildren)
            {
                if (candidate == null || candidate == control || !IsPlainControl(candidate)) continue;
                Undo.DestroyObjectImmediate(candidate.gameObject);
            }

            CharacterMotion motion = keyframe.GetComponentInParent<CharacterMotion>();
            CharacterMotionSkeleton skeleton = motion != null ? motion.GetComponentInChildren<CharacterMotionSkeleton>(true) : null;
            if (created && skeleton != null)
            {
                skeleton.AlignToKeyframe(keyframe);
                Transform source = skeleton.GuideAnimator != null
                    ? skeleton.GuideAnimator.GetBoneTransform(bone)
                    : null;
                if (source != null) control.SetPositionAndRotation(source.position, source.rotation);
            }
            if (existing != control) keyframe.SetEffector(type, control);
            return control;
        }

        private static bool IsPlainControl(Transform value)
            => value.childCount == 0 && value.GetComponents<Component>().Length == 1;

        private static Transform CloneHierarchy(Transform source, Transform parent)
        {
            var clone = new GameObject(source.name);
            clone.hideFlags |= HideFlags.DontSaveInEditor | HideFlags.DontSaveInBuild;
            clone.transform.SetParent(parent, false);
            clone.transform.localPosition = source.localPosition;
            clone.transform.localRotation = source.localRotation;
            clone.transform.localScale = source.localScale;
            for (int i = 0; i < source.childCount; i++) CloneHierarchy(source.GetChild(i), clone.transform);
            return clone.transform;
        }
    }
}

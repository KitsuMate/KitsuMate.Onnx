using System;
using UnityEngine;

namespace KitsuMate.Onnx.Motion
{
    /// <summary>Lightweight component for an EditorOnly transforms guide. It is never required by runtime compilation.</summary>
    [DisallowMultipleComponent]
    public sealed class CharacterMotionSkeleton : MonoBehaviour
    {
        [SerializeField] private CharacterMotion owner;
        [SerializeField] private Animator guideAnimator;
        [SerializeField] private CharacterMotionKeyframe loadedKeyframe;
        [SerializeField, HideInInspector] private Quaternion[] bindLocalRotations;
        [SerializeField, HideInInspector] private bool[] boneAvailability;
        [SerializeField, HideInInspector] private string[] bindTransformPaths;
        [SerializeField, HideInInspector] private Vector3[] bindLocalPositions;
        [SerializeField, HideInInspector] private Quaternion[] bindTransformLocalRotations;
        [SerializeField, HideInInspector] private Vector3[] bindLocalScales;
        [NonSerialized] private Transform[] previewTransforms;
        [NonSerialized] private Vector3[] previewLocalPositions;
        [NonSerialized] private Quaternion[] previewLocalRotations;
        [NonSerialized] private Vector3[] previewLocalScales;

        public CharacterMotion Owner => owner;
        public Animator GuideAnimator => guideAnimator;
        public CharacterMotionKeyframe LoadedKeyframe => loadedKeyframe;
        public ReadOnlyMemory<Quaternion> BindLocalRotations => bindLocalRotations ?? Array.Empty<Quaternion>();
        public bool HasCompleteBindPose => bindTransformPaths != null && bindTransformPaths.Length > 0 &&
                                           bindLocalPositions?.Length == bindTransformPaths.Length &&
                                           bindTransformLocalRotations?.Length == bindTransformPaths.Length &&
                                           bindLocalScales?.Length == bindTransformPaths.Length;
        public bool IsPreviewing => previewTransforms != null;

        public void Initialize(CharacterMotion motion, Animator animator)
        {
            owner = motion;
            guideAnimator = animator;
            CaptureBindPose();
        }

        public void CaptureBindPose()
        {
            Transform[] transforms = GetComponentsInChildren<Transform>(true);
            bindTransformPaths = new string[transforms.Length];
            bindLocalPositions = new Vector3[transforms.Length];
            bindTransformLocalRotations = new Quaternion[transforms.Length];
            bindLocalScales = new Vector3[transforms.Length];
            for (int i = 0; i < transforms.Length; i++)
            {
                bindTransformPaths[i] = RelativePath(transform, transforms[i]);
                bindLocalPositions[i] = transforms[i].localPosition;
                bindTransformLocalRotations[i] = transforms[i].localRotation;
                bindLocalScales[i] = transforms[i].localScale;
            }

            int count = (int)HumanBodyBones.LastBone;
            bindLocalRotations = new Quaternion[count];
            boneAvailability = new bool[count];
            for (int i = 0; i < count; i++)
            {
                Transform bone = guideAnimator != null ? guideAnimator.GetBoneTransform((HumanBodyBones)i) : null;
                bindLocalRotations[i] = bone != null ? bone.localRotation : Quaternion.identity;
                boneAvailability[i] = bone != null;
            }
        }

        public void LoadPose(CharacterMotionKeyframe keyframe)
        {
            if (keyframe == null) throw new ArgumentNullException(nameof(keyframe));
            if (!keyframe.Pose.IsCaptured) throw new InvalidOperationException("The keyframe has no captured pose.");
            ReadOnlySpan<Quaternion> rotations = keyframe.Pose.HumanoidLocalRotations.Span;
            ReadOnlySpan<bool> available = keyframe.Pose.HumanoidBoneAvailability.Span;
            for (int i = 0; i < rotations.Length; i++)
            {
                Transform bone = guideAnimator.GetBoneTransform((HumanBodyBones)i);
                if (bone != null && available[i]) bone.localRotation = rotations[i];
            }
            loadedKeyframe = keyframe;
        }

        public void ResetToBindPose()
        {
            StopPreview();
            if (HasCompleteBindPose)
            {
                for (int i = 0; i < bindTransformPaths.Length; i++)
                {
                    Transform value = string.IsNullOrEmpty(bindTransformPaths[i])
                        ? transform
                        : transform.Find(bindTransformPaths[i]);
                    if (value == null) continue;
                    value.localPosition = bindLocalPositions[i];
                    value.localRotation = bindTransformLocalRotations[i];
                    value.localScale = bindLocalScales[i];
                }
            }
            if (bindLocalRotations == null) return;
            for (int i = 0; i < bindLocalRotations.Length; i++)
            {
                Transform bone = guideAnimator.GetBoneTransform((HumanBodyBones)i);
                if (bone != null && boneAvailability[i]) bone.localRotation = bindLocalRotations[i];
            }
            loadedKeyframe = null;
        }

        private static string RelativePath(Transform root, Transform value)
        {
            if (value == root) return string.Empty;
            var parts = new System.Collections.Generic.Stack<string>();
            for (Transform current = value; current != null && current != root; current = current.parent)
                parts.Push(current.name);
            return string.Join("/", parts);
        }

        public void CapturePose(CharacterMotionKeyframe keyframe)
        {
            if (keyframe == null) throw new ArgumentNullException(nameof(keyframe));
            if (owner == null || guideAnimator == null) throw new InvalidOperationException("The motion skeleton is not initialized.");
            keyframe.Pose.Capture(guideAnimator, keyframe.transform, owner.AvatarSignature);
            loadedKeyframe = keyframe;
        }

        public bool HasUnsavedChanges(float angleTolerance = 0.05f)
        {
            if (loadedKeyframe == null || !loadedKeyframe.Pose.IsCaptured) return false;
            ReadOnlySpan<Quaternion> stored = loadedKeyframe.Pose.HumanoidLocalRotations.Span;
            ReadOnlySpan<bool> available = loadedKeyframe.Pose.HumanoidBoneAvailability.Span;
            for (int i = 0; i < stored.Length; i++)
            {
                Transform bone = guideAnimator.GetBoneTransform((HumanBodyBones)i);
                if (bone != null && available[i] && Quaternion.Angle(bone.localRotation, stored[i]) > angleTolerance) return true;
            }
            return false;
        }

        /// <summary>Samples a clip only onto this editor guide while preserving its prior transform state.</summary>
        public void Preview(AnimationClip clip, float time)
        {
            if (clip == null) throw new ArgumentNullException(nameof(clip));
            if (guideAnimator == null) throw new InvalidOperationException("The motion skeleton is not initialized.");
            if (IsPreviewing) RestorePreviewTransforms();
            else CapturePreviewTransforms();
            if (clip.humanMotion)
            {
                clip.SampleAnimation(gameObject, Mathf.Clamp(time, 0f, clip.length));
            }
            else
            {
                // Raw transform curves require Unity's legacy AnimationClip sampling mode.
                // Use a transient clone rather than changing the generated asset.
                AnimationClip samplingClip = Instantiate(clip);
                samplingClip.hideFlags = HideFlags.HideAndDontSave;
                samplingClip.legacy = true;
                try { samplingClip.SampleAnimation(gameObject, Mathf.Clamp(time, 0f, clip.length)); }
                finally
                {
                    if (Application.isPlaying) Destroy(samplingClip);
                    else DestroyImmediate(samplingClip);
                }
            }

            // Root curves are authored relative to the target Animator's parent. The guide
            // is parented below CharacterMotion, so reinterpret the sampled local root in
            // the target hierarchy before drawing its gizmos.
            Transform targetParent = owner != null && owner.TargetAnimator != null
                ? owner.TargetAnimator.transform.parent
                : null;
            if (targetParent != transform.parent && !clip.humanMotion)
            {
                Vector3 worldPosition = targetParent != null
                    ? targetParent.TransformPoint(transform.localPosition)
                    : transform.localPosition;
                Quaternion worldRotation = targetParent != null
                    ? targetParent.rotation * transform.localRotation
                    : transform.localRotation;
                transform.SetPositionAndRotation(worldPosition, worldRotation);
            }
        }

        /// <summary>Restores the guide state captured before preview began.</summary>
        public void StopPreview()
        {
            if (!IsPreviewing) return;
            RestorePreviewTransforms();
            previewTransforms = null;
            previewLocalPositions = null;
            previewLocalRotations = null;
            previewLocalScales = null;
        }

        private void CapturePreviewTransforms()
        {
            previewTransforms = GetComponentsInChildren<Transform>(true);
            int count = previewTransforms.Length;
            previewLocalPositions = new Vector3[count];
            previewLocalRotations = new Quaternion[count];
            previewLocalScales = new Vector3[count];
            for (int i = 0; i < count; i++)
            {
                previewLocalPositions[i] = previewTransforms[i].localPosition;
                previewLocalRotations[i] = previewTransforms[i].localRotation;
                previewLocalScales[i] = previewTransforms[i].localScale;
            }
        }

        private void RestorePreviewTransforms()
        {
            for (int i = 0; i < previewTransforms.Length; i++)
            {
                Transform value = previewTransforms[i];
                if (value == null) continue;
                value.localPosition = previewLocalPositions[i];
                value.localRotation = previewLocalRotations[i];
                value.localScale = previewLocalScales[i];
            }
        }

        private void OnDisable() => StopPreview();

        private void OnDrawGizmos()
        {
            if (guideAnimator == null) return;
            Gizmos.color = HasUnsavedChanges() ? Color.yellow : new Color(0.25f, 0.9f, 0.9f);
            Draw(HumanBodyBones.Hips, HumanBodyBones.Spine); Draw(HumanBodyBones.Spine, HumanBodyBones.Chest);
            Draw(HumanBodyBones.Chest, HumanBodyBones.Neck); Draw(HumanBodyBones.Neck, HumanBodyBones.Head);
            Draw(HumanBodyBones.Chest, HumanBodyBones.LeftShoulder); Draw(HumanBodyBones.LeftShoulder, HumanBodyBones.LeftUpperArm);
            Draw(HumanBodyBones.LeftUpperArm, HumanBodyBones.LeftLowerArm); Draw(HumanBodyBones.LeftLowerArm, HumanBodyBones.LeftHand);
            Draw(HumanBodyBones.Chest, HumanBodyBones.RightShoulder); Draw(HumanBodyBones.RightShoulder, HumanBodyBones.RightUpperArm);
            Draw(HumanBodyBones.RightUpperArm, HumanBodyBones.RightLowerArm); Draw(HumanBodyBones.RightLowerArm, HumanBodyBones.RightHand);
            Draw(HumanBodyBones.Hips, HumanBodyBones.LeftUpperLeg); Draw(HumanBodyBones.LeftUpperLeg, HumanBodyBones.LeftLowerLeg);
            Draw(HumanBodyBones.LeftLowerLeg, HumanBodyBones.LeftFoot); Draw(HumanBodyBones.Hips, HumanBodyBones.RightUpperLeg);
            Draw(HumanBodyBones.RightUpperLeg, HumanBodyBones.RightLowerLeg); Draw(HumanBodyBones.RightLowerLeg, HumanBodyBones.RightFoot);
        }

        private void Draw(HumanBodyBones a, HumanBodyBones b)
        {
            Transform first = guideAnimator.GetBoneTransform(a), second = guideAnimator.GetBoneTransform(b);
            if (first != null && second != null) Gizmos.DrawLine(first.position, second.position);
        }
    }
}

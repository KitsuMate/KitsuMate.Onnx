using System;
using UnityEditor;
using UnityEngine;

namespace KitsuMate.Onnx.Motion.Editor
{
    internal static class CharacterMotionSampling
    {
        internal static void Sample(CharacterMotionSkeleton skeleton, CharacterMotionKeyframe keyframe, AnimationClip clip, float time)
        {
            if (skeleton == null || keyframe == null || clip == null) throw new ArgumentNullException();
            Transform[] transforms = skeleton.GetComponentsInChildren<Transform>(true);
            var positions = new Vector3[transforms.Length];
            var rotations = new Quaternion[transforms.Length];
            var scales = new Vector3[transforms.Length];
            for (int i = 0; i < transforms.Length; i++)
            {
                positions[i] = transforms[i].localPosition;
                rotations[i] = transforms[i].localRotation;
                scales[i] = transforms[i].localScale;
            }

            bool started = false;
            try
            {
                if (!AnimationMode.InAnimationMode()) { AnimationMode.StartAnimationMode(); started = true; }
                AnimationMode.BeginSampling();
                try { AnimationMode.SampleAnimationClip(skeleton.gameObject, clip, Mathf.Clamp(time, 0f, clip.length)); }
                finally { AnimationMode.EndSampling(); }
                Undo.RecordObject(keyframe, "Sample Character Motion Pose");
                skeleton.CapturePose(keyframe);
                EditorUtility.SetDirty(keyframe);
            }
            finally
            {
                for (int i = 0; i < transforms.Length; i++)
                {
                    if (transforms[i] == null) continue;
                    transforms[i].localPosition = positions[i];
                    transforms[i].localRotation = rotations[i];
                    transforms[i].localScale = scales[i];
                }
                if (started && AnimationMode.InAnimationMode()) AnimationMode.StopAnimationMode();
            }
        }
    }
}

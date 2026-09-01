using System;
using UnityEditor;
using UnityEngine;

namespace KitsuMate.Onnx.Motion.Editor
{
    internal static class CharacterMotionSampling
    {
        internal static void Sample(CharacterMotionSkeleton skeleton, CharacterMotionKeyframe keyframe, AnimationClip clip, int frame)
        {
            if (skeleton == null || keyframe == null || clip == null) throw new ArgumentNullException();
            float frameRate = clip.frameRate > 0f ? clip.frameRate : 30f;
            int maximumFrame = Mathf.Max(0, Mathf.FloorToInt(clip.length * frameRate + 0.0001f));
            if (frame < 0 || frame > maximumFrame)
                throw new ArgumentOutOfRangeException(nameof(frame), $"Frame must be between 0 and {maximumFrame}.");
            float time = Mathf.Min(frame / frameRate, clip.length);
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
                // Animation clips may contain root curves, but authoring root position and
                // heading always come from the CharacterMotionKeyframe Transform.
                skeleton.AlignToKeyframe(keyframe);
                CharacterMotionAuthoringUtility.AlignControls(keyframe, skeleton, true);
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
            skeleton.LoadPose(keyframe);
            SceneView.RepaintAll();
        }
    }
}

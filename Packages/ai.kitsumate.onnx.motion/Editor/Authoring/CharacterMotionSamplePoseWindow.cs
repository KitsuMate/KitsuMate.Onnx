using System;
using UnityEditor;
using UnityEngine;

namespace KitsuMate.Onnx.Motion.Editor
{
    internal sealed class CharacterMotionSamplePoseWindow : EditorWindow
    {
        private CharacterMotionKeyframe keyframe;
        private AnimationClip clip;
        private int sourceFrame;

        internal static void Show(CharacterMotionKeyframe value)
        {
            var window = CreateInstance<CharacterMotionSamplePoseWindow>();
            window.titleContent = new GUIContent("Sample Pose");
            window.keyframe = value;
            window.minSize = window.maxSize = new Vector2(380f, 126f);
            window.position = new Rect(GUIUtility.GUIToScreenPoint(Event.current != null ? Event.current.mousePosition : Vector2.zero), window.minSize);
            window.ShowModalUtility();
        }

        private void OnGUI()
        {
            EditorGUILayout.Space(8f);
            clip = (AnimationClip)EditorGUILayout.ObjectField("Animation Clip", clip, typeof(AnimationClip), false);
            int maximumFrame = clip != null
                ? Mathf.Max(0, Mathf.FloorToInt(clip.length * Mathf.Max(clip.frameRate, 1f) + 0.0001f))
                : 0;
            sourceFrame = EditorGUILayout.IntSlider("Source Frame", Mathf.Clamp(sourceFrame, 0, maximumFrame), 0, maximumFrame);
            using (new EditorGUI.DisabledScope(clip == null || keyframe == null))
            using (new EditorGUILayout.HorizontalScope())
            {
                if (GUILayout.Button("Sample"))
                {
                    try
                    {
                        CharacterMotionSkeleton skeleton = CharacterMotionAuthoringUtility.PrepareKeyframe(keyframe, false);
                        CharacterMotionSampling.Sample(skeleton, keyframe, clip, sourceFrame);
                        Close();
                    }
                    catch (Exception exception)
                    {
                        Debug.LogException(exception);
                        EditorUtility.DisplayDialog("Sample Character Motion Pose", exception.Message, "OK");
                    }
                }
                if (GUILayout.Button("Cancel")) Close();
            }
        }
    }
}

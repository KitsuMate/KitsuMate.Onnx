using System;
using UnityEditor;
using UnityEngine;

namespace KitsuMate.Onnx.Motion.Editor
{
    [CustomEditor(typeof(CharacterMotionKeyframe))]
    internal sealed class CharacterMotionKeyframeEditor : UnityEditor.Editor
    {
        private AnimationClip sampleClip;
        private float sampleTime;

        public override void OnInspectorGUI()
        {
            serializedObject.Update();
            DrawDefaultInspector();
            serializedObject.ApplyModifiedProperties();
            var keyframe = (CharacterMotionKeyframe)target;
            CharacterMotion motion = keyframe.GetComponentInParent<CharacterMotion>();
            CharacterMotionSkeleton skeleton = motion != null ? motion.GetComponentInChildren<CharacterMotionSkeleton>(true) : null;

            EditorGUILayout.Space();
            EditorGUILayout.LabelField("Pose", EditorStyles.boldLabel);
            if (skeleton == null)
                EditorGUILayout.HelpBox("Create the Character Motion skeleton before editing poses.", MessageType.Info);
            else
            {
                if (skeleton.HasUnsavedChanges()) EditorGUILayout.HelpBox("The skeleton has changes that are not captured in this keyframe.", MessageType.Warning);
                using (new EditorGUILayout.HorizontalScope())
                {
                    using (new EditorGUI.DisabledScope(!keyframe.Pose.IsCaptured))
                        if (GUILayout.Button("Load Pose")) Run(() => { RecordSkeleton(skeleton, "Load Character Motion Pose"); skeleton.LoadPose(keyframe); });
                    if (GUILayout.Button("Capture Pose")) Run(() => { Undo.RecordObject(keyframe, "Capture Character Motion Pose"); skeleton.CapturePose(keyframe); EditorUtility.SetDirty(keyframe); });
                    if (GUILayout.Button("Reset Skeleton")) { RecordSkeleton(skeleton, "Reset Character Motion Skeleton"); skeleton.ResetToBindPose(); }
                }

                sampleClip = (AnimationClip)EditorGUILayout.ObjectField("Sample Clip", sampleClip, typeof(AnimationClip), false);
                sampleTime = EditorGUILayout.FloatField("Sample Time", sampleTime);
                using (new EditorGUI.DisabledScope(sampleClip == null))
                    if (GUILayout.Button("Sample Animation Frame")) Run(() => CharacterMotionSampling.Sample(skeleton, keyframe, sampleClip, sampleTime));
            }

            EditorGUILayout.Space();
            EditorGUILayout.LabelField("Effector Handles", EditorStyles.boldLabel);
            DrawEffectorButton(keyframe, CharacterMotionConstraintType.LeftHand);
            DrawEffectorButton(keyframe, CharacterMotionConstraintType.RightHand);
            DrawEffectorButton(keyframe, CharacterMotionConstraintType.LeftFoot);
            DrawEffectorButton(keyframe, CharacterMotionConstraintType.RightFoot);
        }

        private static void DrawEffectorButton(CharacterMotionKeyframe keyframe, CharacterMotionConstraintType type)
        {
            if ((keyframe.Constraints & type) == 0) return;
            if (keyframe.GetEffector(type) == null && GUILayout.Button("Create " + type + " Handle"))
                CharacterMotionSkeletonUtility.CreateEffector(keyframe, type);
        }

        private static void Run(Action action)
        {
            try { action(); }
            catch (Exception exception) { Debug.LogException(exception); EditorUtility.DisplayDialog("Character Motion", exception.Message, "OK"); }
        }

        private static void RecordSkeleton(CharacterMotionSkeleton skeleton, string operation)
        {
            Transform[] transforms = skeleton.GetComponentsInChildren<Transform>(true);
            Undo.RecordObjects(Array.ConvertAll(transforms, value => (UnityEngine.Object)value), operation);
        }
    }
}

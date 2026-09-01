using System;
using UnityEditor;
using UnityEngine;

namespace KitsuMate.Onnx.Motion.Editor
{
    [CustomEditor(typeof(CharacterMotionKeyframe))]
    internal sealed class CharacterMotionKeyframeEditor : UnityEditor.Editor
    {
        private CharacterMotionSceneHandles sceneHandles;

        private void OnEnable()
        {
            sceneHandles = new CharacterMotionSceneHandles((CharacterMotionKeyframe)target);
            var keyframe = (CharacterMotionKeyframe)target;
            if (keyframe.RequiresPose) Run(() => CharacterMotionAuthoringUtility.PrepareKeyframe(keyframe));
        }

        public override void OnInspectorGUI()
        {
            var keyframe = (CharacterMotionKeyframe)target;
            CharacterMotion motion = keyframe.GetComponentInParent<CharacterMotion>();

            EditorGUILayout.LabelField("Character Motion Pose", EditorStyles.boldLabel);
            using (new EditorGUI.DisabledScope(true)) EditorGUILayout.IntField("Frame", keyframe.Frame);

            EditorGUILayout.Space(4f);
            EditorGUILayout.LabelField("Root", EditorStyles.boldLabel);
            using (new EditorGUILayout.HorizontalScope())
            {
                DrawConstraintButton(keyframe, CharacterMotionConstraintType.RootPosition, "Root Position");
                DrawConstraintButton(keyframe, CharacterMotionConstraintType.RootHeading, "Root Heading");
            }

            EditorGUILayout.Space(4f);
            EditorGUILayout.LabelField("Pose", EditorStyles.boldLabel);
            DrawConstraintButton(keyframe, CharacterMotionConstraintType.FullBodyPose, "Full Pose");

            EditorGUILayout.Space(4f);
            EditorGUILayout.LabelField("Inverse Kinematics", EditorStyles.boldLabel);
            using (new EditorGUILayout.HorizontalScope())
            {
                DrawConstraintButton(keyframe, CharacterMotionConstraintType.LeftHand, "Left Hand");
                DrawConstraintButton(keyframe, CharacterMotionConstraintType.RightHand, "Right Hand");
            }
            using (new EditorGUILayout.HorizontalScope())
            {
                DrawConstraintButton(keyframe, CharacterMotionConstraintType.LeftFoot, "Left Foot");
                DrawConstraintButton(keyframe, CharacterMotionConstraintType.RightFoot, "Right Foot");
            }

            EditorGUILayout.Space(6f);
            DrawStatus(keyframe, motion);
            using (new EditorGUILayout.HorizontalScope())
            {
                if (GUILayout.Button("Reset Pose…")) Run(() =>
                {
                    if (EditorUtility.DisplayDialog("Reset Character Motion Pose",
                            "Reset the full pose, IK targets, and poles to their defaults?", "Reset", "Cancel"))
                        CharacterMotionAuthoringUtility.ResetPose(keyframe);
                });
                if (GUILayout.Button("Sample Pose…")) Run(() => CharacterMotionSamplePoseWindow.Show(keyframe));
            }
        }

        private static void DrawConstraintButton(CharacterMotionKeyframe keyframe, CharacterMotionConstraintType type, string label)
        {
            bool current = (keyframe.Constraints & type) != 0;
            bool next = GUILayout.Toggle(current, label, "Button", GUILayout.MinHeight(24f));
            if (next == current) return;
            Run(() => CharacterMotionAuthoringUtility.SetConstraint(keyframe, type, next));
        }

        private static void DrawStatus(CharacterMotionKeyframe keyframe, CharacterMotion motion)
        {
            Vector3 scale = keyframe.transform.lossyScale;
            if (!CharacterMotionSpace.IsFinite(scale) ||
                Mathf.Max(Mathf.Abs(scale.x - 1f),
                    Mathf.Max(Mathf.Abs(scale.y - 1f), Mathf.Abs(scale.z - 1f))) > 1e-3f)
                EditorGUILayout.HelpBox(
                    "Keyframe scale must remain one. This Transform supplies root position and rotation only.",
                    MessageType.Error);
            else if (!keyframe.RequiresPose)
                EditorGUILayout.HelpBox("Enable Full Pose or an IK target to author skeletal pose data.", MessageType.Info);
            else if (!keyframe.Pose.IsCaptured)
                EditorGUILayout.HelpBox("This pose has not been initialized.", MessageType.Warning);
            else if (!keyframe.Pose.UsesCurrentRootSpace)
                EditorGUILayout.HelpBox(
                    "This pose predates keyframe-relative guide roots. Use Recalculate on Character Motion.",
                    MessageType.Warning);
            else if (motion != null && !keyframe.Pose.MatchesAvatar(motion.AvatarSignature))
                EditorGUILayout.HelpBox("This pose was captured for a different avatar. Use Recalculate on Character Motion.", MessageType.Warning);
            else if (!CharacterMotionAuthoringUtility.HasFullPose(keyframe) &&
                     (keyframe.Constraints & (CharacterMotionConstraintType.LeftHand |
                                              CharacterMotionConstraintType.RightHand |
                                              CharacterMotionConstraintType.LeftFoot |
                                              CharacterMotionConstraintType.RightFoot)) != 0)
                EditorGUILayout.HelpBox(
                    "The body is unconstrained. Kimodo may sit, kneel, lean, or otherwise reposition the hidden body " +
                    "to satisfy these targets. End-effectors are not limited by the reference limb length.",
                    MessageType.Info);
            else
                EditorGUILayout.HelpBox("Pose changes are captured automatically.", MessageType.Info);
        }

        private void OnSceneGUI() => sceneHandles?.OnSceneGUI();

        private void OnDisable() => sceneHandles?.FlushPendingPose();

        private static void Run(Action action)
        {
            try { action(); }
            catch (Exception exception)
            {
                Debug.LogException(exception);
                EditorUtility.DisplayDialog("Character Motion", exception.Message, "OK");
            }
        }
    }
}

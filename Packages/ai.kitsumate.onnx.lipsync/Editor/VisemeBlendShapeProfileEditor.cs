using UnityEditor;
using UnityEngine;

namespace KitsuMate.Onnx.LipSync.Editor
{
    [CustomEditor(typeof(VisemeBlendShapeProfile))]
    internal class VisemeBlendShapeProfileEditor : UnityEditor.Editor
    {
        public override void OnInspectorGUI()
        {
            var profile = (VisemeBlendShapeProfile)target;

            EditorGUILayout.Space();
            EditorGUILayout.LabelField("Apply Preset", EditorStyles.boldLabel);
            EditorGUILayout.HelpBox(
                "Applying a preset will overwrite all current mappings.",
                MessageType.Info);

            EditorGUILayout.BeginHorizontal();

            if (GUILayout.Button("VRM / VRoid"))
            {
                Undo.RecordObject(profile, "Apply VRM Preset");
                profile.ApplyPreset(VisemeProfilePreset.VrmVroid);
                EditorUtility.SetDirty(profile);
            }

            if (GUILayout.Button("Daz3D"))
            {
                Undo.RecordObject(profile, "Apply Daz3D Preset");
                profile.ApplyPreset(VisemeProfilePreset.Daz3D);
                EditorUtility.SetDirty(profile);
            }

            if (GUILayout.Button("FACS"))
            {
                Undo.RecordObject(profile, "Apply FACS Preset");
                profile.ApplyPreset(VisemeProfilePreset.FACS);
                EditorUtility.SetDirty(profile);
            }

            if (GUILayout.Button("ARKit"))
            {
                Undo.RecordObject(profile, "Apply ARKit Preset");
                profile.ApplyPreset(VisemeProfilePreset.ARKit);
                EditorUtility.SetDirty(profile);
            }

            if (GUILayout.Button("VRChat"))
            {
                Undo.RecordObject(profile, "Apply VRChat Preset");
                profile.ApplyPreset(VisemeProfilePreset.VRChat);
                EditorUtility.SetDirty(profile);
            }

            if (GUILayout.Button("Clear"))
            {
                Undo.RecordObject(profile, "Clear Mappings");
                profile.ApplyPreset(VisemeProfilePreset.Custom);
                EditorUtility.SetDirty(profile);
            }

            EditorGUILayout.EndHorizontal();
            EditorGUILayout.Space();

            DrawDefaultInspector();
        }
    }
}

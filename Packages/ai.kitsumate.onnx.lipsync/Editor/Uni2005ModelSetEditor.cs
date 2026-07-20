#if UNITY_EDITOR
using KitsuMate.Onnx.Editor;
using KitsuMate.Onnx.LipSync.Uni2005;
using UnityEditor;
using UnityEngine;

namespace KitsuMate.Onnx.LipSync.Editor
{
    [CustomEditor(typeof(Uni2005ModelSet))]
    public sealed class Uni2005ModelSetEditor : UnityEditor.Editor
    {
        public override void OnInspectorGUI()
        {
            serializedObject.Update(); var set = (Uni2005ModelSet)target;
            ModelSetEditorUi.Header(set, "Uni2005 Lip Sync Model Set");
            EditorGUILayout.LabelField("ONNX Models", EditorStyles.boldLabel);
            EditorGUILayout.PropertyField(serializedObject.FindProperty("_acousticModelSource"), new GUIContent("Acoustic Model"));
            EditorGUILayout.LabelField("Auxiliary Files", EditorStyles.boldLabel);
            EditorGUILayout.PropertyField(serializedObject.FindProperty("_vocabulary"));
            EditorGUILayout.PropertyField(serializedObject.FindProperty("_variantDescription"));
            serializedObject.ApplyModifiedProperties(); ModelSetEditorUi.Validation(set);
        }
    }
}
#endif

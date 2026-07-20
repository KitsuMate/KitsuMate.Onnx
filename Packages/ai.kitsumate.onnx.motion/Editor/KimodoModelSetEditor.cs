#if UNITY_EDITOR
using KitsuMate.Onnx.Editor;
using KitsuMate.Onnx.Motion.Kimodo;
using UnityEditor;

namespace KitsuMate.Onnx.Motion.Editor
{
    [CustomEditor(typeof(KimodoModelSet))]
    public sealed class KimodoModelSetEditor : UnityEditor.Editor
    {
        public override void OnInspectorGUI()
        {
            serializedObject.Update();
            var modelSet = (KimodoModelSet)target;
            ModelSetEditorUi.Header(modelSet, "Kimodo Motion Model Set");
            EditorGUILayout.Space();
            EditorGUILayout.LabelField("ONNX Models", EditorStyles.boldLabel);
            EditorGUILayout.PropertyField(serializedObject.FindProperty("motionModel"), new UnityEngine.GUIContent("Motion Model"));
            EditorGUILayout.Space();
            EditorGUILayout.LabelField("Embedding Contract", EditorStyles.boldLabel);
            EditorGUILayout.PropertyField(serializedObject.FindProperty("requiredEmbeddingModelSet"),
                new UnityEngine.GUIContent("Required Embedding Model"));
            serializedObject.ApplyModifiedProperties();
            ModelSetEditorUi.Validation(modelSet);
        }
    }
}
#endif

#if UNITY_EDITOR
using KitsuMate.Onnx.Editor;
using KitsuMate.Onnx.Embeddings;
using UnityEditor;
using UnityEngine;

namespace KitsuMate.Onnx.Embeddings.Editor
{
    [CustomEditor(typeof(Llm2VecModelSet))]
    public sealed class Llm2VecModelSetEditor : UnityEditor.Editor
    {
        public override void OnInspectorGUI()
        {
            serializedObject.Update(); var set = (Llm2VecModelSet)target;
            ModelSetEditorUi.Header(set, "LLM2Vec Model Set");
            EditorGUILayout.LabelField("ONNX Models", EditorStyles.boldLabel);
            EditorGUILayout.PropertyField(serializedObject.FindProperty("encoderSource"), new GUIContent("Encoder"));
            EditorGUILayout.LabelField("Tokenizer and Contract", EditorStyles.boldLabel);
            foreach (string p in new[] { "tokenizerJson", "tokenizerConfig", "sequenceLength", "embeddingDimension" }) EditorGUILayout.PropertyField(serializedObject.FindProperty(p));
            serializedObject.ApplyModifiedProperties(); ModelSetEditorUi.Validation(set);
        }
    }
}
#endif

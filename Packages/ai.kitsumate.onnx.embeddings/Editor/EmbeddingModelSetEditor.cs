#if UNITY_EDITOR
using KitsuMate.Onnx.Editor;
using KitsuMate.Onnx.Embeddings;
using UnityEditor;
using UnityEngine;

namespace KitsuMate.Onnx.Embeddings.Editor
{
    [CustomEditor(typeof(TextEmbeddingModelSet))]
    public sealed class EmbeddingModelSetEditor : UnityEditor.Editor
    {
        public override void OnInspectorGUI()
        {
            serializedObject.Update(); var set = (TextEmbeddingModelSet)target;
            ModelSetEditorUi.Header(set, "Text Embedding Model Set");
            EditorGUILayout.LabelField("ONNX Models", EditorStyles.boldLabel);
            EditorGUILayout.PropertyField(serializedObject.FindProperty("_embeddingModelSource"), new GUIContent("Embedding Model"));
            EditorGUILayout.LabelField("Tokenizer", EditorStyles.boldLabel);
            EditorGUILayout.PropertyField(serializedObject.FindProperty("_vocabulary"));
            EditorGUILayout.PropertyField(serializedObject.FindProperty("_tokenizerModel"));
            EditorGUILayout.LabelField("Configuration", EditorStyles.boldLabel);
            foreach (string p in new[] { "_embeddingDimension", "_maxSequenceLength", "_useMeanPooling", "_normalizeEmbeddings" }) EditorGUILayout.PropertyField(serializedObject.FindProperty(p));
            serializedObject.ApplyModifiedProperties(); ModelSetEditorUi.Validation(set);
        }
    }
}
#endif

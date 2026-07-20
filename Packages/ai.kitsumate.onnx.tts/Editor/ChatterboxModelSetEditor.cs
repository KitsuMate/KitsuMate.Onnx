#if UNITY_EDITOR
using KitsuMate.Onnx.Editor;
using UnityEditor;
using UnityEngine;

namespace KitsuMate.Onnx.Tts.Chatterbox.Editor
{
    [CustomEditor(typeof(ChatterboxModelSet))]
    public sealed class ChatterboxModelSetEditor : UnityEditor.Editor
    {
        public override void OnInspectorGUI()
        {
            serializedObject.Update(); var set = (ChatterboxModelSet)target;
            ModelSetEditorUi.Header(set, "Chatterbox Model Set");
            EditorGUILayout.LabelField("ONNX Models", EditorStyles.boldLabel);
            foreach ((string property, string label) in new[] { ("_speechEncoderSource", "Speech Encoder"), ("_embedTokensSource", "Embed Tokens"), ("_languageModelSource", "Language Model"), ("_conditionalDecoderSource", "Conditional Decoder") })
                EditorGUILayout.PropertyField(serializedObject.FindProperty(property), new GUIContent(label));
            EditorGUILayout.LabelField("Auxiliary Files", EditorStyles.boldLabel);
            foreach (string p in new[] { "_tokenizer", "_cangjieMapping", "_defaultVoice" }) EditorGUILayout.PropertyField(serializedObject.FindProperty(p));
            serializedObject.ApplyModifiedProperties(); ModelSetEditorUi.Validation(set);
        }
    }
}
#endif
